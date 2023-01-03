#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

void print_help(char *prog_name)
{
    printf("Random matrix multiplication for demonstrating cache locality.\n");
    printf("A x B = C\n\n");
    printf("Usage: %s dim1 dim2 dim3\n", prog_name);
    printf("  A has shape (dim1, dim2)\n");
    printf("  B has shape (dim2, dim3)\n");
    printf("  C has shape (dim1, dim3)\n");
    printf("when no args are given, dim1 = dim2 = dim3 = 1024 is assumed\n");
    printf("when a single argument is given, dim1 = dim2 = dim3 = arg\n");
}


int get_param(int idx, char *argv[])
{
    long n;
    char * endp;

    n = strtol(argv[idx], &endp, 10);
    if (endp == argv[idx] || n <= 0 || n > INT_MAX) {
        print_help(argv[0]);
        printf("\nInvalid value %s\n", argv[idx]);
        exit(1);
    }
    return n;
}

// Source: https://ubuntuforums.org/showthread.php?t=1717717&p=10618266#post10618266
double randfrom(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void print_matrix(double * M, int dim1, int dim2)
{
    int i, j;
    for (i = 0; i < dim1; i++) {
        for(j = 0; j < dim2; j++)
            printf("%10.4f", M[i * dim2 + j]);   
        printf("\n");
    }
}



void multiplication_ijk(double * A, double * B, double * C, 
                        int dim1, int dim2, int dim3)
{
    int i, j, k;
    for (i = 0; i < dim1; i++)
        for (j = 0; j < dim3; j++)
            for (k = 0; k < dim2; k++)
                // C[i, j] += A[i, k] * B[k, j]
                // adapt to singular dimension of arrays
                C[i * dim3 + j] += A[i * dim2 + k] * B[k * dim3 + j];
}

void multiplication_ikj(double * A, double * B, double * C, 
                        int dim1, int dim2, int dim3)
{
    int i, j, k;
    for (i = 0; i < dim1; i++)
        for (k = 0; k < dim2; k++)
            for (j = 0; j < dim3; j++)
                // C[i, j] += A[i, k] * B[k, j]
                // adapt to singular dimension of arrays
                C[i * dim3 + j] += A[i * dim2 + k] * B[k * dim3 + j];
}

void multiplication_kij(double * A, double * B, double * C, 
                        int dim1, int dim2, int dim3)
{
    int i, j, k;
    for (k = 0; k < dim2; k++)
        for (i = 0; i < dim1; i++)
            for (j = 0; j < dim3; j++)
                // C[i, j] += A[i, k] * B[k, j]
                // adapt to singular dimension of arrays
                C[i * dim3 + j] += A[i * dim2 + k] * B[k * dim3 + j];
}

void transposed_multiplication(double * A, double * B, double * C, 
                               int dim1, int dim2, int dim3)
{
    int i, j, k;
    clock_t start, end;

    double * Bt = malloc(dim1 * dim2 * sizeof(double));
    if (!Bt) {
        printf("Memory error ");
        return;
    }

    start = clock();
    // we transpose B to Bnew
    for (i = 0; i < dim2; i++)
        for (j = 0 ; j < dim3; j++)
            // Bnew[j, i] = B[i, j]
            // new shape [dim3, dim2]
            Bt[j * dim2 + i] = B[i * dim3 + j];
    end = clock();
    printf("(copy %.2fs) ", (double)(end - start) / CLOCKS_PER_SEC);

    // multiply with transposed field in better

    for (i = 0; i < dim1; i++)
        for (j = 0; j < dim3; j++)
            for (k = 0; k < dim2; k++)
                // C[i, j] += A[i, k] * Bnew[j, k]
                C[i * dim3 + j] += A[i * dim2 + k] * Bt[j * dim2 + k];

    free(Bt);
}


int main(int argc, char *argv[])
{
    int dim1, dim2, dim3;
    int i, j, k;
    double * A;
    double * B;
    double * C;
    clock_t start, end;


    switch (argc) {
        case 1:
            // called with no arguments
            // assume default case
            dim1 = dim2 = dim3 = 1024;
            break;
        case 2:
            // read num and set all parameters

            dim1 = dim2 = dim3 = get_param(1, argv);
            break;

        case 4:
            // read every dim individually
            dim1 = get_param(1, argv);
            dim2 = get_param(2, argv);
            dim3 = get_param(3, argv);
            break;
        
        default:
            print_help(argv[0]);
            printf("\nInvalid number of arguments\n");
            exit(1);

    }

    // populting the double arrays with random numbers
    printf("Setting shapes: A[%d, %d] x B[%d, %d] = C[%d, %d]\n", 
           dim1, dim2, dim2, dim3, dim1, dim3);

    srand(time(NULL));

    A = malloc(dim1 * dim2 * sizeof(double));
    B = malloc(dim2 * dim3 * sizeof(double));
    C = calloc(1, dim1 * dim3 * sizeof(double));  // values are 0.0

    if (A && B && C) {
        for (i = 0; i < dim1; i++)
            for (j = 0; j < dim2; j++)
                A[i * dim2 + j] = randfrom(-1., 1.);
        for (i = 0; i < dim2; i++)
            for (j = 0; j < dim3; j++)
                B[i * dim3 + j] = randfrom(-1., 1.);

    } else {
        // one of the mallocs failed
        printf("Memory error.\n");
        if (A) free(A);
        if (B) free(B);
        if (C) free(C);
        exit(1);
    }
    printf("Matrices are now populated with random double numbers.\n");

    // TEST 1
    printf("Classic matrix multiplication (ijk): ");
    start = clock();
    multiplication_ijk(A, B, C, dim1, dim2, dim3);
    end = clock();
    printf("%.2fs\n", (double)(end - start) / CLOCKS_PER_SEC);


    // TEST 2, reset C before ...
    memset(C, 0, dim1 * dim3 * sizeof(double));
    printf("Transposed matrix multiplication: ");
    start = clock();
    transposed_multiplication(A, B, C, dim1, dim2, dim3);
    end = clock();
    printf("%.2fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    // TEST 3, reset C before ...
    memset(C, 0, dim1 * dim3 * sizeof(double));
    printf("Classic matrix multiplication (ikj): ");
    start = clock();
    multiplication_ikj(A, B, C, dim1, dim2, dim3);
    end = clock();
    printf("%.2fs\n", (double)(end - start) / CLOCKS_PER_SEC);


    // TEST 3, reset C before ...
    memset(C, 0, dim1 * dim3 * sizeof(double));
    printf("Classic matrix multiplication (ikj): ");
    start = clock();
    multiplication_kij(A, B, C, dim1, dim2, dim3);
    end = clock();
    printf("%.2fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    // cleanup
    free(A);
    free(B);
    free(C);
    return 0;
}
