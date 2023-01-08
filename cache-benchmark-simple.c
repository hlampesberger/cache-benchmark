#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>


#define N 1024

// Source: https://ubuntuforums.org/showthread.php?t=1717717&p=10618266#post10618266
double randfrom(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void print_matrix(double (*M)[N])
{
    int i, j;
    for (i = 0; i < N; i++) {
        for(j = 0; j < N; j++)
            printf("%10.4f", M[i][j]);   
        printf("\n");
    }
}


int main(int argc, char *argv[])
{
    int i, j, k;
    clock_t start, end, substart, subend;

    double (*A)[N] = malloc(sizeof(double[N][N]));
    double (*B)[N] = malloc(sizeof(double[N][N]));
    double (*C)[N] = malloc(sizeof(double[N][N]));
    double (*Bt)[N] = malloc(sizeof(double[N][N]));


    if (!A || !B || !C || !Bt) {
        printf("Memory error.\n");
        if (A) free(A);
        if (B) free(B);
        if (C) free(C);
        if (Bt) free(Bt);
        exit(1);
    }


    printf("Populating with random numbers in shapes [%d, %d] ...", N, N);

    srand(time(NULL));

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = randfrom(-1., 1.);
            B[i][j] = randfrom(-1., 1.);
            C[i][j] = 0.0;
        }
    }
    
    printf("done.\n");


    // TEST 1
    printf("Classic matrix multiplication (ijk): ");
    start = clock();

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];

    end = clock();
    printf("%.2fs\n", (double)(end - start) / CLOCKS_PER_SEC);



    // TEST 2
    printf("Transposed matrix multiplication: ");
    memset(C, 0, sizeof(double[N][N]));
    start = clock();

    // transpose B first
    substart = clock();
    for (i = 0; i < N; i++)
        for (j = 0 ; j < N; j++)
            Bt[j][i] = B[i][j];
    subend = clock();
    printf("(copy %.2fs) ", (double)(subend - substart) / CLOCKS_PER_SEC);

    // multiply with transposed field
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++)
                C[i][j] += A[i][k] * Bt[j][k];

    end = clock();
    printf("%.2fs\n", (double)(end - start) / CLOCKS_PER_SEC);



    // TEST 3
    printf("Classic matrix multiplication with changed loop order (ikj): ");
    memset(C, 0, sizeof(double[N][N]));
    start = clock();

    for (i = 0; i < N; i++)
        for (k = 0; k < N; k++)
            for (j = 0; j < N; j++)
                C[i][j] += A[i][k] * B[k][j];

    end = clock();
    printf("%.2fs\n", (double)(end - start) / CLOCKS_PER_SEC);




    free(A);
    free(B);
    free(Bt);
    free(C);

    return 0;
}
