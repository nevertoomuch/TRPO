#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cblas.h"

void my_ssyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
              const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
              const float alpha, const float *A, const int lda,
              const float beta, float *C, const int ldc);

void my_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
              const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
              const double alpha, const double *A, const int lda,
              const double beta, double *C, const int ldc);

int test_ssyrk() {
    int N = 4, K = 3;
    int lda = K;  
    int ldc = N;
    float alpha = 1.0f, beta = 0.0f;
    float A[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
    float C_my[16] = {0}, C_blas[16] = {0};
    
    my_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, K, alpha, A, lda, beta, C_my, ldc);
    cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, K, alpha, A, lda, beta, C_blas, ldc);
    
    for (int i = 0; i < N * N; i++)
        if (fabsf(C_my[i] - C_blas[i]) > 1e-4f) {
            printf("FAIL: SSYRK at index %d: %f vs %f\n", i, C_my[i], C_blas[i]);
            return 1;
        }
    
    printf("PASS: SSYRK\n");
    return 0;
}

int test_dsyrk() {
    int N = 4, K = 3;
    int lda = K;
    int ldc = N;
    double alpha = 1.0, beta = 0.0;
    double A[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
    double C_my[16] = {0}, C_blas[16] = {0};
    
    my_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, K, alpha, A, lda, beta, C_my, ldc);
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, K, alpha, A, lda, beta, C_blas, ldc);
    
    for (int i = 0; i < N * N; i++)
        if (fabs(C_my[i] - C_blas[i]) > 1e-8) {
            printf("FAIL: DSYRK at index %d: %f vs %f\n", i, C_my[i], C_blas[i]);
            return 1;
        }
    
    printf("PASS: DSYRK\n");
    return 0;
}

int main() {
    printf("Running tests...\n");
    int fail = test_ssyrk() + test_dsyrk();
    printf("%s\n", fail ? "TESTS FAILED" : "TESTS PASSED");
    return 1;
}
