#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cblas.h"
#include <omp.h>


void my_ssyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
              const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
              const float alpha, const float *A, const int lda,
              const float beta, float *C, const int ldc) {
    
    if (Trans == CblasNoTrans) {
        if (Uplo == CblasLower) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j <= i; j++) {
                    float sum = 0.0f;
                    for (int l = 0; l < K; l++) {
                        sum += A[i * lda + l] * A[j * lda + l];
                    }
                    C[i * ldc + j] = beta * C[i * ldc + j] + alpha * sum;
                }
            }
        } else {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; i++) {
                for (int j = i; j < N; j++) {
                    float sum = 0.0f;
                    for (int l = 0; l < K; l++) {
                        sum += A[i * lda + l] * A[j * lda + l];
                    }
                    C[i * ldc + j] = beta * C[i * ldc + j] + alpha * sum;
                }
            }
        }
    } else {
        if (Uplo == CblasLower) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j <= i; j++) {
                    float sum = 0.0f;
                    for (int l = 0; l < K; l++) {
                        sum += A[l * lda + i] * A[l * lda + j];
                    }
                    C[i * ldc + j] = beta * C[i * ldc + j] + alpha * sum;
                }
            }
        } else {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; i++) {
                for (int j = i; j < N; j++) {
                    float sum = 0.0f;
                    for (int l = 0; l < K; l++) {
                        sum += A[l * lda + i] * A[l * lda + j];
                    }
                    C[i * ldc + j] = beta * C[i * ldc + j] + alpha * sum;
                }
            }
        }
    }
}

void my_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
              const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
              const double alpha, const double *A, const int lda,
              const double beta, double *C, const int ldc) {
    
    if (Trans == CblasNoTrans) {
        if (Uplo == CblasLower) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j <= i; j++) {
                    double sum = 0.0;
                    for (int l = 0; l < K; l++) {
                        sum += A[i * lda + l] * A[j * lda + l];
                    }
                    C[i * ldc + j] = beta * C[i * ldc + j] + alpha * sum;
                }
            }
        } else {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; i++) {
                for (int j = i; j < N; j++) {
                    double sum = 0.0;
                    for (int l = 0; l < K; l++) {
                        sum += A[i * lda + l] * A[j * lda + l];
                    }
                    C[i * ldc + j] = beta * C[i * ldc + j] + alpha * sum;
                }
            }
        }
    } else {
        if (Uplo == CblasLower) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j <= i; j++) {
                    double sum = 0.0;
                    for (int l = 0; l < K; l++) {
                        sum += A[l * lda + i] * A[l * lda + j];
                    }
                    C[i * ldc + j] = beta * C[i * ldc + j] + alpha * sum;
                }
            }
        } else {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; i++) {
                for (int j = i; j < N; j++) {
                    double sum = 0.0;
                    for (int l = 0; l < K; l++) {
                        sum += A[l * lda + i] * A[l * lda + j];
                    }
                    C[i * ldc + j] = beta * C[i * ldc + j] + alpha * sum;
                }
            }
        }
    }
}