#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "cblas.h"

void my_ssyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
              const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
              const float alpha, const float *A, const int lda,
              const float beta, float *C, const int ldc);

void my_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
              const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
              const double alpha, const double *A, const int lda,
              const double beta, double *C, const int ldc);

void init_matrix_float(float *A, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

void init_matrix_double(double *A, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        A[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
}

double benchmark_ssyrk(int N, int K, int use_my_impl, int num_threads) {
    int lda = N;
    int ldc = N;
    float alpha = 1.0f, beta = 0.0f;
    
    float *A = (float*)malloc(lda * K * sizeof(float));
    float *C = (float*)malloc(ldc * N * sizeof(float));
    
    init_matrix_float(A, lda, K);
    
    omp_set_num_threads(num_threads);
    
    if (use_my_impl) {
        my_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, K, 
                 alpha, A, lda, beta, C, ldc);
    } else {
        cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, K,
                    alpha, A, lda, beta, C, ldc);
    }
    
    double start = omp_get_wtime();
    
    if (use_my_impl) {
        my_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, K,
                 alpha, A, lda, beta, C, ldc);
    } else {
        cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, K,
                    alpha, A, lda, beta, C, ldc);
    }
    
    double end = omp_get_wtime();
    
    free(A);
    free(C);
    
    return (end - start);
}

double benchmark_dsyrk(int N, int K, int use_my_impl, int num_threads) {
    int lda = N;
    int ldc = N;
    double alpha = 1.0, beta = 0.0;
    
    double *A = (double*)malloc(lda * K * sizeof(double));
    double *C = (double*)malloc(ldc * N * sizeof(double));
    
    init_matrix_double(A, lda, K);
    
    omp_set_num_threads(num_threads);
    
    if (use_my_impl) {
        my_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, K,
                 alpha, A, lda, beta, C, ldc);
    } else {
        cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, K,
                    alpha, A, lda, beta, C, ldc);
    }
    
    double start = omp_get_wtime();
    
    if (use_my_impl) {
        my_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, K,
                 alpha, A, lda, beta, C, ldc);
    } else {
        cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, K,
                    alpha, A, lda, beta, C, ldc);
    }
    
    double end = omp_get_wtime();
    
    free(A);
    free(C);
    
    return (end - start);
}

double geometric_mean(double *values, int n) {
    double sum_log = 0.0;
    for (int i = 0; i < n; i++) {
        sum_log += log(values[i]);
    }
    return exp(sum_log / n);
}

void run_ssyrk_performance_tests() {
    printf("\nSSYRK\n");
    
    int N = 2048;
    int K = 2048;
    int iterations = 10;
    int thread_counts[] = {1, 2, 4, 8, 16};
    int num_thread_tests = 5;
    
    printf("Размер массива: %d x %d\n", N, N);
    printf("Количество итераций: %d\n\n", iterations);
    
    for (int t = 0; t < num_thread_tests; t++) {
        int threads = thread_counts[t];
        double perf_ratios[10];
        
        printf("Потоки: %d\n", threads);
        printf("%-6s %-15s %-15s %-15s %-10s\n", "Запуск", "Реализация (с)", "OpenBLAS (с)", "Производит. %", "Отношение");
        
        for (int run = 0; run < iterations; run++) {
            double time_my = benchmark_ssyrk(N, K, 1, threads);
            double time_blas = benchmark_ssyrk(N, K, 0, threads);
            
            double perf_ratio = (time_blas / time_my) * 100.0;
            perf_ratios[run] = perf_ratio;
            
            printf("%-6d %-15.6f %-15.6f %-15.2f %-10.2f\n", 
                   run + 1, time_my, time_blas, perf_ratio, time_my / time_blas);
        }
        
        double geo_mean = geometric_mean(perf_ratios, iterations);
        
        printf("\nСреднее геометрическое производительности: %.2f%%\n", geo_mean);
        printf("Ускорение относительно OpenBLAS: %.2fx\n\n", geo_mean / 100.0);
    }
}

void run_dsyrk_performance_tests() {
    printf("\nDSYRK\n");
    
    int N = 2048;
    int K = 2048;
    int iterations = 10;
    int thread_counts[] = {1, 2, 4, 8, 16};
    int num_thread_tests = 5;
    
    printf("Размер массива: %d x %d\n", N, N);
    printf("Количество итераций: %d\n\n", iterations);
    
    for (int t = 0; t < num_thread_tests; t++) {
        int threads = thread_counts[t];
        double perf_ratios[10];
        
        printf("Потоки: %d\n", threads);
        printf("%-6s %-15s %-15s %-15s %-10s\n", "Запуск", "Реализация (с)", "OpenBLAS (с)", "Производит. %", "Отношение");
        
        for (int run = 0; run < iterations; run++) {
            double time_my = benchmark_dsyrk(N, K, 1, threads);
            double time_blas = benchmark_dsyrk(N, K, 0, threads);
            
            double perf_ratio = (time_blas / time_my) * 100.0;
            perf_ratios[run] = perf_ratio;
            
            printf("%-6d %-15.6f %-15.6f %-15.2f %-10.2f\n", 
                   run + 1, time_my, time_blas, perf_ratio, time_my / time_blas);
        }
        
        double geo_mean = geometric_mean(perf_ratios, iterations);
        
        printf("\nСреднее геометрическое производительности: %.2f%%\n", geo_mean);
        printf("Ускорение относительно OpenBLAS: %.2fx\n\n", geo_mean / 100.0);
    }
}

int main() {
    srand(42);
    
    run_ssyrk_performance_tests();
    run_dsyrk_performance_tests();
    
    return 0;
}