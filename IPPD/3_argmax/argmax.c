#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "argmax.h"

// Sequential implementation of argmax
void argmax_sequential(float *array, int size, int *max_idx, float *max_val) {
    double start, end;
    
    start = omp_get_wtime();
    
    *max_idx = 0;
    *max_val = array[0];
    
    for (int i = 1; i < size; i++) {
        if (array[i] > *max_val) {
            *max_val = array[i];
            *max_idx = i;
        }
    }
    
    end = omp_get_wtime();
    printf("Sequential time: %f\n", end - start);
}

// OpenMP implementation using parallel for with reduction
void argmax_openmp_for(float *array, int size, int *max_idx, float *max_val) {
    double start, end;
    
    start = omp_get_wtime();
    
    *max_idx = 0;
    *max_val = array[0];
    
    #pragma omp parallel
    {
        int local_idx = 0;
        float local_max = array[0];
        
        #pragma omp for
        for (int i = 1; i < size; i++) {
            if (array[i] > local_max) {
                local_max = array[i];
                local_idx = i;
            }
        }
        
        #pragma omp critical
        {
            if (local_max > *max_val) {
                *max_val = local_max;
                *max_idx = local_idx;
            }
        }
    }
    
    end = omp_get_wtime();
    printf("OpenMP for time: %f\n", end - start);
}

// Recursive function to find the maximum using divide and conquer
void find_max_recursive(float *array, int start, int end, int *max_idx, float *max_val, int level) {
    // Base case: small enough to compute directly
    if (end - start <= 1000) {
        *max_idx = start;
        *max_val = array[start];
        
        for (int i = start + 1; i < end; i++) {
            if (array[i] > *max_val) {
                *max_val = array[i];
                *max_idx = i;
            }
        }
        return;
    }
    
    // Divide the array into two parts
    int mid = start + (end - start) / 2;
    int left_idx, right_idx;
    float left_max, right_max;
    
    // Create tasks for recursive calls (use tasks for a certain number of levels)
    if (level < 4) {
        #pragma omp task shared(left_idx, left_max)
        find_max_recursive(array, start, mid, &left_idx, &left_max, level + 1);
        
        #pragma omp task shared(right_idx, right_max)
        find_max_recursive(array, mid, end, &right_idx, &right_max, level + 1);
        
        #pragma omp taskwait
    } else {
        find_max_recursive(array, start, mid, &left_idx, &left_max, level + 1);
        find_max_recursive(array, mid, end, &right_idx, &right_max, level + 1);
    }
    
    // Combine results
    if (left_max >= right_max) {
        *max_val = left_max;
        *max_idx = left_idx;
    } else {
        *max_val = right_max;
        *max_idx = right_idx;
    }
}

// OpenMP implementation using tasks and divide-and-conquer
void argmax_openmp_task(float *array, int size, int *max_idx, float *max_val) {
    double start, end;
    
    start = omp_get_wtime();
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            find_max_recursive(array, 0, size, max_idx, max_val, 0);
        }
    }
    
    end = omp_get_wtime();
    printf("OpenMP task time: %f\n", end - start);
}
