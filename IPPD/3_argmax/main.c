#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "argmax.h"

int main(int argc, char *argv[]) {
    int size;
    float *array;
    int max_idx_seq, max_idx_for, max_idx_task;
    float max_val_seq, max_val_for, max_val_task;
    
    // Check command line arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <array_size>\n", argv[0]);
        return 1;
    }
    
    // Parse array size
    size = atoi(argv[1]);
    if (size <= 0) {
        fprintf(stderr, "Error: Array size must be a positive integer\n");
        return 1;
    }
    
    // Print number of threads
    #pragma omp parallel
    {
        #pragma omp single
        printf("Running with %d threads\n", omp_get_num_threads());
    }
    
    // Allocate and initialize array with random values
    array = (float *)malloc(size * sizeof(float));
    if (!array) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Seed the random number generator
    srand(time(NULL));
    
    // Initialize array with random floating-point values
    for (int i = 0; i < size; i++) {
        array[i] = (float)rand() / RAND_MAX * 1000.0;
    }
    
    // Make one element significantly larger to make verification easier
    int special_idx = rand() % size;
    array[special_idx] = 10000.0;
    printf("Special maximum value (10000.0) inserted at index %d\n", special_idx);
    
    // Run sequential version
    printf("\nRunning sequential argmax...\n");
    argmax_sequential(array, size, &max_idx_seq, &max_val_seq);
    printf("Maximum value: %f at index %d\n", max_val_seq, max_idx_seq);
    
    // Run OpenMP for version
    printf("\nRunning OpenMP for argmax...\n");
    argmax_openmp_for(array, size, &max_idx_for, &max_val_for);
    printf("Maximum value: %f at index %d\n", max_val_for, max_idx_for);
    
    // Run OpenMP task version
    printf("\nRunning OpenMP task argmax...\n");
    argmax_openmp_task(array, size, &max_idx_task, &max_val_task);
    printf("Maximum value: %f at index %d\n", max_val_task, max_idx_task);
    
    // Verify all implementations found the same maximum
    if (max_idx_seq == max_idx_for && max_idx_for == max_idx_task &&
        max_val_seq == max_val_for && max_val_for == max_val_task) {
        printf("\nAll implementations found the same maximum. CORRECT!\n");
    } else {
        printf("\nImplementations found different maxima. ERROR!\n");
        printf("Sequential: value %f at index %d\n", max_val_seq, max_idx_seq);
        printf("OpenMP for: value %f at index %d\n", max_val_for, max_idx_for);
        printf("OpenMP task: value %f at index %d\n", max_val_task, max_idx_task);
    }
    
    // Clean up
    free(array);
    
    return 0;
}
