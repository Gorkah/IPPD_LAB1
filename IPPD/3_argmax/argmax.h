#ifndef ARGMAX_H
#define ARGMAX_H

void argmax_sequential(float *array, int size, int *max_idx, float *max_val);
void argmax_openmp_for(float *array, int size, int *max_idx, float *max_val);
void argmax_openmp_task(float *array, int size, int *max_idx, float *max_val);

#endif
