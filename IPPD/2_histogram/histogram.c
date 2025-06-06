
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include "random.h"

// uncomment this #define if you want diagnostic output
// #define     DEBUG         0

#define num_trials 1000000 // number of x values
#define num_buckets 50     // number of buckets in hitogram
static long xlow = 0.0;    // low end of x range
static long xhi = 100.0;   // High end of x range

int initHist(long *hist)
{
    for (int i = 0; i < num_buckets; i++)
        hist[i] = 0;
    return 0;
}

int analyzeResults(double time, long *hist)
{
    double sumh = 0.0, sumhsq = 0.0, ave, std_dev;
    // compute statistics ... ave, std-dev for whole histogram and quartiles
    for (int i = 0; i < num_buckets; i++)
    {
        sumh += (double)hist[i];
        sumhsq += (double)hist[i] * hist[i];
    }

    ave = sumh / num_buckets;
    std_dev = sqrt(sumhsq - sumh * sumh / (double)num_buckets);

    printf(" histogram for %d buckets of %d values\n", num_buckets, num_trials);
    printf(" ave = %f, std_dev = %f\n", (float)ave, (float)std_dev);
    printf(" in %f seconds\n", (float)time);

    return 0;
}

int main()
{
    double *x = (double *)malloc(num_trials * sizeof(double));
    long hist[num_buckets]; // the histogram
    double bucket_width;    // the width of each bucket in the histogram
    double time;

    // print number of threads
    #pragma omp parallel
    {
        #pragma omp single
        printf("%d threads\n", omp_get_num_threads());
    }

    seed(xlow, xhi); // seed the random number generator over range of x
    bucket_width = (xhi - xlow) / (double)num_buckets;

    // fill the array
    for (int i = 0; i < num_trials; i++)
        x[i] = drandom();

    ////////////////////////////////////////////////////////////////
    // Assign x values to the right histogram bucket -- sequential
    ////////////////////////////////////////////////////////////////

    printf("Sequential");

    initHist(hist);

    // Assign x values to the right historgram bucket
    time = omp_get_wtime();
    for (int i = 0; i < num_trials; i++)
    {

        long ival = (long)(x[i] - xlow) / bucket_width;

        hist[ival]++;

#ifdef DEBUG
        printf("i = %d,  xi = %f, ival = %d\n", i, (float)x[i], ival);
#endif
    }

    time = omp_get_wtime() - time;

    analyzeResults(time, hist);

    ////////////////////////////////////////////////////////////////
    // Assign x values to the right histogram bucket -- critical
    ////////////////////////////////////////////////////////////////

    printf("par with critical\n");
    initHist(hist);
    time = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < num_trials; i++) {
        long ival = (long)((x[i] - xlow) / bucket_width);
        #pragma omp critical
        {
            hist[ival]++;
        }
    }

    time = omp_get_wtime() - time;
    analyzeResults(time, hist);

    ////////////////////////////////////////////////////////////////
    // Assign x values to the right histogram bucket -- locks
    ////////////////////////////////////////////////////////////////

    printf("par with locks ");
    initHist(hist);
    time = omp_get_wtime();

    omp_lock_t locks[num_buckets];
    for (int i = 0; i < num_buckets; i++) {
        omp_init_lock(&locks[i]);
    }

    #pragma omp parallel for
    for (int i = 0; i < num_trials; i++) {
        long ival = (long)((x[i] - xlow) / bucket_width);
        omp_set_lock(&locks[ival]);
        hist[ival]++;
        omp_unset_lock(&locks[ival]);
    }

    for (int i = 0; i < num_buckets; i++) {
        omp_destroy_lock(&locks[i]);
    }

    time = omp_get_wtime() - time;
    analyzeResults(time, hist);


    ////////////////////////////////////////////////////////////////
    // Assign x values to the right histogram bucket -- reduction
    ////////////////////////////////////////////////////////////////

    printf("par with reduction\n");
    initHist(hist);
    time = omp_get_wtime();

    long hist_private[num_buckets * omp_get_max_threads()];
    for (int i = 0; i < num_buckets * omp_get_max_threads(); i++)
        hist_private[i] = 0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long *local_hist = &hist_private[tid * num_buckets];

        #pragma omp for
        for (int i = 0; i < num_trials; i++) {
            long ival = (long)((x[i] - xlow) / bucket_width);
            local_hist[ival]++;
        }
    }

    for (int i = 0; i < num_buckets; i++) {
        for (int t = 0; t < omp_get_max_threads(); t++) {
            hist[i] += hist_private[t * num_buckets + i];
        }
    }

    time = omp_get_wtime() - time;
    analyzeResults(time, hist);

   
    //////////////////////////////////////////////////////////////// 
   
    free(x);
    return 0;
}
