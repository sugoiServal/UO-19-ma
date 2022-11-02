/*
 * SAUS algorithm from "Linear Models with Many Cores and CPUs: A Stochastic Atomic Update Scheme" by E.Raff et. al
 */

#ifndef _SAUS_h_
#define _SAUS_h_

#include "timer.h"
#include "thread_array.h"


int saus_atomic_update(thread_array_t iterate, data_t *data, int thread_num, double prob_threshold, int* update_count);
int saus_accumulate(thread_array_t iterate, int num_threads, int thread_id, int num_features);

int saus_test_instance(thread_array_t iterate, data_t *data, int thread_num, double prob_threshold);
int saus_initialize(int num_features, int num_threads);
int saus_deinitialize(void);


// This will be a lower bound on the number of collisions in
// atomic increment. Since it is not made thread-safe, threads
// may overwrite each others' increments.
extern unsigned int SAUS_num_atomic_dec_collisions;


#endif // _SAUS_h_
