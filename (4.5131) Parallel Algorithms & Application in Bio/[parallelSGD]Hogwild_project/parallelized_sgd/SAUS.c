#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "data.h"
#include "problem.h"
#include "thread_array.h"
#include "psgd_analysis.h"
#include "SAUS.h"


// This will be a lower bound on the number of collisions. Since
// it is not made thread-safe, threads may overwrite each others'
// increments.
unsigned int SAUS_num_atomic_dec_collisions = 0;


static thread_array_t /*sparse_array_t*/ sparse_sample_grads;
static thread_array_t /*sparse_array_t*/ thread_local_grads;
static int num_threads;			//local num_thread 
static thread_array_t /*unsigned int*/ rng_seedp; // seed state for random number generator, one for each thread

static ttimer_t *gradient_timers;
static ttimer_t *coord_update_timers;


//used by hogwild
//__atomic_exchange: Built-in Function: void __atomic_exchange (type *ptr, type *val, type *ret, int memorder)
//This is the generic version of an atomic exchange. It stores the contents of *val into *ptr. The original value of *ptr is copied into *ret.
static void atomic_decrement(double *dest, double dec_amt) {
	double ret_val;
	double orig_val;
	double dec_val;
	// Perform a compare-and-swap. If the update was of dest was
	//   atomic, the returned value should match the orig_val's
	//   read of the destination.
	do {
		orig_val = *dest;
		dec_val = orig_val - dec_amt;
		__atomic_exchange(dest, &dec_val, &ret_val, __ATOMIC_RELAXED);
		if (ret_val != orig_val)
			SAUS_num_atomic_dec_collisions++;   //if retrieve an different original value, it indicate the collision has occur
	} while (ret_val != orig_val);
}


int saus_test_instance(thread_array_t iterate, data_t *data, int thread_num, double prob_threshold) {
	sparse_array_t *thread_local_grad = &TA_idx(thread_local_grads, thread_num, sparse_array_t);
	
	for (int index = 0; index < 1000; index++){
		printf("init value of local feature idx#%i: %f \n", index, ((thread_local_grad->pts)[index].value));
	}

}

int saus_atomic_update(thread_array_t iterate, data_t *data, int thread_num, double prob_threshold, int* update_count) {
	// Get random sample
	//    rand_r is reentrant (thread safe)
	int rand_index = rand_r(&TA_idx(rng_seedp, thread_num, unsigned int)) % data->num_samples;
	sparse_array_t sparse_sample_X = data->sparse_X[rand_index];
	double sample_y = data->y[rand_index];
	
	// Evaluate gradient
	sparse_array_t sparse_sample_grad = TA_idx(sparse_sample_grads, thread_num, sparse_array_t); //GET BUFFER-current thread's gradient sparse array
	sparse_sample_grad.len = sparse_sample_X.len;
#ifdef TRACK_GRADIENT_COORDUPDATE
	timer_start(&gradient_timers[thread_num]);
#endif
	gradient(iterate, sparse_sample_X, sample_y, &sparse_sample_grad);
#ifdef TRACK_GRADIENT_COORDUPDATE
	timer_pause(&gradient_timers[thread_num]);
#endif
#ifdef TRACK_GRADIENT_COORDUPDATE
	timer_start(&coord_update_timers[thread_num]);
#endif	

	sparse_array_t *thread_local_grad = &TA_idx(thread_local_grads, thread_num, sparse_array_t);
	for (int i = 0; i < sparse_sample_grad.len; i++) {
		//thread_local_grads[i] <- sparse_sample_grad[i]
		int index    = sparse_sample_grad.pts[i].index;   		//index in target iterate
		double value = sparse_sample_grad.pts[i].value;			//the value to be write
		double *local_i_address = &((thread_local_grad->pts)[index].value);  //the current parameter value's address in the local_grad
		atomic_decrement(local_i_address, get_stepsize()*value);  //TA_idx get the address of the target TA

		// filp the coin
		double rand_bernoulli= ((double)rand_r(&TA_idx(rng_seedp, thread_num, unsigned int)))/(double)RAND_MAX;
		if (rand_bernoulli<prob_threshold){
			//printf("id%i %f\n",index, TA_idx(iterate, index, double));
			//printf("%f\n",*local_i_address);
			atomic_decrement(&TA_idx(iterate, index, double), (*local_i_address));  //TA_idx get the address of the target TA
			//printf("%f\n",TA_idx(iterate, index, double));
			*local_i_address = 0;
			//printf("%f ///////////////////////\n",*local_i_address);
			*update_count+=1;
		}	
	}

#ifdef TRACK_GRADIENT_COORDUPDATE
	timer_pause(&coord_update_timers[thread_num]);
#endif	
	return 0;
}


int saus_accumulate(thread_array_t iterate, int num_threads, int thread_id, int num_features){  //num_threads:how many are there; thread_num: who am I
	int working_lb;
	int working_ub;
	if (num_threads == 1){
		working_lb = 0;
		working_ub = num_features-1;
	}
	else if (thread_id == 0){
		working_lb = 0;
		working_ub = floor(((double)thread_id+1)*(double)num_features/(double)num_threads);
	}
	else if(thread_id == num_threads-1){
		working_lb = floor((double)thread_id*(double)num_features/(double)num_threads) + 1;
		working_ub = num_features-1;
	}
	else{
		working_lb = floor((double)thread_id*(double)num_features/(double)num_threads) + 1;
		working_ub = floor(((double)thread_id+1)*(double)num_features/(double)num_threads);
	}

	for (int thread_id=0; thread_id < num_threads; thread_id++){
		sparse_array_t *thread_local_grad = &TA_idx(thread_local_grads, thread_id, sparse_array_t);
		for (int index = working_lb; index <= working_ub; index++){   //index fraction in inerate
			double value = (thread_local_grad->pts)[index].value;
			atomic_decrement(&TA_idx(iterate, index, double), value);  //TA_idx get the address of the target TA
			(thread_local_grad->pts)[index].value = 0;  //clear the cache after update to iterate
		}
	}
	return 0;
}


//data->num_feature and num_threads from main(args)
int saus_initialize(int num_features, int num_thr) {
	num_threads = num_thr;
	srand(time(NULL));

	// the sparse array: enough space to hold all point, but not necessarily use all(only up to len)

    // Initialize thread_local_grads * each thread, and malloc it with size of num_features
	malloc_thread_array(&thread_local_grads, num_threads);
	for (int n = 0; n < num_threads; n++) {
		TA_idx(thread_local_grads, n, sparse_array_t).len = num_features;
		TA_idx(thread_local_grads, n, sparse_array_t).pts = (sparse_point_t *) malloc(num_features * sizeof(sparse_point_t));
		for (int i = 0; i < num_features; i++){
			TA_idx(thread_local_grads, n, sparse_array_t).pts[i].index = i;
			TA_idx(thread_local_grads, n, sparse_array_t).pts[i].value = 0;
		}
	}

	// Initialize sparse_sample_grads (used to store calculated gradients)
	malloc_thread_array(&sparse_sample_grads, num_threads);
	for (int n = 0; n < num_threads; n++) {
		TA_idx(sparse_sample_grads, n, sparse_array_t).len = 0;
		TA_idx(sparse_sample_grads, n, sparse_array_t).pts = (sparse_point_t *) malloc(num_features * sizeof(sparse_point_t));
	}
	// Initialize timers
	gradient_timers = (ttimer_t *) malloc(num_thr * sizeof(ttimer_t));
	coord_update_timers = (ttimer_t *) malloc(num_thr * sizeof(ttimer_t));
	for (int n = 0; n < num_threads; n++) {
#ifdef __linux__
		timer_initialize(&gradient_timers[n], TIMER_SCOPE_THREAD);
		timer_initialize(&coord_update_timers[n], TIMER_SCOPE_THREAD);
#endif // __linux__
#ifdef __APPLE__
		timer_initialize(&gradient_timers[n], TIMER_SCOPE_PROCESS);
		timer_initialize(&coord_update_timers[n], TIMER_SCOPE_PROCESS);
#endif // __APPLE__
	}
	// Initialize RNG seed states
	malloc_thread_array(&rng_seedp, num_thr);
	for (int n = 0; n < num_thr; n++) {
		TA_idx(rng_seedp, n, unsigned int) = rand() + n;
	}
	return 0;
}


int saus_deinitialize(void) {
	// Free thread_local_grads
	for (int n = 0; n < num_threads; n++) {
		free(TA_idx(thread_local_grads, n, sparse_array_t).pts);
	}
	free_thread_array(&thread_local_grads);

	// Free sparse_sample_grads
	for (int n = 0; n < num_threads; n++) {
		free(TA_idx(sparse_sample_grads, n, sparse_array_t).pts);
	}
	free_thread_array(&sparse_sample_grads);
	// Free timers
	for (int n = 0; n < num_threads; n++) {
		timer_deinitialize(&gradient_timers[n]);
		timer_deinitialize(&coord_update_timers[n]);
	}
	free(gradient_timers);
	free(coord_update_timers);
	// Free RNG seed states
	free_thread_array(&rng_seedp);
	// Print lower bound on # of atomic_decrement collisions
	printf("There were at least %d collisions in atomic_decrement\n", SAUS_num_atomic_dec_collisions);
	return 0;
}


// abstract in timer.h, and used in last part of run_psgd_general_analysis()
int timer_get_internal_timer_stats_SAUS(timerstats_t *gradient_stats, timerstats_t *coord_update_stats) {
	int rc;
	for (int n = 0; n < num_threads; n++) {
		rc = timer_get_stats(&gradient_timers[n], &gradient_stats[n]);
		if (rc)
			return rc;
		rc = timer_get_stats(&coord_update_timers[n], &coord_update_stats[n]);
		if (rc)
			return rc;
	}
	return 0;
}
