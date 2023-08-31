#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <pthread.h>
#include "problem.h"
#include "psgd_analysis.h"
#include "SAUS.h"




/*
 *  Analysis and algorithm wrappers
 */


#define THREADJOB_NONE				0
#define THREADJOB_RECORD_ITERATES	1


// Condition vars to start all threads at the same time
pthread_cond_t sync_start_cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t sync_start_mutex = PTHREAD_MUTEX_INITIALIZER;

 
typedef struct _algowrapperargs_t {               //thread related resources and parameters
	int num_iters;
	int thread_num;
	int num_threads;
	int num_features;
	double prob_threshold;  //the updating threshold for SAUS
	int threadjob;
	log_t *log;
	timerstats_t *threadstats;
	data_t *data;
	pthread_cond_t *sync_cond;
	pthread_mutex_t *sync_mutex;
	pthread_barrier_t *barrier;
	thread_array_t /*double*/ iterate; // array of length data->num_features
} algowrapperargs_t;


static void* algo_wrapper(void *wrapperargs) {     //each thread execute the following process
	int rc;
	int log_step; // # iters between logging
	ttimer_t timer;
	algowrapperargs_t *args = (algowrapperargs_t *) wrapperargs;
	log_step = args->num_iters / get_current_problem().num_log_points;
#ifdef __linux__
	timer_initialize(&timer, TIMER_SCOPE_THREAD);
#endif // __linux__
#ifdef __APPLE__
	timer_initialize(&timer, TIMER_SCOPE_PROCESS);
#endif // __APPLE__

	// Wait at starting line for release by condition 
	pthread_mutex_lock(&sync_start_mutex);
    pthread_cond_wait(&sync_start_cond, &sync_start_mutex); 
    pthread_mutex_unlock(&sync_start_mutex); // unlocking for other threads
	timer_start(&timer);

	// Run algo for num_iters iterations
	for (int i = 1; i <= args->num_iters; i++) {
		rc = current_problem.algo_update_func(args->iterate, args->data, args->thread_num);
		if (rc)
			pthread_exit(NULL);
		// Log iterate and timestamp values, if required(every iteration)
		if (args->threadjob == THREADJOB_RECORD_ITERATES) {
			if (i % log_step == 0) {
				int sz = args->log->size;
				if (sz+1 <= args->log->capacity) {
					// copy iterate value
					for (int j = 0; j < args->data->num_features; j++) {
						args->log->iterates[sz][j] = TA_idx(args->iterate, j, double);
					}
					// copy timer
					timer_pause(&timer);
					args->log->timestamps[sz] = timer;
					timer_start(&timer);
					// inc log size
					args->log->size++;
				}
			}
		}
	}
	
	// Handle this thread's timer and stats
	timer_pause(&timer);
	timer_get_stats(&timer, args->threadstats);
	timer_deinitialize(&timer);

	pthread_exit(NULL);
}


static void* saus_algo_wrapper(void *wrapperargs) { 
	int rc;
	int log_step; // # iters between logging
	ttimer_t timer;
	algowrapperargs_t *args = (algowrapperargs_t *) wrapperargs;
	log_step = args->num_iters / get_current_problem().num_log_points;
#ifdef __linux__
	timer_initialize(&timer, TIMER_SCOPE_THREAD);
#endif // __linux__
#ifdef __APPLE__
	timer_initialize(&timer, TIMER_SCOPE_PROCESS);
#endif // __APPLE__
	//printf("thread #%i on saus", args->thread_num);
	// Wait at starting line for release by condition 
	pthread_mutex_lock(&sync_start_mutex);
    pthread_cond_wait(&sync_start_cond, &sync_start_mutex); 
    pthread_mutex_unlock(&sync_start_mutex); // unlocking for other threads
	timer_start(&timer);	
	
/*
 Generic Learning Framework for SAUS: pseudocode
	weight vector w <-0, Probability multiplier prob = 1

	for each epoch do:         //epoch = args->num_iters/epoch_size		
		for iter_int_epoch, each thread do:
			saus_atomic_update(target, data, thread_num) {
				sampling x from X;
				calculate gradient /beta from x;
				rand = u(0,1);
				for each element in /beta:
					/*star*\ local_grads[i] <- /beta[i]
					if rand <= prob*1:
						automic update: target<-local_grads[i]
						local_grads[i] = 0				
			}
		if (end_of_epoch):
			wait for other thread(need wait since overwrite is possible) BARRIER(end_of_epoch)

			accumulate(target, all_thread_local_grads, thread_id) {  /* how to pass all_thread_local_grads to thread i??
				thread_id_work_area = get_work_area(thread_id)
				for i in(0, thread_num):
					local* = get_local_grad(thread_num)
					for element in range(thread_id_work_area):
						target[element] = local*[element]
		}	*/

	//implement here
	int epoch_size = 1;     //TODO: make it an option 
	printf("epoch size %i\n", epoch_size);
	int update_count = 0;
	int epoch_num = 0;
	for (int i = 1; i <= args->num_iters; i++) {

		// saus_atomic_update step and logging
		rc = saus_atomic_update(args->iterate, args->data, args->thread_num, args->prob_threshold, &update_count);
		if (rc)
			pthread_exit(NULL);
		// Log iterate and timestamp values, if required(every iteration)
		if (args->threadjob == THREADJOB_RECORD_ITERATES) {
			if (i % log_step == 0) {
				int sz = args->log->size;
				if (sz+1 <= args->log->capacity) {
					// copy iterate value
					for (int j = 0; j < args->data->num_features; j++) {
						args->log->iterates[sz][j] = TA_idx(args->iterate, j, double);
					}
					// copy timer
					timer_pause(&timer);
					args->log->timestamps[sz] = timer;
					timer_start(&timer);
					// inc log size
					args->log->size++;
				}
			}
		}
		// at the end of a epoch, start accumulate step (all thread simultaneously)
		//printf("num_iter %i, step %i, step_size %i\n", args->num_iters, i, (i % epoch_size));
 		if (i % epoch_size == 0) {
			epoch_num++;
			pthread_barrier_wait(args->barrier);
			//printf("thread#%i, accumulate in epoch %i\n", args->thread_num, epoch_num);
			saus_accumulate(args->iterate, args->num_threads, args->thread_num, args->num_features);
			pthread_barrier_wait(args->barrier);
		} 
	} 
	printf("there are %i updates in thread #%i\n", update_count,args->thread_num);
	// Handle this thread's timer and stats
	timer_pause(&timer);
	timer_get_stats(&timer, args->threadstats);
	timer_deinitialize(&timer);
	pthread_exit(NULL);
} 
/*
// unit test for run_psgd_general_analysis()
static void* saus_algo_wrapper(void *wrapperargs){
	int rc;
	int log_step; // # iters between logging
	ttimer_t timer;
	algowrapperargs_t *args = (algowrapperargs_t *) wrapperargs;
	printf("hello iam thread #%i for SAUS\n", args->thread_num);
	rc = saus_test_instance(args->iterate, args->data, args->thread_num, args->prob_threshold);

	sleep(20);
	int ret;
	printf("thread #%i has reached the barrier point\n", args->thread_num);
	ret = pthread_barrier_wait(args->barrier);
	if (ret ==  PTHREAD_BARRIER_SERIAL_THREAD || ret == 0){
		printf("all threads, includes #%i has reached the barrier point\n", args->thread_num);
	}
	else{ printf("no barrier\n");}
} 
*/
int run_psgd_general_analysis(int num_threads, data_t *data, log_t *log, timerstats_t *main_thread_stats, timerstats_t *threads_stats_array, timerstats_t *gradient_stats_array, timerstats_t *coord_update_stats_array) {
	int rc;
	//args
	algowrapperargs_t *args;

	//pthreading related
	pthread_t *threads;
	pthread_attr_t attr;
	pthread_barrier_t barrier; 
	void *status;
	ttimer_t main_thread_timer;
	pthread_mutex_t sync_mutex = PTHREAD_MUTEX_INITIALIZER;
	pthread_cond_t sync_cond = PTHREAD_COND_INITIALIZER;

	//the target parameters to be updated ==> args
	thread_array_t /*double*/ iterate;   

	// Initialize, alloc, and set thread joinable
	args = (algowrapperargs_t *) malloc(num_threads * sizeof(algowrapperargs_t));
	threads = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
	pthread_attr_init(&attr);	
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);   //set joinable
	timer_initialize(&main_thread_timer, TIMER_SCOPE_PROCESS);
	malloc_thread_array(&iterate, data->num_features);
	srand(time(0));
	for (int i = 0; i < data->num_features; i++){	
		double value = (double)rand()/(double)RAND_MAX;
		TA_idx(iterate, i, double) = value;
	}
	rc = current_problem.algo_init_func(data->num_features, num_threads);  //algorithm specific address allocate
	if (rc)
		return rc;

	// Start wrapper threads
	for (int thread_num = 0; thread_num < num_threads; thread_num++) {
		// Fill args for this thread
		args[thread_num].num_iters = get_current_problem().num_total_iter / num_threads;
		args[thread_num].thread_num = thread_num;
		args[thread_num].num_threads = num_threads;
		args[thread_num].num_features = data->num_features;
		if (!thread_num) {
			args[thread_num].threadjob = THREADJOB_RECORD_ITERATES;
			args[thread_num].log = log;
		} else {
			args[thread_num].threadjob = THREADJOB_NONE;
			args[thread_num].log = NULL;
		}
		args[thread_num].threadstats = &threads_stats_array[thread_num];
		args[thread_num].data = data;
		args[thread_num].sync_cond = &sync_cond;
		args[thread_num].sync_mutex = &sync_mutex;
		args[thread_num].iterate = iterate;


		// Create thread with args
		if (get_current_problem().saus_flag) {
			//initiate barrier object here
			int ret;
			ret = pthread_barrier_init(&barrier, NULL, num_threads);
			if (ret){
				return ret;
			}
			//barrier to args
			for (int thread_num = 0; thread_num < num_threads; thread_num++) {
				args[thread_num].barrier = &barrier;
			} 
			int prob_multiplier = 1;			//TODO: make it an option
			args[thread_num].prob_threshold = (1*(double)prob_multiplier)/(double)num_threads;
			rc = pthread_create(&threads[thread_num], &attr, saus_algo_wrapper, (void *)&args[thread_num]);
		}
		else {
			rc = pthread_create(&threads[thread_num], &attr, algo_wrapper, (void *)&args[thread_num]);
		}
		if (rc)
			return rc;
	}

	// Start main thread's timer and release threads
	sleep(1); // wait for threads to hit condition
	timer_start(&main_thread_timer);
	rc = pthread_cond_broadcast(&sync_start_cond);
	if (rc)
		return rc;

	// Wait for all threads to finish
	for (int thread_num = 0; thread_num < num_threads; thread_num++) {
		rc = pthread_join(threads[thread_num], &status);
		if (rc)
			return rc;
	}

	// Stop main thread timer
	timer_pause(&main_thread_timer);
	timer_get_stats(&main_thread_timer, main_thread_stats);

	// Get gradient / coord updt stats
	timer_get_internal_timer_stats(gradient_stats_array, coord_update_stats_array);

	// Clean up
	free(threads);
	free_thread_array(&iterate);
	timer_deinitialize(&main_thread_timer);
	pthread_attr_destroy(&attr);
	if (get_current_problem().saus_flag){
		pthread_barrier_destroy(&barrier);
	}
	current_problem.algo_deinit_func();
	return 0;
}




/*
 *  Logging
 */
/* 
int randomize_iterate(thread_array_t *iterate, data_t *data){
	for (int i = 0; i < data->num_features; i++){
		srand(time(0));
		double value = (double)rand()/(double)RAND_MAX;
		double *address = &TA_idx(iterate, i, double);
	}
} */

int log_initialize(log_t *log, int num_data_features) {
	// Alloc iterates array
	log->iterates = (double **) malloc(get_current_problem().num_log_points*sizeof(double *));
	// Alloc each iterate
	for (int i = 0; i < get_current_problem().num_log_points; i++) {
		int iterate_size = num_data_features*sizeof(double);
		double *iterate = (double *) malloc(iterate_size);
		memset(iterate, 0, iterate_size);
		log->iterates[i] = iterate;
	}
	// Alloc timestamp array
	log->timestamps = (ttimer_t *) malloc(get_current_problem().num_log_points*sizeof(ttimer_t));
	log->size = 0;
	log->capacity = get_current_problem().num_log_points;
	log->num_data_features = num_data_features;
	return 0;
}


int log_free(log_t *log) {
	// Free each iterate
	for (int i = 0; i < get_current_problem().num_log_points; i++) {
		free(log->iterates[i]);
	}
	// Free iterates array
	free(log->iterates);
	// Free timestamp array
	free(log->timestamps);
	return 0;
}




/*
 *  Debug printouts
 */


void print_dense_array(double *arr, int len) {
	printf("[ ");
	for (int i = 0; i < len; i++) {
		printf("%f ", arr[i]);
	}
	printf("]\n");
}


void print_sparse_array(sparse_array_t *arr) {
	printf("[ ");
	for (int i = 0; i < arr->len; i++) {
		printf("%d:%f ", arr->pts[i].index, arr->pts[i].value);
	}
	printf("]\n");
}





/*
 * Writing results to files
 */


static int write_results_log(char *results_dir, int num_threads, log_t *log) {
	FILE *fp;
	char filename[130];
	sprintf(filename, "%s/log_%dthreads.csv", results_dir, num_threads);
	fp = fopen(filename,"w");
	if (!fp)
		return -1;
	// Write header
	fprintf(fp, "Time, Iterate\n");
	// Write each log
	double start_cumulative = log->timestamps[0].real_cumulative;
	for (int i = 0; i < get_current_problem().num_log_points; i++) {
		// Write real time elapsed
		double time_elapsed = timer_get_elapsed(start_cumulative, log->timestamps[i].real_cumulative);
		fprintf(fp, "%f, ", time_elapsed);
		// Write iterate
		fprintf(fp, "[");
		for (int j = 0; j < log->num_data_features; j++) {
			fprintf(fp, "%f", log->iterates[i][j]);
			if (j != log->num_data_features-1) {
				fprintf(fp, ", ");
			}
		}
		fprintf(fp, "]");
		fprintf(fp, "\n");
	}
	fclose(fp);
	return 0;
}


static int write_results_threads_stats(int num_threads, char *results_dir, timerstats_t main_thread_stats, timerstats_t *threads_stats_arr) {
	FILE *fp;
	char filename[130];
	sprintf(filename, "%s/threadstats_%dthread.csv", results_dir, num_threads);
	fp = fopen(filename,"w");
	if (!fp)
		return -1;
	// Write header
	fprintf(fp, "Threadname, Real, User, Sys\n");
	// Write main thread stats
	fprintf(fp, "Main, %f, %f, %f\n", main_thread_stats.real, main_thread_stats.user, main_thread_stats.sys);
	// Write other threads stats
	for (int i = 0; i < num_threads; i++) {
		fprintf(fp, "Thread%d, %f, %f, %f\n", i, threads_stats_arr[i].real, threads_stats_arr[i].user, threads_stats_arr[i].sys);
	}
	fclose(fp);
	return 0;
}


static int write_results_grad_coord(int num_threads, char *results_dir, timerstats_t *gradient_stats_array, timerstats_t *coord_update_stats_array) {
	FILE *fp;
	char filename[130];
	sprintf(filename, "%s/gradcoord_%dthread.csv", results_dir, num_threads);
	fp = fopen(filename,"w");
	if (!fp)
		return -1;
#ifdef TRACK_GRADIENT_COORDUPDATE
	// Write header
	fprintf(fp, "Threadname, Grad_Real, Grad_User, Grad_Sys, Coord_Real, Coord_User, Coord_Sys\n");
	// Write threads stats
	for (int i = 0; i < num_threads; i++) {
		fprintf(fp, "Thread%d, %f, %f, %f, %f, %f, %f\n", i,
				gradient_stats_array[i].real,
				gradient_stats_array[i].user,
				gradient_stats_array[i].sys,
				coord_update_stats_array[i].real,
				coord_update_stats_array[i].user,
				coord_update_stats_array[i].sys);
	}
#else
	// Write note that we didn't track this info
	fprintf(fp, "No stats available. Did not track them.");
#endif // TRACK_GRADIENT_COORDUPDATE
	fclose(fp);
	return 0;
}


int create_results_dir(char *algorithm_name, char *problem_type, char *input_filename, char *ret_results_dir) {
	int ret_i = 0;
	int i = 0;
	// algorithm name
	i = 0;
	while (i < strlen(algorithm_name)) {
		ret_results_dir[ret_i] = algorithm_name[i];
		i++;
		ret_i++;
	}
	ret_results_dir[ret_i] = '_';
	ret_i++;

	// problem type
	i = 0;
	while (i < strlen(problem_type)) {
		ret_results_dir[ret_i] = problem_type[i];
		i++;
		ret_i++;
	}
	ret_results_dir[ret_i] = '_';
	ret_i++;

	// input filename
	i = 0;
	while (input_filename[i] != '.' && i < strlen(input_filename)) {
		ret_results_dir[ret_i] = input_filename[i];
		i++;
		ret_i++;
	}

	ret_results_dir[ret_i] = '\0';
	return mkdir(ret_results_dir, 0777);
}


int write_results_to_file(int num_threads, char *results_dir, log_t *log, timerstats_t main_thread_stats, timerstats_t *threads_stats_array, timerstats_t *gradient_stats_array, timerstats_t *coord_update_stats_array) {
	int rc;
	rc = write_results_log(results_dir, num_threads, log);
	if (rc)
		return rc;
	rc = write_results_threads_stats(num_threads, results_dir, main_thread_stats, threads_stats_array);
	if (rc)
		return rc;
	rc = write_results_grad_coord(num_threads, results_dir, gradient_stats_array, coord_update_stats_array);
	if (rc)
		return rc;
	return 0;
}
