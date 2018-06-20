/*

This script provides utilities for training a restricted Boltzmann machine.  It is the C analogue of the Python scripts in the repository root directory.  We intend to get speedup here using Intel Intrinsics AVX2 SIMD instructions, together with multithreaded processing.

*/
#include<string.h>
#include<unistd.h>
#include<pthread.h>
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<immintrin.h>
#include<assert.h>
#include<time.h>

#define float4 __m128
#define float8 __m256
#define double2 __m128d
#define double4 __m256d
#define uint4 __m128i
#define uint8 __m256i
#define ulong4 __m256i
#define int4 __m128i
#define ulong2 __m128i

#define MAX_THREADS 128
#define NANOSECONDS_PER_SECOND 1e9
#define AVX2_BYTE_ALIGNMENT 32
#define NUM_TRIALS 100000

/****** DEFINE THE RBM STRUCT INITIALIZER / DESTORYER *******/

typedef struct RBM {
    uint32_t v_dim, h_dim; // 8 bytes
    uint32_t r, c; // 8 bytes. Smallest multiples of 8 greater than v_dim and h_dim
    float8** W; // Store the rows of W
    float8** WT; // Also store the columns of W
} RBM;

/*
//FIXME: Update for addition of WT
RBM* init_rbm(unsigned int v, unsigned int h) {
    
    RBM* out = (RBM *) malloc( sizeof(RBM) );
    out->v_dim = v;
    out->h_dim = h;
    out->r = 8 * (v / 8) + 8;
    out->c = 8 * (h / 8) + 8;

    // The weights of the RBM are stored as a dynamically allocated set of rows
    if(out->W = malloc( (1 + v/8) * sizeof(float8 *) ) == NULL) exit(-1);
    if(out->WT = malloc( (1 + h/8) * sizeof(float8 *) ) == NULL) exit(-1);
    // Initialize all our rows/columns to zero
    
    for (int i = 0; i < h; i++) {
        // Allocate memory for each row
        if(out->W[i] = (float *) malloc( v * sizeof(float) )) return NULL;
    }
    
    return out;
}

void destroy_rbm(RBM* r) {
    // Free the memory being used to store each row 
    for (int i = 0; i < r->h_dim; i++) {
        free(r->W[i]);
        r->W[i] = NULL;
    }
    for (int i = 0; i < r->h_dim; i++) {
        free(r->WT[i]);
        r->WT[i] = NULL;
    }
    // Free the memory being used to store pointers to the rows
    free(r->W);
    r->W = NULL;
    free(r->WT);
    r->WT = NULL;
    // Finally free the memory being used to store the RBM
    free(r);
}
*/ 

/*************************************************************/

//  Task scheduler for multithreading matrix multiplication

/*************************************************************/

typedef void *ThreadPool;
// Declares a type "Func" an instance of which is a pointer to a function f: (void *) -> void
typedef void (*Func)(void *);
typedef void *Args;

// Struct to store function executions in a linked list
typedef struct work_t {
    void (*task) (void *);
    void* args;
    struct work_t* next;
} work_t;

typedef struct _ThreadPool {
    int num_threads;
    int size;
    int shutdown;
    int reject_new_work;
    pthread_t* threads;
    work_t* head;
    work_t* tail;
    pthread_mutex_t lock;
    pthread_cond_t not_empty;  
    pthread_cond_t empty;
} _ThreadPool;

// This function runs inside the threads when they spawn.  It's an infinite loop that waits for tasks to get loaded into the queue.  When it sees a new task, it executes the work.
void* getWork(ThreadPool p) {
    
    _ThreadPool* pool = (_ThreadPool *) p;
    work_t* job;

    while(1) { 
        // Lock the ThreadPool so that no other threads can pull tasks off
        pthread_mutex_lock( &(pool->lock) );

        // If the task pool is empty, wait until it isn't empty anymore.  
        while (pool->size == 0) {
            // When the queue is empty, the threads will hold here and wait for dispatch() to signal not_empty
            pthread_cond_wait( &(pool->not_empty), &(pool->lock) ); // LINE (*)
            // If we get the shutdown signal, then end the thread.
            if(pool->shutdown) { 
                pthread_mutex_unlock( &(pool->lock) );
                pthread_exit(NULL);
            }
        }
        // Grab a job off the queue
        job = pool->head;
        pool->size--;
        if (job->next != NULL) {
            pool->head = job->next;
        } // If this was the last job in the queue
        else {
            pool->head = NULL;
            pool->tail = NULL;
        }
         
        if (pool->size == 0 && ! pool->shutdown) {
            // There may or may not be a thread who has called deleteThreadPool and is waiting for a signal that the queue is empty.
            pthread_cond_signal( &(pool->empty) );
        }
        // Now that we've grabbed a task, we can unlock the ThreadPool so that other threads can take a task.
        pthread_mutex_unlock( &(pool->lock) );
        // Do the work
        (job->task)(job->args);
        // Now that the work is done, free the job task.
        free(job);
    }
}

void enqueueTask(ThreadPool p, Func f, Args args) {
    _ThreadPool* pool = (_ThreadPool *) p; 
    work_t* job; ;
    if ((job = malloc( sizeof(work_t) )) == NULL) {
        printf("Could not allocate memory for new job when enqueueing task.\n");
        exit(-1);
    }
    // Initialize the job
    job->task = f;
    job->args = args;
    job->next = NULL;
    // Now add the job to the queue.  First make sure no other threads can grab jobs while you're adding a new one.
    pthread_mutex_lock( &(pool->lock) );
    // When we're shutting down, we want to make sure not to enqueue new tasks.
    if (pool->reject_new_work) {
        free(job);
        pthread_mutex_unlock( &(pool->lock) );
        return;
    }
    // If there are existing jobs in the pool, then add this one to the end.  Otherwise start a new list.
    if (pool->size) {
        pool->tail->next = job;
        pool->tail = job;
        pool->size++; 
    }
    else {
        pool->head = job;
        pool->tail = job;
        pool->size++; 
        // Signal to the waiting threads that there is a new job to be processed
        pthread_cond_signal( &(pool->not_empty) );
    }
    // All done! Unlock the mutex.
    pthread_mutex_unlock( &(pool->lock) );
    return;
}

ThreadPool createThreadPool(int num_threads) {
    _ThreadPool* pool;
    if ((pool= malloc( sizeof(_ThreadPool) )) == NULL ) {
        printf("Memory unavailable for call to malloc(pool)\n");
        exit(-1);
    }
    
    pool->size = 0;
    pool->reject_new_work = 0;
    pool->num_threads = (num_threads > MAX_THREADS) ? MAX_THREADS : num_threads;
    pool->shutdown = 0;
    pool->head = NULL;
    pool->tail = NULL;

    if ((pool->threads = malloc( pool->num_threads * sizeof(pthread_t) )) == NULL) {
        printf("Memory unavailable for call to malloc(pthread_t)\n");
        exit(-1);
    }
    if( pthread_mutex_init(&(pool->lock),NULL) ) {
        printf("Error encountered when initializing pthread_mutex_t\n");
        exit(-1);
    }
    if( pthread_cond_init(&(pool->empty),NULL) ) {
        printf("Error encountered when initializing pthread_cond_t\n");	
        exit(-1); 
    } 
    if( pthread_cond_init(&(pool->not_empty),NULL) ) {
        printf("Error encountered when initializing pthread_cond_t\n");	
        exit(-1); 
    } 

    // Start up the threads
    for (int i = 0; i < pool->num_threads; i++) {
        if( pthread_create(&(pool->threads[i]), NULL, getWork, pool) ) {
            // Eject with error code if one of the threads fails to create 
            printf("Error encountered when spawning thread with pthread_create()\n");
            exit(-1);
        }
    }

    return (ThreadPool) pool;
}

void destroyThreadPool(ThreadPool p) {
    _ThreadPool* pool = (_ThreadPool *) p;
    
    pthread_mutex_lock( &(pool->lock) );
    pool->reject_new_work = 1;
    while (pool->size != 0) {
        // Wait here until we get the signal that the queue is empty and all tasks are complete.  The worker threads will be waiting at LINE (*)
        pthread_cond_wait( &(pool->empty), &(pool->lock) );
    }
    // Give the shutdown signal
    pool->shutdown = 1;
    // Broadcast to the waiting threads that they may now pass LINE (*) 
    pthread_cond_broadcast( &(pool->not_empty) );
    pthread_mutex_unlock( &(pool->lock) );
    for (int i = 0; i < pool->num_threads; i++) {
        // There may have been threads waiting at the initial mutex of our getWork function, in which case, they will not have received the OK signal. 
        pthread_cond_broadcast( &(pool->not_empty) );

        // FIXME: Master thread can pass all pthread_cond_broadcasts before any of the worker threads have had a chance to hit the cond_wait.  This really only matters when you call destroyThreadPool *immediately* after createThreadPool without doing any work in between.
        pthread_join(pool->threads[i], NULL);
    }
    // All the parallel threads have been terminated.  Free the resources we malloc'd
    free(pool->threads);
    pthread_mutex_destroy(&(pool->lock));
	pthread_cond_destroy(&(pool->empty));
	pthread_cond_destroy(&(pool->not_empty));
    free(pool);
	return;

}
/*************************************************************/

//  Utilities for matrix-vector multiplication using SIMD

/*************************************************************/

/*

Here is the basic idea I'm going for.  First, the function multiply_block uses AVX2 functions to quickly compute the matrix-vector multiplication of an 8x8 on an 8 component vector.  So, we store a general (m x n)-matrix as the upper left submatrix of a larger matrix tiled using 8x8 blocks, padding the remaining elements with zeros.  The horizontal strips of 8x8 blocks are stored as a dynamically allocated array of "Blocks".  When we want to compute a general matrix-vector product, we traverse the list for each strip, multiplying the relevant 8x8 blocks against the appropriate entries in the vector.  

To speed up execution of matrix multiplication, we spawn many threads to handle processing of all the strips

*/

// Store strips of 8x8 blocks as a linked list
typedef struct Block {
    float8 row[8]; // Rows of the block
} Block;

// Helper struct for distributed heap allocated memory to multiple threads without having to create a lot of copies
typedef struct LinearMapData {
    Block** A;                // 8 bytes The matrix, partitioned strips, each containing 8x8 blocks
    float8* v;                // 8 bytes The vector to multiply
    float8* result;           // 8 bytes Place to store the answer
    uint32_t num_rows;        // 4 bytes Number of 8-float chunks in v
    uint32_t num_cols;        // 4 bytes Number of 8-float chunks in v
    uint32_t num_row_blocks;  // 4 bytes Number of 8-float chunks in v
    uint32_t num_col_blocks;  // 4 bytes Number of 8-float chunks in v
} LinearMapData;

// Create/destroy functions defined below main
LinearMapData* createLinearMap(uint32_t num_rows, uint32_t num_cols);
void destroyLinearMap(LinearMapData* data);

static inline uint8 shuffle(void) {
    return _mm256_set_epi32(7,3,6,2,5,1,4,0);
}

void multiply_block(float8* rows, float8* col, float8* out) {
    /*  This picture is helping me understand right now. I'm tired.

    x * * * * * * * *  *
    y * * * * * * * *  *
    u * * * * * * * *  *
    v * * * * * * * *  *
    a * * * * * * * *  *
    b * * * * * * * *  *
    c * * * * * * * *  *
    d * * * * * * * *  *

    What is hadd doing:
    x1 x2 x3 x4 x5 x6 x7 x8 : y1 y2 y3 y4 y5 y6 y7 y8
    hadd again...
    x12 x34 y12 y34 x56 x78 y56 y78 
    a12 a34 b12 b34 a56 a78 b56 b78

    Gives this.
    x1234 y1234 a1234 b1234 x5678 y5678 a5678 b5678

    Then permute the bits to get this:
    x1234 x5678 y1234 y5678 a1234 a5678 b1234 b5678

    Meanwhile, do the same thing to the other rows:
    u1234 u5678 v1234 v5678 c1234 c5678 d1234 d5678

    A final hadd gives:

    x y u v a b c d

    Which is what we want.

    */
     
    // x1234 y1234 a1234 b1234 x5678 y5678 a5678 b5678
    //   0     4     1     5     2     6     3     7 
    // This permutation is implemented in static inline shuffle()

    // Multiply rows of A against the column vector
    /*
    buf[0] = _mm256_mul_ps(rows[0],*col);
    buf[1] = _mm256_mul_ps(rows[1],*col);
    buf[2] = _mm256_mul_ps(rows[2],*col);
    buf[3] = _mm256_mul_ps(rows[3],*col);
    buf[4] = _mm256_mul_ps(rows[4],*col);
    buf[5] = _mm256_mul_ps(rows[5],*col);
    buf[6] = _mm256_mul_ps(rows[6],*col);
    buf[7] = _mm256_mul_ps(rows[7],*col);
*/
    *out = _mm256_add_ps(*out, _mm256_hadd_ps(
                                    _mm256_permutevar8x32_ps(_mm256_hadd_ps(_mm256_hadd_ps(_mm256_mul_ps(rows[0],*col),_mm256_mul_ps(rows[1],*col)), _mm256_hadd_ps(_mm256_mul_ps(rows[4],*col),_mm256_mul_ps(rows[5],*col))), shuffle()),
                                    _mm256_permutevar8x32_ps(_mm256_hadd_ps(_mm256_hadd_ps(_mm256_mul_ps(rows[2],*col),_mm256_mul_ps(rows[3],*col)), _mm256_hadd_ps(_mm256_mul_ps(rows[6],*col),_mm256_mul_ps(rows[7],*col))), shuffle())
                                ));
    return;
}

void affineMap(LinearMapData* data, float8* bias);

/*
// Thread concurrent AVX2 matrix-vector multiplication Ax.  
void* dot(void* args) {
    
    ThreadArgs* arg = (ThreadArgs *) args;

    Block** A = arg->data->A;
    
    // Allocate strips to this thread
    int strips_per_thread = arg->data->num_row_blocks / MAX_THREADS + 1, strip;

    for (int j = 0; j < strips_per_thread; j++) {
       
        // Figure out which strip of 8 rows we're currently processing
        strip = strips_per_thread * arg->threadID + j;
        
        // We might have more threads than strips to process; if we do, just end the thread.
        if (strip > arg->data->num_row_blocks - 1) {
            pthread_exit(NULL);
        }
        
        // Otherwise, for each block of columns, multiply the block and store the result.
        for (int i = 0; i < arg->data->num_col_blocks; i++) multiply_block(A[strip][i].row, &arg->data->v[strip], &arg->data->result[strip]);

    }
    // After processing all strips, end the thread.
    // pthread_exit(NULL);
    return NULL;
}

// This is the function that dispatches work to all the threads.  The k'th thread will be assigned all the rows j such that floor( j / MAX_THREADS ) = k
void linearMap(LinearMapData* data) {

    // Initialize an array of threads and the arguments that will be passed to them
    pthread_t threads[MAX_THREADS];
    ThreadArgs args[MAX_THREADS];
    int rc; 
    int MAX_THREAD_SPAWN = (data->num_row_blocks < MAX_THREADS) ? data->num_row_blocks : MAX_THREADS;

    for (int i = 0; i < MAX_THREAD_SPAWN; i++) {
        args[i].data = data;
        args[i].threadID = i;
    }        
}
*/
void delete_rbm(RBM* r) {

}


// Intrinsic exponential function
float8 _mm256_exp_ps(float8 x) {
    // Evaluates exp(x) using Taylor series + Horner's method
    float8 b = _mm256_set1_ps( (float) 1 / 40320 );
    b = _mm256_fmadd_ps(b,x, _mm256_set1_ps( (float) 1 / 5040 ));
    b = _mm256_fmadd_ps(b,x, _mm256_set1_ps( (float) 1 / 720 ));
    b = _mm256_fmadd_ps(b,x, _mm256_set1_ps( (float) 1 / 120 ));
    b = _mm256_fmadd_ps(b,x, _mm256_set1_ps( (float) 1 / 24 ));
    b = _mm256_fmadd_ps(b,x, _mm256_set1_ps( (float) 1 / 6 ));
    b = _mm256_fmadd_ps(b,x, _mm256_set1_ps( (float) 1 / 2 ));
    b = _mm256_fmadd_ps(b,x, _mm256_set1_ps( (float) 1 ));
    b = _mm256_fmadd_ps(b,x, _mm256_set1_ps( (float) 1 ));
    return b;
}

// Broadcasted sigmoid function over a vector of length N
float* sigmoid(float* x, unsigned int N) {
    
    float8 one = _mm256_set1_ps(1.0);
    float8 zero = _mm256_set1_ps(0.0);
    float* result = (float *) _mm_malloc(N*sizeof(float),32);
    if(result == NULL) return 0;

    for (int i = 0; i < N; i += 8) {
        float8 data = _mm256_load_ps(x + i);
        // Compute 1 / (1 + exp(-x)) on 8 consecutive elements
        data = _mm256_div_ps(one,  _mm256_add_ps( _mm256_exp_ps( _mm256_sub_ps(zero, data)), one ));
        // Store the result 
        _mm256_store_ps(result + i, data);
    }
    return result;

}

void testFunction(void* out) {
    for (int i = 0; i < NUM_TRIALS; i++) {
        *(int *)out += 1;
    }
    return;
}

int main(void) {

    // Initialize a vector of ones for sigmoid test
    // float __attribute__ (( aligned(32) )) ones[2048] = {[0 ... 2047] = 1.0};
    /*    
    LinearMapData* data = createLinearMap(16,1000);
    
    struct timespec tstart={0,0}, tend={0,0};
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    for (int i = 0; i < NUM_TRIALS; i++)  linearMap(data);
    clock_gettime(CLOCK_MONOTONIC, &tend);
    printf("Matrix multiplication took about %lu nanoseconds\n",
           (tend.tv_nsec - tstart.tv_nsec)/NUM_TRIALS);
    
    float* res2 = (float *) data->result;
    printf("%f %f %f %f %f %f %f %f \n", res2[0],res2[1],res2[2],res2[3],res2[4], res2[5], res2[6], res2[7]);
    
    destroyLinearMap(data);
//    _mm_free(res);
    */

    /* TESTING THREADPOOL IMPLEMENTATION */
    int num_threads = 4; 
    clock_t start, end;
    double cpu_time_used; 
    ThreadPool p = createThreadPool(num_threads); 
    int* out = malloc(8*sizeof(int));
    memset(out,0,8*sizeof(int));

    start = clock();
    for (int j = 0; j < 8; j++) {
        enqueueTask(p, &testFunction, (out+j));
    }
    end = clock();
    cpu_time_used = 1e9*((double) (end - start)) / CLOCKS_PER_SEC; 
    printf("Multithreaded version took about %lf nanoseconds\n", cpu_time_used);
    sleep(1); 
    destroyThreadPool(p);

    printf("%d\n", out[0]);
    printf("%d\n", out[1]);
    printf("%d\n", out[2]);
    printf("%d\n", out[3]);
    printf("%d\n", out[4]);
    printf("%d\n", out[5]);
    printf("%d\n", out[6]);
    printf("%d\n", out[7]);
    memset(out, 0, 8*sizeof(int)); 
    free(out);
   /* 
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    for (int i = 0; i < NUM_TRIALS; i++)  testFunction(&out);
    clock_gettime(CLOCK_MONOTONIC, &tend);
    printf("Normal version took about %lu nanoseconds\n",
           (tend.tv_nsec - tstart.tv_nsec)/NUM_TRIALS);
    printf("%d\n", out);
   */ 
    return 0;

}


/*

STRUCT CREATE/DESTROY FUNCTION DEFINITIONS

*/

// Initializes a linear map data for the linear system Ax = 0 with A = 0 and x = 0
LinearMapData* createLinearMap(uint32_t num_rows, uint32_t num_cols) {
    
    LinearMapData* data = malloc(sizeof(LinearMapData));
     
    data->num_rows = num_rows;
    data->num_cols = num_cols;
    // The number of 8x8 blocks required to embed the matrix A in the upper left corner
    data->num_row_blocks = ( num_rows % 8 ) ? num_rows/8 + 1 : num_rows / 8;
    data->num_col_blocks = ( num_cols % 8 ) ? num_cols/8 + 1 : num_cols / 8;
    
    // Initialize the column vector v and set it to zero.
    data->v = _mm_malloc(data->num_col_blocks * sizeof(float8), AVX2_BYTE_ALIGNMENT);
    for (int i = 0; i < data->num_col_blocks; i++) data->v[i] = _mm256_set1_ps(0.0); 
    
    // Initialize the output vector result = Av and set to zero
    data->result = _mm_malloc(data->num_row_blocks * sizeof(float8), AVX2_BYTE_ALIGNMENT);
    for (int i = 0; i < data->num_row_blocks; i++) data->result[i] = _mm256_set1_ps(0.0); 
    
    data->A = malloc(data->num_row_blocks * sizeof(Block *));
    for (int i = 0; i < data->num_row_blocks; i++) {
        data->A[i] = malloc(data->num_col_blocks * sizeof(Block) );
        for (int j = 0; j < data->num_col_blocks; j++ ) {
            for (int k = 0; k < 8; k++) data->A[i][j].row[k] = _mm256_set1_ps(0.0);
        }
    }

    return data;
}

void destroyLinearMap(LinearMapData* data) { 
    for (int i = 0; i < data->num_row_blocks; i++) {
        free(data->A[i]);
    }
    free(data->A);
    _mm_free(data->result);
    _mm_free(data->v);
    free(data);
    data = NULL;
    return;
}
