/*

This script provides utilities for training a restricted Boltzmann machine.  It is the C analogue of the Python scripts in the repository root directory.  We intend to get speedup here using Intel Intrinsics AVX2 SIMD instructions, together with multithreaded processing.

*/


#include<pthread.h>
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<immintrin.h>
#include<assert.h>
#include <time.h>

#define float4 __m128
#define float8 __m256
#define double2 __m128d
#define double4 __m256d
#define uint4 __m128i
#define uint8 __m256i
#define ulong4 __m256i
#define int4 __m128i
#define ulong2 __m128i

#define CACHE_SIZE 128
#define MAX_THREADS 1024

/****** DEFINE THE RBM STRUCT INITIALIZER / DESTORYER *******/

typedef struct RBM {
    uint32_t v_dim, h_dim; // 8 bytes
    float** W; // Some bytes - 8 byte aligned
};

RBM* init_rbm(unsigned int v, unsigned int h) {
    
    RBM* out = (RBM *) malloc( sizeof(RBM) );
    out->v_dim = v;
    out->h_dim = h;
    // The weights of the RBM are stored as a dynamically allocated set of rows
    if(out->W = malloc( h * sizeof(float *) ) == NULL) return NULL;

    for (int i = 0; i < h; i++) {
        // Allocate memory for each row
        if(out->W[i] = (float *) malloc( v * sizeof(float) )) return NULL;
    }
    
    return out;
}

void destroy_rbm(RBM* r) {
    // Free the memory being used to store each row 
    for (int i = 0; i < r->h_dim; i++) {
        free(out->W[i]);
        out->W[i] = NULL;
    }
    // Free the memory being used to store pointers to the rows
    free(out->W);
    out->W = NULL;
    // Finally free the memory being used to store the RBM
    free(r);
}

/*************************************************************/

//  Utilities for matrix-vector multiplication using SIMD and multithreaded processing

/*************************************************************/

// Helper struct for distributed heap allocated memory to multiple threads without having to create a lot of copies
typedef struct Av {
    // This packs nicely into a 32-byte struct
    float** A;       // 8 bytes
    float* v;       // 8 bytes
    float* result;  // 8 bytes
    uint32_t l;
};

typedef struct ThreadArgs {
    Av* data;
    uint32_t threadID;
};

// This is the function that dispatches work to all the threads.  The k'th thread will be assigned all the rows j such that floor( j / MAX_THREADS ) = k
void A_x(Av* data) {

    // Initialize an array of threads and the arguments that will be passed to them
    pthread_t threads[MAX_THREADS];
    ThreadArgs args[MAX_THREADS];
    
    for (int i = 0; i < MAX_THREADS; i++) {
        args[i].data = data;
        args[i].threadID = i;
        if(pthread_create(  threads[i], 
                            NULL, 
                            dot, 
                            &args[i])) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

}

// Thread concurrent AVX2 matrix-vector multiplication Ax.  
// Both x and rows(A) must be 32-byte aligned and have the same length
// (void *) will get cast to (ThreadArgs *)
void* dot(void* args) {
    
    ThreadArgs* argv = (ThreadArgs *) args;
    // 7 rows, 4 threads -> 7/4 + 1 = 2
    // Thread 0: 0,1
    // Thread 1: 2,3
    // Thread 2: 4,5
    // Thread 3: 6
    uint32_t first_row = args->threadID * (1 + MAX_ROWS / MAX_THREADS);
    uint32_t row = first_row; 

    float* y = argv->data->v;

    while ( (row < MAX_ROWS) && (row / MAX_THREADS == first_row) ) {

        float8 x8, y8, prod, sum;

        float* x = argv->data->A[row];

        for(int i = 0; i < l; i++) {
            // Load 8 values 
            x8 = _mm256_load_ps(x + i);
            y8 = _mm256_load_ps(y + i);
            // Take the product of those values
            prod = _mm256_mul_ps(x8,y8);
            // Horizontal sum
            
            argv->data->result[row] += _mm256_cvtss_f32 (__m256 a)

        }
        
        row++;
    }
}

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

int main(void) {

    // Initialize a vector of ones
    float __attribute__ (( aligned(32) )) ones[2048] = {[0 ... 2047] = 1.0};
    clock_t start, end;
    float* res = (float *) _mm_malloc(2048*sizeof(float),32);

    start = clock();
    for (int i = 0; i < 10000; i++) {
        res = sigmoid(ones, 2048); 
    }
    end = clock();
    printf("%f %f %f %f", res[0],res[100],res[245],res[1890]);
    printf("%lu nanoseconds\n", ((end - start) / (CLOCKS_PER_SEC / 100000 )));
   
    _mm_free(res);
    return 0;
}

