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
    uint32_t r, c; // 8 bytes. Smallest multiples of 8 greater than v_dim and h_dim
    float8** W; // Store the rows of W
    float8** WT; // Also store the columns of W
};

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

/*************************************************************/

//  Utilities for matrix-vector multiplication using SIMD and multithreaded processing

/*************************************************************/

// Helper struct for distributed heap allocated memory to multiple threads without having to create a lot of copies
typedef struct Av_add_b {
    // This packs nicely into a 32-byte struct
    float8** A;      // 8 bytes
    float8* v;       // 8 bytes
    float8* b;       // 8 bytes
    float8* result;  // 8 bytes
    uint32_t l;     // 4 bytes
};

typedef struct ThreadArgs {
    Av_add_b* data;
    uint32_t threadID;
};

void multiply_block(float8* rows, float8 col, float8* out) {
     float temp[8];
     temp[0] = _mm256_mul_ps(rows[0],col);
}
// Block 11 * y1 + Block12 * y2 + ...
/*  This picture is helping me understand right now. I'm tired.

* * * *  *
* * * *  *
* * * *  *
* * * *  *

*/


// This is the function that dispatches work to all the threads.  The k'th thread will be assigned all the rows j such that floor( j / MAX_THREADS ) = k
void affine(Av_add_b* data) {

    // Initialize an array of threads and the arguments that will be passed to them
    pthread_t threads[MAX_THREADS];
    ThreadArgs args[MAX_THREADS];
    
    data->result = data->b;

    for (int i = 0; i < MAX_THREADS; i++) {
        args[i].data = data;
        args[i].threadID = i;
        if(pthread_create(  threads[i], 
                            NULL, 
                            dot, 
                            &args[i])) {
            // Eject with error code if one of the threads fails to create 
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
    // These choices distribute rows evenly over all threads 
    uint32_t first_row = args->threadID * (1 + MAX_ROWS / MAX_THREADS);
    uint32_t row = first_row; 
    
    // Pull some pointers out of Av* to make code more readable
    float* y = argv->data->v;
    uint32_t l = argv->data->l;
    
    while ( (row < MAX_ROWS) && (row / MAX_THREADS == first_row) ) {

        float8 x8, y8, prod, sum;

        // This is the row of the matrix we're currently working with
        float* x = argv->data->A[row];

        // Go through the elements 8 at a time
        for(int i = 0; i < l; i += 8) {
            // Load 8 values 
            x8 = _mm256_load_ps(x + i);
            y8 = _mm256_load_ps(y + i);
            // Take the product of those values
            prod = _mm256_mul_ps(x8,y8);
            // Horizontal sum
            sum = _mm256_hadd_ps(prod, _mm256 
            argv->data->result[row] += _mm256_cvtss_f32(sum);

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

