/*

This script provides utilities for training a restricted Boltzmann machine.  It is the C analogue of the Python scripts in the repository root directory.  We intend to get speedup here using Intel Intrinsics AVX2 SIMD instructions, together with multithreaded processing.

*/

#include<pthread.h>
#include<string.h>
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
#define NANOSECONDS_PER_SECOND 1000000000
#define AVX2_BYTE_ALIGNMENT 32
#define NUM_TRIALS 1000

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

//  Utilities for matrix-vector multiplication using SIMD and multithreaded processing

/*************************************************************/

/*

Here is what I'm going to try: keeping strips of 8 rows at a time.  

->    ->    ->    ****
            ->    ****
            ->    ****
            ->    ****

      ->    ->    ****
            ->    ****
            ->    ****
            ->    ****
*/

// Helper struct for distributed heap allocated memory to multiple threads without having to create a lot of copies
typedef struct LinearMapData {
    float8*** A;              // 8 bytes The matrix, partitioned into 8 row strips
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


// Helper struct for passing multiple arguments to a threaded function
typedef struct ThreadArgs {
    LinearMapData* data;
    uint32_t threadID;
} ThreadArgs;

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
     
    float8 buf[8];

    // x1234 y1234 a1234 b1234 x5678 y5678 a5678 b5678
    //   0     4     1     5     2     6     3     7 
    // Implement the permutation above
    const float8 shuffle = _mm256_set_epi32(7,3,6,2,5,1,4,0);

    // Multiply rows of A against the column vector
    buf[0] = _mm256_mul_ps(rows[0],*col);
    buf[1] = _mm256_mul_ps(rows[1],*col);
    buf[2] = _mm256_mul_ps(rows[2],*col);
    buf[3] = _mm256_mul_ps(rows[3],*col);
    buf[4] = _mm256_mul_ps(rows[4],*col);
    buf[5] = _mm256_mul_ps(rows[5],*col);
    buf[6] = _mm256_mul_ps(rows[6],*col);
    buf[7] = _mm256_mul_ps(rows[7],*col);

    *out = _mm256_add_ps(*out, _mm256_hadd_ps(
                                    _mm256_permutevar8x32_ps(_mm256_hadd_ps(_mm256_hadd_ps(buf[0],buf[1]), _mm256_hadd_ps(buf[4],buf[5])), shuffle),
                                    _mm256_permutevar8x32_ps(_mm256_hadd_ps(_mm256_hadd_ps(buf[2],buf[3]), _mm256_hadd_ps(buf[6],buf[7])), shuffle)
                                ));
}

/*

We parallelize matrix-vector multiplication by using block_muliply to compute 8 values of the outupt vector in each thread - i.e. we break the matrix A up into 8 row strips, and assign each strip it's own thread.

*/


void affineMap(LinearMapData* data, float8* bias);

// Thread concurrent AVX2 matrix-vector multiplication Ax.  
// Both x and rows(A) must be 32-byte aligned and have the same length
// (void *) will get cast to (ThreadArgs *)
void* dot(void* args) {
    
    ThreadArgs* arg = (ThreadArgs *) args;
    float8 buf[8];

    float8*** A = arg->data->A;
    
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
        for (int i = 0; i < arg->data->num_col_blocks; i++) {

            buf[0] = A[strip][0][i];
            buf[1] = A[strip][1][i];
            buf[2] = A[strip][2][i];
            buf[3] = A[strip][3][i];
            buf[4] = A[strip][4][i];
            buf[5] = A[strip][5][i];
            buf[6] = A[strip][6][i];
            buf[7] = A[strip][7][i];
            
            multiply_block(buf, &arg->data->v[strip], &arg->data->result[strip]);

        }
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

    for (int i = 0; i < MAX_THREADS; i++) {
        args[i].data = data;
        args[i].threadID = i;
        if( (rc = pthread_create(  &threads[i], 
                            NULL, 
                            dot, 
                            &args[i]))) {
            // Eject with error code if one of the threads fails to create 
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
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
    LinearMapData* data = createLinearMap(16,1000);

    destroyLinearMap(data);
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
    data->v = _mm_malloc(data->num_col_blocks * sizeof(float8 *), AVX2_BYTE_ALIGNMENT);
    memset(data->v, 0, data->num_col_blocks * sizeof(float8 *));
   
    // Initialize the output vector result = Av and set to zero
    data->result = _mm_malloc(data->num_row_blocks * sizeof(float8 *), AVX2_BYTE_ALIGNMENT);
    memset(data->result, 0, data->num_row_blocks * sizeof(float8 *));
   
    data->A = malloc(data->num_row_blocks * sizeof(float8 **)); /*
    for (int i = 0; i < data->num_row_blocks; i++) {
        data->A[i] = malloc(8 * sizeof(float8 *));
        for (int j = 0; j < 8; j++) {
            data->A[i][j] = (float8 *) _mm_malloc(data->num_col_blocks * sizeof(float8), AVX2_BYTE_ALIGNMENT);
            for (int k = 0; k < data->num_col_blocks; k++) { 
                data->A[i][j][k] = _mm256_set1_ps(0.0);
            }
        }
    }
    */
    return data;
}

void destroyLinearMap(LinearMapData* data) { 
   /* 
    for (int i = 0; i < data->num_row_blocks; i++) {
        for (int j = 0; j < 8; j++) _mm_free(data->A[i][j]);
        free(data->A[i]);
    }*/
    free(data->A); 
    _mm_free(data->result); 
    _mm_free(data->v);
    free(data);
    data = NULL;
    return;
}
