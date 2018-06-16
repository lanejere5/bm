#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>

int main(void) {
    printf("Here are the alignments used by GCC on your architecture: \n");
    printf("char: %lu\n", __alignof__(char));
    printf("uchar: %lu\n", __alignof__(unsigned char));
    printf("short: %lu\n", __alignof__(short));
    printf("int: %lu\n", __alignof__(int));
    printf("uint: %lu\n", __alignof__(unsigned int));
    printf("uint32_t: %lu\n", __alignof__(uint32_t));
    printf("long int: %lu\n", __alignof__(unsigned long));
    printf("long unsigned: %lu\n", __alignof__(double));
    printf("float: %lu\n", __alignof__(float));
    printf("double: %lu\n", __alignof__(double));
    printf("long double: %lu\n", __alignof__(long double)); 
    printf("void *: %lu\n", __alignof__(void *));
    printf("char *: %lu\n", __alignof__(char *));
    printf("uchar *: %lu\n", __alignof__(unsigned char *));
    printf("short *: %lu\n", __alignof__(short *));
    printf("int *: %lu\n", __alignof__(int *));
    printf("uint *: %lu\n", __alignof__(unsigned int *));
    printf("long int *: %lu\n", __alignof__(unsigned long *));
    printf("long unsigned *: %lu\n", __alignof__(double *));
    printf("float *: %lu\n", __alignof__(float *));
    printf("double *: %lu\n", __alignof__(double *));
    printf("long double *: %lu\n", __alignof__(long double *));
    printf("float **: %lu\n", __alignof__(float **));
    printf("Here are the sizes of various data types on your architecture: \n");
    printf("char: %lu\n", sizeof(char));
    printf("uchar: %lu\n", sizeof(unsigned char));
    printf("short: %lu\n", sizeof(short));
    printf("int: %lu\n", sizeof(int));
    printf("uint: %lu\n", sizeof(unsigned int));
    printf("uint32_t: %lu\n", sizeof(uint32_t));
    printf("long int: %lu\n", sizeof(unsigned long));
    printf("long unsigned: %lu\n", sizeof(double));
    printf("float: %lu\n", sizeof(float));
    printf("double: %lu\n", sizeof(double));
    printf("long double: %lu\n", sizeof(long double)); 
    printf("void *: %lu\n", sizeof(void *));
    printf("char *: %lu\n", sizeof(char *));
    printf("uchar *: %lu\n", sizeof(unsigned char *));
    printf("short *: %lu\n", sizeof(short *));
    printf("int *: %lu\n", sizeof(int *));
    printf("uint *: %lu\n", sizeof(unsigned int *));
    printf("long int *: %lu\n", sizeof(unsigned long *));
    printf("long unsigned *: %lu\n", sizeof(double *));
    printf("float *: %lu\n", sizeof(float *));
    printf("double *: %lu\n", sizeof(double *));
    printf("long double *: %lu\n", sizeof(long double *));
    printf("float **: %lu\n", sizeof(float **));
    return 0;
}
