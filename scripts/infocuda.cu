#include <stdio.h>

//Function that catches the error
void testCUDA(cudaError_t error, const char *file, int line){

    if (error != cudaSuccess) {
       printf("There is an error in file %s at line %d\n", file, line);
       exit (EXIT_FAILURE);
    }
}

//Has to be defined in the comppilation in order to get the correct value of the 
//macro __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

 //Device code
__global__ void empty_k(void){
}

//Host code
int main (void){

    int count;
    cudaDeviceProp prop;
 
    empty_k<<<1,1>>>();
    testCUDA(cudaGetDeviceCount (&count));
    printf("The number of devices available is %i GPUs \n", count);
    testCUDA(cudaGetDeviceProperties(&prop, count -1));
    printf("Name: %s\n", prop.name);
    printf("Global memory size in octet (bytes): %u\n", prop.totalGlobalMem);
    printf("Shared memeory size per block: %ld\n", prop.sharedMemPerBlock);
    printf("Number of registers per block: %i\n", prop.regsPerBlock);
    printf("Number of threads in a warp: %i\n", prop.warpSize);
    printf("Maximum number of threads that can be launched per block: %i\n", prop.maxThreadsPerBlock);
    printf("Maximum number of threads that can be launched: %i x %i x %i\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
    printf("Maximum number  GridSize: %i X %i X %i\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Total constant memory size: %ld\n", prop.totalConstMem);
    printf("Clock rate; %i\n", prop.clockRate);
    return 0;
}
