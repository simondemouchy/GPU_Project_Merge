#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Timer.h"

#define L (11)
#define numbBlocks (1)
#define threadsPerBlock (2*L/numbBlocks)

//Kernel definition
__global__ void par_merge_path_k(int *aGPU, int *bGPU, int *mGPU, int size){
    int i = threadIdx.x;
    int Kx; 
    int Ky;
    //int Px;   
    int Py; 
    int offset; 
    int Qx; 
    int Qy; 
    if(i>size){
        Kx = i -size;
        Ky = size;
        //Px = size;
        Py = i -size; 
    }
    else{
        Kx = 0;
        Ky = i;
        //Px = i;
        Py = 0; 
    }
    while(true){
        offset = abs(Ky-Py)/2;
        Qx = Kx + offset; 
        Qy = Ky - offset; 
        if((Qy>=0) && (Qx <= size) && (Qy ==size || Qx==0 || aGPU[Qy]>bGPU[Qy-1]) ){
            if((Qx==size) || (Qy==0) ||(aGPU[Qy-1]<=bGPU[Qx])){
                if((Qy<size) && (Qx==size || aGPU[Qy]<= bGPU[Qx])){
                    mGPU[i] = aGPU[Qy]; 
                }
                else{
                    mGPU[i] = bGPU[Qx]; 
                }
                break; 
            }
            else{
                Kx = Qx + 1;
                Ky = Qy - 1;  
            }
        }
        else{
            //Px = Qx -1 ; 
            Py = Qy +1; 
        } 
    }   
}


// Wrapper definition 
void par_merge_path(int *a, int *b, int *m){

    // Memory allocation on GPU
    int *AGPU, *BGPU, *MGPU;
    cudaMalloc(&AGPU, L*sizeof(int));
    cudaMalloc(&BGPU, L*sizeof(int));    
    cudaMalloc(&MGPU, 2*L*sizeof(int));

    // Transfert from Host (CPU) To Device (GPU)
    cudaMemcpy(AGPU, a, L*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(BGPU, b, L*sizeof(int), cudaMemcpyHostToDevice);

	// kernel invocation with L threads
    par_merge_path_k<<<numbBlocks, threadsPerBlock>>>(AGPU, BGPU, MGPU, L);
    cudaDeviceSynchronize();

    // Transfert from Device To Host
    cudaMemcpy(m, MGPU, 2*L*sizeof(int),cudaMemcpyDeviceToHost);  

   // Free memory on GPU
    cudaFree(AGPU);
    cudaFree(BGPU);
    cudaFree(MGPU);    
}


int main(){

    float TimerAddOne;								// GPU timer instructions
	cudaEvent_t start, stop;						// GPU timer instructions
	cudaEventCreate(&start);				        // GPU timer instructions
	cudaEventCreate(&stop);				            // GPU timer instructions
	cudaEventRecord(start,0);				        // GPU timer instructions

    // Memory allocation on CPU
    int *A, *B, *M; 
    A = (int*)malloc(L*sizeof(int));
    B = (int*)malloc(L*sizeof(int));
    M = (int*)malloc(2*L*sizeof(int));

    // Initialize input vectors
    for(int i = 0; i < L; i++){
        A[i] = 2*i;
        B[i] = 2*i+1;
    }
    
    //Wrapper 
    par_merge_path(A, B, M);
     
    cudaEventRecord(stop,0);				                // GPU timer instructions
	cudaEventSynchronize(stop);			                    // GPU timer instructions
	cudaEventElapsedTime(&TimerAddOne,start, stop);			// GPU timer instructions
 
	printf("GPU Timer for the addition on the GPU of scalars : %f ms\n", TimerAddOne);    


    for(int i = 0; i < 2*L; i++){
        printf("Result : %d \n", M[i]);

    }
    
    // Free memory on CPU
    free(A);
    free(B);
    free(M); 
           
    return 0;
}

