#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Timer.h"

// Define vectors size (>20)
#define LA (1000000)
#define LB (1000000)
#define LM (LA+LB)
#define numbBlocks (1)
#define threadsPerBlock (1024)

// Kernel definition
__global__ void par_merge_path_k(int *aGPU, int *bGPU, int *mGPU, int sizeA, int sizeB, int loop_idx){
    int i = threadIdx.x + 1024*loop_idx;
    if(i<sizeA+sizeB){
        int Kx; 
        int Ky;
        //int Px;   
        int Py; 
        int offset; 
        int Qx; 
        int Qy; 
        if(i>sizeA){
            Kx = i -sizeA;
            Ky = sizeA;
            //Px = sizeA;
            Py = i -sizeA; 
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
            if((Qy>=0) && (Qx <= sizeB) && (Qy ==sizeA || Qx==0 || aGPU[Qy]>bGPU[Qx-1]) ){
                if((Qx==sizeB) || (Qy==0) ||(aGPU[Qy-1]<=bGPU[Qx])){
                    if((Qy<sizeA) && (Qx==sizeB || aGPU[Qy]<= bGPU[Qx])){
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
}

// Wrapper definition 
void par_merge_path(int *a, int *b, int *m){

    // Memory allocation on GPU
    int *AGPU, *BGPU, *MGPU;
    cudaMalloc(&AGPU, LA*sizeof(int));
    cudaMalloc(&BGPU, LB*sizeof(int));    
    cudaMalloc(&MGPU, LM*sizeof(int));

    // Transfert from Host (CPU) To Device (GPU)
    cudaMemcpy(AGPU, a, LA*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(BGPU, b, LB*sizeof(int), cudaMemcpyHostToDevice);

	// kernel invocation with L threads
    for(int loop_idx = 0; loop_idx  < int(LM/1024)+ 1; loop_idx ++){
        par_merge_path_k<<<numbBlocks, threadsPerBlock>>>(AGPU, BGPU, MGPU, LA, LB, loop_idx);
    }
    cudaDeviceSynchronize();

    // Transfert from Device To Host
    cudaMemcpy(m, MGPU, LM*sizeof(int),cudaMemcpyDeviceToHost);  

   // Free memory on GPU
    cudaFree(AGPU);
    cudaFree(BGPU);
    cudaFree(MGPU);    
}


int main(){
    
    // Memory allocation on CPU
    int *A, *B, *M; 
    A = (int*)malloc(LA*sizeof(int));
    B = (int*)malloc(LB*sizeof(int));
    M = (int*)malloc(LM*sizeof(int));

    // Initialize input vectors
    for(int i = 0; i < LA; i++){
        A[i] = 2*i;
    }
    for(int i = 0; i < LB; i++){
        B[i] = 2*i+1;
    }

    // GPU timer instructions - Initialisation
    float TimerAddOne;								
	cudaEvent_t start, stop;						
	cudaEventCreate(&start);				        
	cudaEventCreate(&stop);				            
	cudaEventRecord(start,0);

    // Wrapper 
    par_merge_path(A, B, M);
    
    // GPU timer instructions - End
    //!\\ Time is given in ms
    cudaEventRecord(stop,0);				                
	cudaEventSynchronize(stop);			                    
	cudaEventElapsedTime(&TimerAddOne,start, stop);
    
    // Print execution report
	printf("EXECUTION TIME :\n Timer for the GPU Merge Path algorithm : %f ms (i.e. %f s)\n", TimerAddOne, TimerAddOne/1000);    
    printf("PARAMETERS :\n");
    printf(" Size of A : %d\n",LA);
    printf(" Size of B : %d\n",LB);
    printf(" Number of blocks : %d\n",numbBlocks );
    printf(" Number of threads by block : %d\n",threadsPerBlock);
    printf(" Number of times a thread is crossed : %d\n", int(LM/1024)+1);
    printf("MERGING RESULTS :\n");
    printf(" Input A: [");
    for(int i = 0; i < 10; i++){
        printf("%d, ", A[i]);
    }
    printf("..., "); 
    for(int i = LA-10; i < LA-1; i++){
        printf("%d, ", A[i]);
    }
    printf("%d]\n", A[LA-1]);
    printf(" Input B: [");
    for(int i = 0; i < 10; i++){
        printf("%d, ", B[i]);
    }
    printf("..., "); 
    for(int i = LB-10; i < LB-1; i++){
        printf("%d, ", B[i]);
    }
    printf("%d]\n", B[LB-1]);
    printf(" Output M: [");
    for(int i = 0; i < 10; i++){
        printf("%d, ", M[i]);
    }
    printf("..., "); 
    for(int i = LM-10; i < LM-1; i++){
        printf("%d, ", M[i]);
    }
    printf("%d]\n", M[LM-1]);
    printf("\n");
    
    // Free memory on CPU
    free(A);
    free(B);
    free(M); 
           
    return 0;
}