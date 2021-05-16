#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Timer.h"

// Define vectors size (>20)
#define N 100
#define LA 2500
#define LB 5000
#define LM (LA + LB)
#define threadsPerBlock (1024)

// Kernel definition
__global__ void par_merge_path_k(int A[][LA], int B[][LB], int C[][LM]){
    for(int loop_idx = 0; loop_idx  < int(LM/1024)+ 1; loop_idx ++){
        int idx = blockIdx.x;
        int i = threadIdx.x + 1024*loop_idx; 
        int sizeA = LA; 
        int sizeB = LB;
        if(i<LM){
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
                if((Qy>=0) && (Qx <= sizeB) && (Qy ==sizeA || Qx==0 || A[idx][Qy]>B[idx][Qx-1]) ){
                    if((Qx==sizeB) || (Qy==0) ||(A[idx][Qy-1]<=B[idx][Qx])){
                        if((Qy<sizeA) && (Qx==sizeB || A[idx][Qy]<= B[idx][Qx])){
                            C[idx][i] = A[idx][Qy]; 
                        }
                        else{
                            C[idx][i] = B[idx][Qx]; 
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
}


int main(){

    // Initialize input matrices
	int A[N][LA]; 
	int B[N][LB]; 
	int M[N][LM]; 

	for(int i = 0; i < N; i++) {
        for(int j = 0; j < LA; j++) {
            A[i][j] = 2*j+i*10;
        }
    }   

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < LB; j++) {
            B[i][j] = 2*j+i*10+1 ;
        }
    }   

	for(int i = 0; i < N; i++) {
        for(int j = 0; j < LM; j++) {
            M[i][j] = 2*j+1;
        }
    }

    // GPU timer instructions - Initialisation
    float TimerAddOne;								
	cudaEvent_t start, stop;						
	cudaEventCreate(&start);				        
	cudaEventCreate(&stop);				            
	cudaEventRecord(start,0);

    int (*AGPU)[LA], (*BGPU)[LB], (*MGPU)[LM];
	
	//Memories alloc on device
	cudaMalloc((void**)&AGPU, N*LA*sizeof(int));
	cudaMalloc((void**)&BGPU, N*LB*sizeof(int));
	cudaMalloc((void**)&MGPU, N*LM*sizeof(int));

	//Transfer from host to device
	cudaMemcpy(AGPU, A, N*LA*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(BGPU, B, N*LB*sizeof(int), cudaMemcpyHostToDevice);

	//Call the kernel
	par_merge_path_k << <N, threadsPerBlock >> >(AGPU, BGPU, MGPU);

	//Transfer from device to host
	cudaMemcpy(M, MGPU, N*LM*sizeof(int), cudaMemcpyDeviceToHost);

    // GPU timer instructions - End
    //!\\ Time is given in ms
    cudaEventRecord(stop,0);				                
	cudaEventSynchronize(stop);			                    
	cudaEventElapsedTime(&TimerAddOne,start, stop);

    // Print execution report
	printf("EXECUTION TIME :\n Timer for the GPU Batch Merge Path algorithm : %f ms (i.e. %f s)\n", TimerAddOne, TimerAddOne/1000);  
    printf("PARAMETERS :\n");
    printf(" Shape of A : %d x %d\n",N,LA);
    printf(" Shape of B : %d x %d\n",N,LB);
    printf(" Number of blocks : %d\n",N );
    printf(" Number of threads by block : %d\n",threadsPerBlock);
    printf(" Number of times the threads are crossed : %d\n", int(LM/1024)+1);
    printf("MERGING RESULTS :\n");
    printf("Input A:\n");
    for(int i = 0; i < 5; i++){
        if(i==0){printf("[[");}
        else{printf(" [");}
        for(int j =0; j<15; j++){
            printf("%d, ", A[i][j]);
        }
        printf("...");
        for(int j =LA-15; j<LA; j++){
            printf(", %d", A[i][j]);
        }
        printf("]\n");
    }
    printf(" ...]\n");
    printf("Input B:\n");
    for(int i = 0; i < 5; i++){
        if(i==0){printf("[[");}
        else{printf(" [");}
        for(int j =0; j<15; j++){
            printf("%d, ", B[i][j]);
        }
        printf("...");
        for(int j =LB-15; j<LB; j++){
            printf(", %d", B[i][j]);
        }
        printf("]\n");
    }
    printf(" ...]\n");
    printf("Output M:\n");
    for(int i = 0; i < 5; i++){
        if(i==0){printf("[[");}
        else{printf(" [");}
        for(int j =0; j<15; j++){
            printf("%d, ", M[i][j]);
        }
        printf("...");
        for(int j =LM-15; j<LM; j++){
            printf(", %d", M[i][j]);
        }
        printf("]\n");
    }
    printf(" ...]\n");
	//Delocate memories 
	cudaFree(AGPU);
	cudaFree(BGPU);
	cudaFree(MGPU);

	//free(A), free(B), free(C);
	return 0;
}
