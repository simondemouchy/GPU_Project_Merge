#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Timer.h"

// Define vectors size (>20)
#define LA (2000)
#define LB (2000)
#define LM (LA+LB)
#define N (100)
#define numbBlocks (N)
#define threadsPerBlock (1024) //(int** A, int** B, int** M, int sizeA, int sizeB, int sizeM)

// Kernel definition
__global__ void par_merge_path_k(int** aGPU, int** bGPU, int** mGPU, int sizeA, int sizeB){
    for(int loop_idx = 0; loop_idx  < int(LM/1024)+ 1; loop_idx ++){
        int i = blockIdx.x ;
        int j = threadIdx.x + threadsPerBlock*loop_idx;
        printf("M : %d\n", aGPU[i][j]);
        //if(i==0){printf("j:%d\n",j);};
        //printf("value %d\n",idx);
        // printf("Thread.x:\n");
        // printf("value %d\n",idx);
        if(j<sizeA+sizeB){
            int Kx; 
            int Ky;
            //int Px;   
            int Py; 
            int offset; 
            int Qx; 
            int Qy; 
            if(j>sizeA){
                Kx = j -sizeA;
                Ky = sizeA;
                //Px = sizeA;
                Py = j -sizeA; 
            }
            else{
                Kx = 0;
                Ky = j;
                //Px = j;
                Py = 0; 
            }
            while(true){
                offset = abs(Ky-Py)/2;
                Qx = Kx + offset; 
                Qy = Ky - offset; 
                if((Qy>=0) && (Qx <= sizeB) && (Qy ==sizeA || Qx==0 || aGPU[i][Qy]>bGPU[i][Qx-1]) ){
                    if((Qx==sizeB) || (Qy==0) ||(aGPU[i][Qy-1]<=bGPU[i][Qx])){
                        if((Qy<sizeA) && (Qx==sizeB || aGPU[i][Qy]<= bGPU[i][Qx])){
                            mGPU[i][j] = aGPU[i][Qy]; 
                        }
                        else{
                            mGPU[i][j] = bGPU[i][Qx]; 
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

// Wrapper definition 
void par_merge_path(int **a, int **b, int **m){

    // Memory allocation on GPU (int **)malloc(N * sizeof(int*))
    int **AGPU, **BGPU, **MGPU;
    cudaMalloc((void**)&AGPU, N*sizeof(int));
    cudaMalloc((void**)&BGPU, N*sizeof(int));
    cudaMalloc((void**)&MGPU, N*sizeof(int));

    // Transfert from Host (CPU) To Device (GPU)
    cudaMemcpy(AGPU, a, N*LA*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(BGPU, b, N*LB*sizeof(int), cudaMemcpyHostToDevice);

    // for (int i = 0; i < N; i++){
    //     cudaMalloc((void*)&AGPU[i], LA * sizeof(int));
    //     cudaMalloc((void*)&BGPU[i], LB * sizeof(int));
    //     cudaMalloc((void*)&MGPU[i], LM * sizeof(int));
    // }

    // cudaMemcpy(MGPU, m, N*LM*sizeof(int), cudaMemcpyHostToDevice);

	// kernel invocation with L threads
    par_merge_path_k<<<numbBlocks, threadsPerBlock>>>(AGPU, BGPU, MGPU, LA, LB);    
    cudaDeviceSynchronize();

    // Transfert from Device To Host
    cudaMemcpy(m, MGPU, N*LM*sizeof(int),cudaMemcpyDeviceToHost);  

   // Free memory on GPU
    cudaFree(AGPU);
    cudaFree(BGPU);
    cudaFree(MGPU);    
}


int main(){

    // Memory allocation on CPU
    int **mat_A = (int **)malloc(N * sizeof(int*));
    for(int i = 0; i < N; i++) mat_A[i] = (int *)malloc(LA * sizeof(int));

    int **mat_B = (int **)malloc(N * sizeof(int*));
    for(int i = 0; i < N; i++) mat_B[i] = (int *)malloc(LB * sizeof(int));

    int **mat_M = (int **)malloc(N * sizeof(int*));
    for(int i = 0; i < N; i++) mat_M[i] = (int *)malloc(LM * sizeof(int));

    // Initialize input matrices
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < LA; j++) {
            mat_A[i][j] = 2*j;
        }
    }   

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < LB; j++) {
            mat_B[i][j] = 2*j+1;
        }
    }   
    
    // CPU timer instructions
    Timer TimerAddOne;							
    TimerAddOne.start();	

    // Run sequential algorithm 
    par_merge_path(mat_A, mat_B, mat_M);
    
    // Print execution report
    TimerAddOne.add();							
    printf("EXECUTION TIME :\n Timer for the CPU Batch Merge Path algorithm : %f ms (i.e. %f s)\n", 
            (float)TimerAddOne.getsum()*1000, (float)TimerAddOne.getsum());	
    printf("PARAMETERS :\n");
    printf(" Shape of A : %d x %d\n",N,LA);
    printf(" Shape of B : %d x %d\n",N,LB);
    printf("MERGING RESULTS :\n");
    printf("Input A:\n");
    for(int i = 0; i < 5; i++){
        if(i==0){printf("[[");}
        else{printf(" [");}
        for(int j =0; j<15; j++){
            printf("%d, ", mat_A[i][j]);
        }
        printf("...");
        for(int j =LA-15; j<LA; j++){
            printf(", %d", mat_A[i][j]);
        }
        printf("]\n");
    }
    printf(" ...]\n");
    printf("Input B:\n");
    for(int i = 0; i < 5; i++){
        if(i==0){printf("[[");}
        else{printf(" [");}
        for(int j =0; j<15; j++){
            printf("%d, ", mat_B[i][j]);
        }
        printf("...");
        for(int j =LB-15; j<LB; j++){
            printf(", %d", mat_B[i][j]);
        }
        printf("]\n");
    }
    printf(" ...]\n");
    printf("Output M:\n");
    for(int i = 0; i < 5; i++){
        if(i==0){printf("[[");}
        else{printf(" [");}
        for(int j =0; j<15; j++){
            printf("%d, ", mat_M[i][j]);
        }
        printf("...");
        for(int j =LM-15; j<LM; j++){
            printf(", %d", mat_M[i][j]);
        }
        printf("]\n");
    }
    printf(" ...]\n");

    // Free memory
    free(mat_A);
    free(mat_B);
    free(mat_M);
           
    return 0;
}