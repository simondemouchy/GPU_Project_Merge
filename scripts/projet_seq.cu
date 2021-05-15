#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Timer.h"

//firstly, we write the sequential merge path, called as "Algorithm 1" in the project desciption

void seq_merge_path(int *A, int *B, int *M, int sizeA, int sizeB, int sizeM){
    int i=0, j=0; 
    while(i+j < sizeM){
        if(i>=sizeA){
            M[i+j] = B[j];
            j+=1;
        }
        else if(j>= sizeB || A[i]<B[j]){
            M[i+j] = A[i];
            i+=1;
        }
        else{
            M[i+j] = B[j];
            j+=1;
        }
    }
}


int main(){

    // Memory allocation on CPU
    // We define 2 ordered arrays of different size
    int *A, *B, *M, LA, LB, LM;
    LA = 250000;
    LB = 500000;
    LM = LA+LB;
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

    // CPU timer instructions
    Timer TimerAddOne;							
    TimerAddOne.start();	

    // Run sequential algorithm 
    seq_merge_path(A,B,M,LA,LB,LM); 

     // Print execution report
    TimerAddOne.add();							
	printf("EXECUTION TIME :\n Timer for the CPU Merge Path algorithm : %f ms (i.e. %f s)\n", 
           (float)TimerAddOne.getsum()*1000, (float)TimerAddOne.getsum());	
    printf("PARAMETERS :\n");
    printf(" Size of A : %d\n",LA);
    printf(" Size of B : %d\n",LB);
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
    
    // Free memory
    free(A);
    free(B);
    free(M); 

    return 0;
}