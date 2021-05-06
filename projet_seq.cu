#include <stdio.h>
#include "Timer.h"

void seq_merge_path(int *A, int *B, int *M, int size){
    int card_M = 2*size; 
    int i=0, j=0; 
    while(i+j < card_M){
        if(i>=size){
            M[i+j] = B[j];
            j+=1;
        }
        else if(j>= size || A[i]<B[j]){
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
    Timer TimerAddOne;							// CPU timer instructions    

    //Memory allocation
    int *A, *B, *M, L;
    L = 5; 
    A = (int*)malloc(L*sizeof(int));
    B = (int*)malloc(L*sizeof(int));
    M = (int*)malloc(L*sizeof(int));

    TimerAddOne.start();							// CPU timer instructions

    for(int i = 0; i < L; i++){
        A[i] = 2*i;
        B[i] = 2*i+1;
    }

    //Run sequential algorithm 
    seq_merge_path(A,B,M,L); 

    TimerAddOne.add();							// CPU timer instructions
	printf("CPU Timer for the addition on the CPU of scalars : %f s\n", 
           (float)TimerAddOne.getsum());			// CPU timer instructions


    for(int i = 0; i < L; i++){
        printf("Result A : %d \n", A[i]);
        printf("Result B : %d \n", B[i]);
    }      
    for(int i = 0; i < 2*L; i++){
        printf("Result : %d \n", M[i]);

    }
    
    //Free memory
    free(A);
    free(B);
    free(M); 

    return 0;
}