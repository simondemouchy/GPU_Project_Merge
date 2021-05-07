#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

    //Memory allocation
    int *A, *B, *M, L;
    L = 500000; 
    A = (int*)malloc(L*sizeof(int));
    B = (int*)malloc(L*sizeof(int));
    M = (int*)malloc(2*L*sizeof(int));

    //Initialize input vectors 
    for(int i = 0; i < L; i++){
        A[i] = 2*i;
        B[i] = 2*i+1;
    }

    // CPU timer instructions
    Timer TimerAddOne;							
    TimerAddOne.start();	

    //Run sequential algorithm 
    seq_merge_path(A,B,M,L); 

    //Print execution time 
    TimerAddOne.add();							
	printf("CPU Timer for the merge path sequential algorithm on the CPU : %f s\n", 
           (float)TimerAddOne.getsum());	

    //Print values to check results 
    printf("BEGINNING OF ARRAYS :\n");
    printf("Array A : ");
    for(int i = 0; i < 10; i++){
        printf("%d, ", A[i]);
    } 
    printf("...\n");
    printf("Array B : ");
    for(int i = 0; i < 10; i++){
        printf("%d, ", B[i]);
    } 
    printf("...\n");
    printf("Array M : ");
    for(int i = 0; i < 20; i++){
        printf("%d, ", M[i]);
    } 
    printf("...\n");
    printf("END OF ARRAYS :\n");
    printf("Array A : ...");
    for(int i = L-10; i < L; i++){
        printf(", %d", A[i]);
    } 
    printf("\n");
    printf("Array B : ...");
    for(int i = L-10; i < L; i++){
        printf(", %d", B[i]);
    } 
    printf("\n");
    printf("Array M : ...");
    for(int i = 2*L-20; i < 2*L; i++){
        printf(", %d", M[i]);
    } 
    printf("\n");
    
    //Free memory
    free(A);
    free(B);
    free(M); 

    return 0;
}