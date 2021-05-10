#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Timer.h"

#define LA (250000)
#define LB (500000)
#define LM  (LA+LB)
#define N (100)

void seq_merge_path(int** A, int** B, int** M, int sizeA, int sizeB, int sizeM){
    for(int idx_row = 0; idx_row < N; idx_row++){
        int i=0, j=0; 
        while(i+j < sizeM){
            if(i>=sizeA){
                M[idx_row][i+j] = B[idx_row][j];
                j+=1;
            }
            else if(j>= sizeB || A[idx_row][i]<B[idx_row][j]){
                M[idx_row][i+j] = A[idx_row][i];
                i+=1;
            }
            else{
                M[idx_row][i+j] = B[idx_row][j];
                j+=1;
            }
        }
    }
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
    seq_merge_path(mat_A, mat_B, mat_M, LA, LB, LM);
    
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