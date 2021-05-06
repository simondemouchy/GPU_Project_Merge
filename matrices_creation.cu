#include <stdio.h>
#include <stdlib.h>

#define L (512)
#define N (100)

int main(){

    int mat_A[N][L];
    int mat_B[N][L];
    
    int (*d_A)[N];      //pointers to arrays of dimension N
    int (*d_B)[N];      //pointers to arrays of dimension N

    //allocate values to matrices
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < L; j++) {
            mat_A[i][j] = 2*j;
            mat_B[i][j] = 2*j+1;
        }
    }   

    //Display results
    printf("Result A: ");
    for(int i = 0; i < 10; i++){
        for(int j =0; j<L; j++){
            printf("%d, ", mat_A[i][j]);
        }
    }

    printf("\n");

    printf("Result B: ");
    for(int i = 0; i < 10; i++){
        for(int j =0; j<L; j++){
            printf("%d, ", mat_B[i][j]);
        }
    }

    //allocation
    cudaMalloc((void**)&d_A, (N*L)*sizeof(int));
    cudaMalloc((void**)&d_B, (N*L)*sizeof(int));
           
    return 0;
}

