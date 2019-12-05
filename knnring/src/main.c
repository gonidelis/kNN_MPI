#include <stdio.h>
#include <stdlib.h>
#include "../inc/knnring.h"
#include <cblas.h>
#include <math.h>
#include <time.h>

int main()
{
    srand(time(NULL));

    int n= 8;
    int m= 2;
    int d= 2;
    int k = 2;


    double *X = malloc(n*d *sizeof(double));
    double *Y = malloc(m*d *sizeof(double));

    for(int i=0 ; i<n*d; i++)
    {

        X[i] = (double)rand()/(double)RAND_MAX;
    }

    for(int j=0 ; j<m*d; j++)
    {
        Y[j] = (double)rand()/(double)RAND_MAX;
    }

    printf("X matrix is : \n");
    for(int i=0; i<n; i++){
        for(int j=0; j<d; j++)
        {
            printf("%f,", X[i+j*n]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Y matrix is : \n");
    for(int i=0; i<m; i++){
        for(int j=0; j<d; j++)
        {
            printf("%f,", Y[i+j*m]);
        }
        printf("\n");
    }
    printf("\n");


    //select k nearest neighbors
    knnresult knnres = kNN(X, Y,n, m,d,k);
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<k;j++)
        {
            printf("Index: %d\n",knnres.nidx[j*k+i] );
            printf("Distance: %f\n",knnres.ndist[j*m+i] );
        }
    }

    return 0;
}

