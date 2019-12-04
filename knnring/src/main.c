#include <stdio.h>
#include <stdlib.h>
#include "../inc/knnring.h"
#include <cblas.h>
#include <math.h>
#include <time.h>


double dist(double *X, double *Y, int i, int j, int d, int n, int m){

  /* compute distance */
  double dist = 0;
  for (int l = 0; l < d; l++){
    dist += ( X[l*n+i] - Y[l*m+j] ) * ( X[l*n+i] - Y[l*m+j] );
    printf("X[%d]=%f, Y[%d]= %f\n",l*n+i, X[l*n+i],l*m+j, Y[l*m+j]);
  }

  return sqrt(dist);
}


// ==================
// === VALIDATION ===
// ==================

//! kNN validator
/*!
   The function asserts correctness of the kNN results by:
     (i)   Checking that reported distances are correct
     (ii)  Validating that distances are sorted in non-decreasing order
     (iii) Ensuring there are no other points closer than the kth neighbor
*/
int validateResult( knnresult knnres, double * corpus, double * query,
                    int n, int m, int d, int k ) {

  /* loop through all query points */
  for (int j = 0; j < m; j++ ){

    /* max distance so far (equal to kth neighbor after nested loop) */
    double maxDist = -1;

    /* mark all distances as not computed */
    int * visited = (int *) calloc( n, sizeof(int) );

    /* go over reported k nearest neighbors */
    for (int i = 0; i < k; i++ ){

      /* keep list of visited neighbors */
      visited[ knnres.nidx[i*m + j] ] = 1;

      /* get distance to stored index */
      double distxy = dist( corpus, query, knnres.nidx[i*m + j], j, d, n, m );

      /* make sure reported distance is correct */
      if ( abs( knnres.ndist[i*m + j] - distxy ) > 1e-8 ){
          printf("wrong distance\n");
          return 0;
      }

      /* distances should be non-decreasing */
      if ( knnres.ndist[i*m + j] < maxDist )
      {
          printf("distances are decreasing at %d, %d\n", j, i);
          return 0;
      }
      /* update max neighbor distance */
      maxDist = knnres.ndist[i*m + j];

    } /* for (k) -- reported nearest neighbors */

    /* now maxDist should have distance to kth neighbor */

    /* check all un-visited points */
    for (int i = 0; i < n; i++ ){

      /* check only (n-k) non-visited nodes */
      if (!visited[i]){

        /* get distance to unvisited vertex */
        double distxy = dist( corpus, query, i, j, d, n, m );

        /* point cannot be closer than kth distance */
        if ( distxy < maxDist )
        {
            printf("point closer than kth missed at %d, %d, distxy = %f\n", j,i, distxy);
            return 0;
        }
      } /* if (!visited[i]) */

    } /* for (i) -- unvisited notes */

    /* deallocate memory */
    free( visited );

  } /* for (j) -- query points */

  /* return */
  return 1;

}





int main()
{
    srand(time(NULL));

    int n= 8;
    int m= 2;
    int d= 2;
    int k = 2;


    double *X = malloc(n*d *sizeof(double));
    double *Y = malloc(m*d *sizeof(double));
    /*
    X = [1 2
        3 4
        5 6
        7 8]
    Y = [ 1 4
         2 5]
    D = [ 2.0000   3.1623
        2.0000   1.4142
        4.4721   3.1623
        7.2111   5.8310]
    */

    /*
    X[0] = 0.10;
    X[1] = 0.49;
    X[2] = 0.92;
    X[3] = 0.51;
    X[4] = 0.791;
    X[5] = 0.50;
    X[6] = 0.46;
    X[7] = 0.91;
    X[8] = 0.72;
    X[9] = 0.27;
    X[10]= 0.60;
    X[11] = 0.10;
    X[12] = 0.23;
    X[13] = 0.1;
    X[14] = 0.92;
    X[15] = 0.2;



    Y[0] = 0.30;
    Y[1] = 0.62;
    Y[2] = 0.64;
    Y[3] = 0.78;
    */


    for(int i=0 ; i<n*d; i++)
    {

        X[i] = (double)rand()/(double)RAND_MAX;
        //X[i+j*n]
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

    int isValid = validateResult( knnres, X, Y, n, m, d, k );

    printf("~~~~~~~%d~~~~~~~~~\n", isValid);



    return 0;
}

/*
knnresult kNN(double *X, double *Y, int n, int m, int d, int k)
{
    //full D array
    double *D  = malloc(n*m*sizeof(double));
    //Calculate the 2nd part using BLAS
    double *Z= calloc(n*m, sizeof(double));

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, d, 1, X, 4, Y, 2, 0, Z, 4);

    printf("\nZ matrix is:\n");
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++)
        {
            Z[i+j*m] *= 2;
            printf("%f,", Z[i+j*m]);
        }
        printf("\n");
    }

    //Calculate 1st and 3rd part of the sum

    //keep sumRows(X^2)
    double A[m];
    for(int i=0 ; i<m; i++)
    {
        int sumrow = 0;
        for(int j=0; j<d; j++)
        {
            X[i+j*m] *= X[i+j*m];
            sumrow += X[i+j*m];
        }
        A[i] = sumrow;
    }

    //keep sumRows(Y^2)
    double B[n];
    for(int i=0 ; i<n; i++)
    {
        int sumrow = 0;
        for(int j=0; j<d; j++)
        {
            Y[i+j*n] *= Y[i+j*n];
            sumrow += Y[i+j*n];
        }
        B[i] = sumrow;
    }


    printf("\nA matrix keep sumRows(X^2):\n");
    for(int i=0; i<m; i++){
        //for(int j=0; j<d; j++)
        {
            printf("%f,", A[i]);
        }
        printf("\n");
    }

    printf("\nB matrix keep sumRows(Y^2):\n");
    for(int i=0; i<n; i++){
        //for(int j=0; j<d; j++)
        {
            printf("%f,", B[i]);
        }
        printf("\n");
    }

    for(int i=0 ; i<m ; i++)
    {
        for(int j=0; j<n; j++)
        {
            D[i+j*m] = A[i]+B[j]-Z[i+j*m];
        }
    }

    printf("\n******************");
    printf("\nD matrix is:\n");
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++)
        {
            D[i+j*m] = sqrt(D[i+j*m]);
            printf("%f,", D[i+j*m]);
        }
        printf("\n");
    }


}
*/
