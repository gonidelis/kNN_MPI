#include <stdio.h>
#include <stdlib.h>
#include "../inc/knnring.h"
#include <cblas.h>
#include <math.h>


//swap used in quickselect

knnresult kNN(double *X, double *Y,  int n, int m, int d, int k)
{
    double *D;
    /*
    initialize indices[] array that will
    *give update knn_ring.nidx[] after q-select
    */
    int *indices = malloc(m*n*sizeof(int));
    for(int i=0 ; i<m*n; i++)
    {
        indices[i] = i%n;
    }

    knnresult knn_ring;
    knn_ring.m = m;
    knn_ring.k = k;

    //Matlab formula converted in C
    D = distances(X, Y, n, m, d);

    //allocate space for each query point's neighbors
    knn_ring.ndist = malloc(k*m*sizeof(double));
    knn_ring.nidx = malloc(k*m * sizeof(int));

    /*for k = 3
    *ndist =[ qpoint1.neigh1
            qpoint1.neigh2]
            qpoint1.neigh3
            qpoint2.neigh1
            qpoint2.neigh2
            qpoint2.neigh3
            ...]
    */

    /*
    printf("\nD matrix before q-select is:\n");
    for(int i=0 ; i<n ; i++)
    {
        for(int j=0; j<m; j++)
        {
            printf("%f, ", D[i+j*n]);
        }
        printf("\n");
    }
    */


    /*
    printf("\nindices[] matrix before:\n");
    for(int i=0 ; i<n ; i++)
    {
        for(int j=0; j<m; j++)
        {
            printf("%d         ", indices[i+j*n]);
        }
        printf("\n");
    }
    */

    for(int i=0 ; i<m; i++)
    {
        for(int j=k; j>0 ; j--)
        {
            /*
            *Find k-nearest neighbors using quickselect - sort
            *on D[] array.
            *Pass indices[] array as well in order
            *for it to update in parallel with D[].
            */
            if(j==k)
            {
                quickSelect(D, i*n, i*n+n, j, indices);
            }
            else
            {
                //printf("i*n - i*n+n-1 -> j+i*n-1 = %d , %d, %d\n", i*n,i*n+k, j+i*n-1);
                quickSelect(D, i*n, i*n+j, j, indices);
            }
        }
    }

    /*
    printf("\nD matrix after q-select is:\n");
    for(int i=0 ; i<n ; i++)
    {
        for(int j=0; j<m; j++)
        {
            printf("%f, ", D[i+j*n]);
        }
        printf("\n");
    }
    */
    /*
    printf("\nindices[] matrix after q-select:\n");
    for(int i=0 ; i<n ; i++)
    {
        for(int j=0; j<m; j++)
        {
            printf("%d         ", indices[i+j*n]);
        }
        printf("\n");
    }
    */

    for(int i=0; i<k; i++)
    {
        for(int j=0; j<m; j++)
        {
            //printf("j*n+i =%d, indices[j*n+i]=%d\n", j*n+i, indices[j*n+i]);
            knn_ring.nidx[i*m+j]=indices[j*n+i];
            knn_ring.ndist[i*m+j]=D[j*n+i];
        }
    }

    /*
    printf("ndix - ndist:\n");
    for(int i=0;i<m*k;i++)
    {
        printf("%d , %f\n",knn_ring.nidx[i], knn_ring.ndist[i]);
    }
    */

    return knn_ring;
}

double *distances(double *X, double *Y, int n, int m, int d)
{
    //D(istances) array
    double *D  = malloc(n*m*sizeof(double));
    //Z holds X*Y.' product
    double *Z= calloc(n*m, sizeof(double));
    //sumRows of X,Y squared
    double *Xrows;
    double *Yrows;



    //calculate Z product using BLAS
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, m, d, 1, X, n, Y, m, 0, Z, n);

    /*
    printf("\nZ (X*Y.') matrix is:\n");
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++)
        {
            printf("%f,", Z[i+j*m]);
        }
        printf("\n");
    }
    */

    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++)
        {
            Z[i+j*m] *= 2;
        }
    }



    //Calculate sumRow(X^2), sumRow(Y^2) terms
    Xrows = sumRowPow(X, n, d);
    Yrows = sumRowPow(Y, m, d);

    /*
    printf("\nXrows matrix keeps sumRows(X^2):\n");
    for(int i=0; i<n; i++)
    {
        printf("%f,", Xrows[i]);
        printf("\n");
    }
    printf("\nYrows matrix keeps sumRows(Y^2):\n");
    for(int i=0; i<m; i++)
    {
        printf("%f,", Yrows[i]);
        printf("\n");
    }
    */


    //printf("\nD matrix is:\n");
    for(int i=0 ; i<n ; i++)
    {
        for(int j=0; j<m; j++)
        {
            //! If dist < 0.000001, do it 0
            if(Xrows[i]+Yrows[j]-Z[i+j*n] < 0.0000001)
            {
                D[n*j+i]=0.0;
            }
            else
            {
                D[i+j*n] = sqrt(Xrows[i]+Yrows[j]-Z[i+j*n]);

            }

            //printf("%f,", D[i+j*n]);
        }
        //printf("\n");
    }

    free(Z);
    free(Xrows);
    free(Yrows);
    return D;
}

double *sumRowPow(double *X, int m , int d)
{
    /*Calculate sumRows(X^2), sumRows(Y^2) addends of the sum*/
    double *A = malloc(m * sizeof(double));
    for(int i=0 ; i<m; i++)
    {
        double sumrow = 0;
        for(int j=0; j<d; j++)
        {
            sumrow += X[i+j*m] * X[i+j*m];
        }
        A[i] = sumrow;
    }
    return A;
}


int partition(double arr[], int l, int r, int *indices)
{
	double x = arr[r], temp;
	int i = l;
	int tempidx;
	for (int j = l; j <= r - 1; j++) {
		if (arr[j] <= x) {
			//swap array
			temp=arr[i];
			arr[i]= arr[j];
			arr[j]= temp;
			//swap indices
			tempidx=indices[i];
			indices[i]= indices[j];
			indices[j]= tempidx;
			i++;
		}
	}
	//swap array
	temp=arr[i];
	arr[i]= arr[r];
	arr[r]= temp;
	//swap indices
	tempidx=indices[i];
	indices[i]= indices[r];
	indices[r]= tempidx;
	return i;
}

// This function returns k'th smallest
// element in arr[l..r] using QuickSort
// based method. ASSUMPTION: ALL ELEMENTS
// IN ARR[] ARE DISTINCT
double kthSmallest(double arr[], int l, int r, int k, int *indices)
{
	// If k is smaller than number of
	// elements in array
	if (k > 0 && k <= r - l + 1) {

		// Partition the array around last
		// element and get position of pivot
		// element in sorted array
		int index = partition(arr, l, r, indices);

		// If position is same as k
		if (index - l == k - 1)
			return arr[index];

		// If position is more, recur
		// for left subarray
		if (index - l > k - 1)
			return kthSmallest(arr, l, index - 1, k, indices);

		// Else recur for right subarray
		return kthSmallest(arr, index + 1, r, k - index + l - 1, indices);
	}

	// If k is more than number of
	// elements in array
	//return INT_MAX;
}

// Driver program to test above methods
void quickSelect(double *arr,int l, int r,int k, int *indices){

	kthSmallest(arr, l, r-1, k, indices);

	//printf("k is:%d\n",k);
	//printf("K-th smallest element is: %lf\n",kthSmallest(arr, 0, n - 1, k));
}
