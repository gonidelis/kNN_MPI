#include <stdio.h>
#include <stdlib.h>
#include "../inc/knnring.h"
#include <cblas.h>
#include <mpi.h>
#include <math.h>
#include <float.h>


knnresult distrAllkNN(double* X, int n,int d,int k)
{
    int id, world_size;
    double *query = X;

    double *corpus = malloc(n*d*sizeof(double));
    double *corpus_buff = malloc(n*d*sizeof(double));
    double *corpus_send = malloc(n*d*sizeof(double));

    int *indices = malloc(n*n*sizeof(int));
    double *D;
    knnresult knn_ring;

    knn_ring.ndist = malloc(k*n*sizeof(double));
    knn_ring.nidx = malloc(k*n*sizeof(int));

    MPI_Comm_rank(MPI_COMM_WORLD,  &id);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Status status;

    //query remains the same for each process in all iterations
    for (int i = 0; i < n*d ; i++)
    {
        corpus[i]=query[i];
    }

    knn_ring.m=n;
    knn_ring.k=k;

    for(int step = 0 ; step < world_size ; step++)
    {

        //even processes send first while odd process receive
        //first in order to avoid deadlock - achieve synchronization
        MPI_Request reqsend, reqrecv;

        int src = id-1;
        if(id == 0)
        {
            src = world_size-1;
        }
        int dst = (id+1)%world_size;

        for(int i = 0 ; i<n*d ; i ++)
        {
            corpus_send[i] = corpus[i];
        }
        if(step!=world_size-1)
        {
            MPI_Isend(corpus_send, n*d, MPI_DOUBLE, dst, 1, MPI_COMM_WORLD, &reqsend);
            MPI_Irecv(corpus_buff, n*d, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, &reqrecv);
        }

        //define indices[], considering rank and step
        for(int i = 0 ; i < n ; i++)
        {
            for(int j = 0 ; j < n ; j++)
            {

                indices[j+i*n] = (id-1)*n+j-step*n;

                //modular arithmetic is applied to indices[]
                if(indices[j+i*n]<0)
                {
                    indices[j+i*n]=((id-step-1)%world_size+world_size)*n+j;

                }
                if(indices[j+i*n]>=world_size*n)
                {
                    indices[j+i*n]=((id-step-1)%world_size)*n+j;
                }
            }
        }

        D = distances(corpus, query, n, n, d);

        //find the k nearest elements
        for(int i=0;i<n;i++){
          for(int j=k;j>0;j--){
            if(j==k){
              quickSelect(D,i*n,i*n+n,j,indices);
            }else{
              quickSelect(D,i*n,i*n+j,j,indices);
            }
          }
        }

        //compare local knnResult to global knnResult
        //update knnresult with every newly receives corpus
        double *temp_ndist = malloc(k*n*sizeof(double));
        int *temp_idx = malloc(k*n*sizeof(int));
        if(step==0)
        {

            for(int i=0; i<k ; i++)
            {
                for(int j=0 ; j<n ; j++)
                {

                    knn_ring.ndist[i*n+j]=D[j*n+i];
                    knn_ring.nidx[i*n+j]=indices[j*n+i];
                }
            }
        }
        else
        {
            //for every query point
            for(int i = 0 ; i < n ; i++)
            {
                int l=0, m=0, index=0;
                //for all point's neighbors
                while(index<k)
                {
                    if(knn_ring.ndist[l*n+i]<D[i*n+m])
                    {
                        temp_ndist[index*n+i]=knn_ring.ndist[l*n+i];
                        temp_idx[index*n+i] = knn_ring.nidx[l*n+i];

                        index++;
                        l++;
                    }
                    else
                    {

                        temp_ndist[index*n+i]=D[i*n+m];
                        temp_idx[index*n+i]=indices[i*n+m];

                        index++;
                        m++;
                    }

                }


            }

            knn_ring.ndist = temp_ndist;
            knn_ring.nidx = temp_idx;
        }

        if(step!=world_size-1)
        {
            //update corpus asynchronously
            MPI_Wait(&reqrecv,&status);
            MPI_Wait(&reqsend, &status);
            for(int i=0 ; i<n*d ; i++)
            {
                corpus[i] = corpus_buff[i];
            }
        }

        //Global Reductions all to all
        double local_min = DBL_MAX, local_max=-DBL_MAX;
        for(int i=0 ; i<knn_ring.m*k; i++)
        {
            if(knn_ring.ndist[i] != 0 && knn_ring.ndist[i] < local_min)
            {
                local_min = knn_ring.ndist[i];
            }
            if(knn_ring.ndist[i] > local_max)
            {
                local_max = knn_ring.ndist[i];
            }
        }
        double global_min, global_max;
        MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        
    }





    free(D);
    free(corpus);
    free(indices);


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

    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++)
        {
            Z[i+j*m] *= 2;
        }
    }

    //Calculate sumRow(X^2), sumRow(Y^2) terms
    Xrows = sumRowPow(X, n, d);
    Yrows = sumRowPow(Y, m, d);

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
        }
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
}
