#ifndef QUICKSELECT_H
#define QUICKSELECT_H

void swap(double *a, double *b)
{
    double temp = *a;
    *a = *b;
    *b = temp;
}
void iswap(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(double *A,int left, int right)
{
    int i = left, x;
    double pivot = A[right];
 
    for (x = left; x < right; x++){
        if (A[x] <= pivot){
            swap(&A[i], &A[x]);
            i++;
        }
    }
 
    swap(&A[i], &A[right]);
    return i;
}

int sortPartition(double *A,int *index,int low, int high)
{
    int i = low-1;
    double pivot = A[high];
 
    for (int j = low; j <= high-1; j++){
        if (A[j] < pivot){
            i++;
            swap(&A[i], &A[j]);
            iswap(&index[i], &index[j]);
        }
    }
 
    swap(&A[i+1], &A[high]);
    iswap(&index[i+1], &index[high]);
    
    return (i+1);
}

void quicksort(double *A, int *index, int low, int high)
{
    if(low<high)
    {
        int pi = sortPartition(A,index,low,high);

        quicksort(A,index,low,pi-1);
        quicksort(A,index,pi+1,high);
    }
}

double quickselect(double *A,int left, int right, int k)
{
 
    //p is position of pivot in the partitioned array
    int p = partition(A, left, right);
 
    //k equals pivot got lucky
    if (p == k-1){
        return A[p];
    }
    //k less than pivot
    else if (k - 1 < p){
        return quickselect(A, left, p - 1, k);
    }
    //k greater than pivot
    else{
        return quickselect(A, p + 1, right, k);
    }
}

double ksmallest(double *A, int n, int k)
{
 
    int left = 0; 
    int right = n - 1; 
 
    return quickselect(A, left, right, k);
}

#endif