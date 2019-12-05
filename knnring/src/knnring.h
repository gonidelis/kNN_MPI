#ifndef KNNRING_H
#define KNNRING_H


// Definition of the kNN result struct
typedef struct knnresult
{
    int *nidx;      //indices of nearest neighbors
    double *ndist;  //distance of nearest neighbors
    int m;          //number of query points
    int k;          //number of nearest neighbors
}knnresult;

knnresult kNN(double *X, double *Y, int n, int m, int d, int k);
knnresult distrAllkNN(double* X, int n,int d,int k);
void print_col_maj_array(double *array,int n,int d,  int id);
void merge(int *arr1, int len1, int *arr2, int len2, int *arr, int k);
double *distances(double *X, double *Y, int n, int m, int d);
double *sumRowPow(double *X, int n , int d);
double partition(double a[], int left, int right, int pivotIndex, int* indices);
double quickselect(double A[], int left, int right, int k, int *indices);

#endif /* HELLO_WORLD_H */
