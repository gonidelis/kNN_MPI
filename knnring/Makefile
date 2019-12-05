# define the C/C++ compiler to use,default here is clang
SHELL := /bin/bash
CC = gcc-7
MPICC = mpicc

.PHONY: lib

knnring_sequential:
	cd src; $(CC) main.c knnring_sequential.c -lm -lcblas -o knnring_sequential;
	./knnring_sequential

lib:
	mkdir -p lib;
	cd src; $(CC) -c knnring_sequential.c -lm -lcblas; cd ..
	cd src; ar rcs ../lib/knnring_sequential.a knnring_sequential.o; cd ..

	cd src; $(MPICC) -c  knnring_synchronous.c -lm -lcblas; cd ..
	cd src; ar rcs ../lib/knnring_synchronous.a knnring_synchronous.o; cd ..

	cd src; $(MPICC) -c  knnring_asynchronous.c -lm -lcblas; cd ..
	cd src; ar rcs ../lib/knnring_asynchronous.a knnring_asynchronous.o; cd ..
