# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = gcc-7
MPICC = mpicc
MPIRUN = mpirun -np 4

test_sequential:
	#tar -xvzf code.tar.gz
	cd knnring; make lib; cd ..
	cd knnring; cp lib/*.a inc/knnring.h ../; cd ..
	$(CC) tester.c knnring_sequential.a -o $@ -lm -lopenblas
	./test_sequential


test_synchronous:
	#tar -xvzf code.tar.gz
	cd knnring; make lib; cd ..
	cd knnring; cp lib/*.a inc/knnring.h ../; cd ..
	$(MPICC) tester_synchronous.c knnring_synchronous.a -o $@ -lm -lopenblas
	$(MPIRUN) ./test_synchronous


test_asynchronous:
	#tar -xvzf code.tar.gz
	cd knnring; make lib; cd ..
	cd knnring; cp lib/*.a inc/knnring.h ../; cd ..
	$(MPICC) tester_asynchronous.c knnring_asynchronous.a -o $@ -lm -lopenblas
	$(MPIRUN) ./test_asynchronous
