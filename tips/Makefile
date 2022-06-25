# A Makefile for the pure-C part of the book

CC=clang
CFLAGS=-I. -O0

hilbert.o : hilbert.c hilbert.h
	$(CC) $(CFLAGS) -c hilbert.c

hilbert_test.o : hilbert_test.c hilbert.h minunit.h
	$(CC) $(CFLAGS) -c hilbert_test.c

hilbert_loop.o : hilbert_loop.c hilbert.h 
	$(CC) $(CFLAGS) -c hilbert_loop.c

hilbert_test : hilbert.o hilbert_test.o
	$(CC) -o hilbert_test hilbert.o hilbert_test.o

hilbert_loop : hilbert.o hilbert_loop.o
	$(CC) -o hilbert_loop hilbert.o hilbert_loop.o

test: hilbert_test
	hilbert_test

loop: hilbert_loop
	hilbert_loop

clean :
	rm -f *.o hilbert_test hilbert_loop
