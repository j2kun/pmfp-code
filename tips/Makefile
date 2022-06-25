# A Makefile for the pure-C part of the book

CC=clang
CFLAGS=-I.

hilbert.o : hilbert.c hilbert.h
	$(CC) $(CFLAGS) -c hilbert.c

hilbert_test.o : hilbert_test.c hilbert.h minunit.h
	$(CC) $(CFLAGS) -c hilbert_test.c

hilbert_test : hilbert.o hilbert_test.o
	$(CC) -o hilbert_test hilbert.o hilbert_test.o

test: hilbert_test
	hilbert_test

clean :
	rm -f hilbert.o hilbert_test.o hilbert_test
