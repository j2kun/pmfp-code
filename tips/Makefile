# A Makefile for the pure-C part of the book

CC=clang
CFLAGS=-I. -O0

hilbert.o : hilbert.c hilbert.h
	$(CC) $(CFLAGS) -c $^

hilbert_test.o : hilbert_test.c hilbert.h minunit.h
	$(CC) $(CFLAGS) -c $^

hilbert_loop_naive.o : hilbert_loop_naive.c hilbert.h 
	$(CC) $(CFLAGS) -c $^

hilbert_loop.o : hilbert_loop.c hilbert.h 
	$(CC) $(CFLAGS) -c $^

hilbert_loop_2.o : hilbert_loop_2.c hilbert.h hilbert_loop_2.h
	$(CC) $(CFLAGS) -c $^

hilbert_test : hilbert.o hilbert_test.o
	$(CC) -o $@ $^

hilbert_loop : hilbert.o hilbert_loop.o
	$(CC) -o $@ $^

hilbert_loop_2 : hilbert.o hilbert_loop_2.o
	$(CC) -o $@ $^

hilbert_loop_naive : hilbert.o hilbert_loop_naive.o
	$(CC) -o $@ $^

test: hilbert_test
	hilbert_test
	
binaries: hilbert_loop hilbert_loop_naive hilbert_loop_2
	touch $^

loop: hilbert_loop hilbert_loop_naive
	hilbert_loop && hilbert_loop_naive

clean :
	rm -f *.o *.gch *.out* hilbert_test hilbert_loop hilbert_loop_naive hilbert_loop_2
