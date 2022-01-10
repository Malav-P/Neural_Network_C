main: mnist.c nn_functions.c
	gcc -O3 mnist.c nn_functions.c -o main -I.
	