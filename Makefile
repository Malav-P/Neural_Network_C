main: mnist_training.c nn_functions.c
	gcc  -O3 mnist_training.c nn_functions.c -o main -I.
	