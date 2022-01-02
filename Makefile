main: example_mnist_dataset.c nn_functions.c
	gcc -O3 example_mnist_dataset.c nn_functions.c -o main -I.
	