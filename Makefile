main: example_mnist_dataset.c nn_functions.c optimizers.c activations.c alloc_dealloc.c
	gcc -O3 example_mnist_dataset.c nn_functions.c optimizers.c activations.c alloc_dealloc.c -o main -I.
	