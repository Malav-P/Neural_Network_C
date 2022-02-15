bin/main: src/mnist.c src/nn_functions.c
# 	gcc -o nn_functions.o -c nn_functions.c
	gcc -O3 src/mnist.c src/nn_functions.c -o bin/main.exe -I include/ -I src/.

clean:
	rm -f bin/*.exe
	rm -f weights_biases/*.txt
	