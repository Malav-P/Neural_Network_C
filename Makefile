all: nn_lib bin/mnist.exe

nn_lib : $(patsubst neuralnet/src/%.c, lib/%.o, $(wildcard neuralnet/src/*.c))
	ar rvs lib/libNEURALNET.a $^

lib/%.o : neuralnet/src/%.c
	gcc -O3 -g -c -pedantic -I neuralnet/include/  $< -o $@

#bin/main: src/mnist.c src/nn_functions.c
# 	gcc -o nn_functions.o -c nn_functions.c
	#gcc -O3 src/mnist.c src/nn_functions.c -o bin/main.exe -I include/ -I src/.

bin/%.exe : src/%.c 
	gcc -O3 -g -pedantic -I neuralnet/include/ -I lib/ $< -o $@ lib/libNEURALNET.a


clean:
	rm -f bin/*.exe
	rm -f weights_biases/*.txt
	rm -f lib/*
	