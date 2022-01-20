main: mnist.c nn_functions.c
# 	gcc -o nn_functions.o -c nn_functions.c
	gcc -O3 mnist.c nn_functions.c -o main -I.

clean:
	rm *.exe
	rm *.txt
	