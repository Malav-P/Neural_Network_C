#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include "structs_and_globals.h"
#include "activations.h"
#include "alloc_dealloc.h"
#include "optimizers.h"


//********************************************************************************************

// assign a pointer to an int and a pointer to the first element of an array of int that describe the number of layers and the number of neurons in each layer of the network.
void network_params(int* n, int* s){

	network = n;  // assign the global pointer network to the given pointer n.
	size = s;     // assign the global pointer size to the given pointer s.
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// feed the activations in the ith layer forward and compute the resulting activations in the (i+1)th layer using the function (*f).
void feed_fwd_H(int i, struct NEURAL_NET* my_net, void (*f)(double*, int)){
	int j, k;
	double* curr_L = my_net->activations_N[i];
	double* next_L = my_net->activations_N[i+1];
	double** wgt_matr_w = my_net->weights_W[i];

	for (j=0;j<network[i+1];j++){
		double answer=0;

		for (k=0;k<network[i];k++){
			answer += curr_L[k]*wgt_matr_w[k][j];
		}

		next_L[j] = answer + my_net->biases_N[i+1][j];
	}

	(*f)(next_L, network[i+1]);
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// print the output activations of the neural network
void print_output(struct NEURAL_NET* my_net){
	int i;

	for (i=0;i<network[*size-1];i++){
		printf("%lf, ", my_net->activations_N[*size-1][i]);
	}
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// initialize the network with random weights and biases.
struct NEURAL_NET* initialize_network(int* n, int* s){
	network_params(n, s);
	struct NEURAL_NET* net = malloc(sizeof*net);

	net->activations_N = alloc_N();   // net->activations_N[i][j] is the activation of the jth neuron in the ith layer
	net->biases_N = he_init_N();       // net->biases_N[i][j] is the bias of the jth neuron in the ith layer
	net->weights_W = he_init_W();      // net->weights_W[i][j][k] is the weight connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
	net->gradients_W = r_init_W();     // net->gradients_W[i][j][k] is the computed gradient of the cost function wrt net->weights_W[i][j][k] after the last training iteration of the network. It is initialized to random values to avoid premature training termination.

	return net;
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// enter input data into first layer of network. Feed these activations forward throughout the rest of the network.
void feed_fwd(double input[], struct NEURAL_NET* my_net){
	int i;
	double* first_L = my_net->activations_N[0];

	for (i=0;i<network[0];i++){
		first_L[i] = input[i];
	}

	for (i=0;i<*size-2;i++){
		feed_fwd_H(i, my_net, &activ);
	}

	feed_fwd_H(*size-2, my_net, &out_activ);
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// check if the gradients in the network are all close to zero (below a set precision value). Return true if this is so. Return false otherwise.
bool min_reached(struct NEURAL_NET* my_net){
	int i,j,k;
	double precision = 0.00000001;

	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			    if (my_net->gradients_W[i][j][k] > precision){
			    	return false;
			    }
			}
	    }        
	}
	return true;
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// train the network
void train_network(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int no_of_inputs, int batch_size, int epochs, int optimizer){
	int i,j;

	int iterations = ceil(no_of_inputs/batch_size);
	int remaining_samples = no_of_inputs%batch_size;

	if (remaining_samples == 0){

		for (j=0;j<epochs;j++){
			
			// mini batch SGD
			if (optimizer == 1){

				for (i=0;i<iterations;i++){
					// int x, y, z;
					// for (x=0;x<*size-1;x++){

					// 	for (y=0;y<network[x];y++){

					// 		for (z=0;z<network[x+1];z++){
					// 		    printf("%lf, ",my_net->gradients_W[x][y][z]);
					// 		}
					//     }        
					// }

					train_batch(my_net, inputs, outputs, batch_size*i, batch_size);

					if (min_reached(my_net)){
						printf("mini batch descent algorithm has reached sufficient precision after %d iterations on epoch number %d \n", i+1, j+1);
						return;
					}
				}
			}

			// momentum
			else if (optimizer == 2){
				double*** v_W = init_W();
				double** v_N = init_N();

				for (i=0;i<iterations;i++){
					// int x, y, z;
					// for (x=0;x<*size-1;x++){

					// 	for (y=0;y<network[x];y++){

					// 		for (z=0;z<network[x+1];z++){
					// 		    printf("%lf, ",my_net->gradients_W[x][y][z]);
					// 		}
					//     }        
					// }
					train_batch_momentum(my_net, inputs, outputs, batch_size*i, batch_size, v_W, v_N);

					if (min_reached(my_net)){
						dealloc_W(v_W);
						dealloc_N(v_N);
						printf("momentum algorithm has reached sufficient precision after %d iterations on epoch number %d \n", i+1, j+1);
						return;
					}
				}

				dealloc_W(v_W);
				dealloc_N(v_N);
			}

			// adagrad
			else if (optimizer == 3){
				double*** a_W = init_W();
				double** a_N = init_N();

				for (i=0;i<iterations;i++){
					// int x, y, z;
					// for (x=0;x<*size-1;x++){

					// 	for (y=0;y<network[x];y++){

					// 		for (z=0;z<network[x+1];z++){
					// 		    printf("%lf, ",my_net->gradients_W[x][y][z]);
					// 		}
					//     }        
					// }
					train_batch_adagrad(my_net, inputs, outputs, batch_size*i, batch_size, a_W, a_N);

					if (min_reached(my_net)){
						dealloc_W(a_W);
						dealloc_N(a_N);
						printf("adagrad algorithm has reached sufficient precision after %d iterations on epoch number %d \n", i+1, j+1);
						return;
					}
				}

				dealloc_W(a_W);
				dealloc_N(a_N);
			}

			// RMSprop
			else if (optimizer == 4){
				double*** s_W = init_W();
				double** s_N = init_N();

	
				for (i=0;i<iterations;i++){
					// int x, y, z;
					// for (x=0;x<*size-1;x++){

					// 	for (y=0;y<network[x];y++){

					// 		for (z=0;z<network[x+1];z++){
					// 		    printf("%lf, ",my_net->gradients_W[x][y][z]);
					// 		}
					//     }        
					// }
					train_batch_RMSprop(my_net, inputs, outputs, batch_size*i, batch_size, s_W, s_N);

					if (min_reached(my_net)){
						dealloc_W(s_W);
						dealloc_N(s_N);
						printf("RMSprop algorithm has reached sufficient precision after %d iterations on epoch number %d \n", i+1, j+1);
						return;
					}
				}

				dealloc_W(s_W);
				dealloc_N(s_N);
			}

			// adadelta
			else if (optimizer == 5){
				double*** s_W = init_W();
				double** s_N = init_N();

				double*** D_W = init_W();
				double** D_N = init_N();

				double*** deltas_W = init_W();
				double** deltas_N = init_N();

	
				for (i=0;i<iterations;i++){
					// int x, y, z;
					// for (x=0;x<*size-1;x++){

					// 	for (y=0;y<network[x];y++){

					// 		for (z=0;z<network[x+1];z++){
					// 		    printf("%lf, ",my_net->gradients_W[x][y][z]);
					// 		}
					//     }        
					// }
					train_batch_adadelta(my_net, inputs, outputs, batch_size*i, batch_size, s_W, s_N, deltas_W, deltas_N, D_W, D_N);

					if (min_reached(my_net)){
						dealloc_W(s_W);
						dealloc_N(s_N);
						dealloc_W(D_W);
						dealloc_N(D_N);
						dealloc_W(deltas_W);
						dealloc_N(deltas_N);
						printf("adadelta algorithm has reached sufficient precision after %d iterations on epoch number %d \n", i+1, j+1);
						return;
					}
				}

				dealloc_W(s_W);
				dealloc_N(s_N);

				dealloc_W(D_W);
				dealloc_N(D_N);

				dealloc_W(deltas_W);
				dealloc_N(deltas_N);
			}

			// adam
			else if (optimizer == 6){
				double*** m_W = init_W();
				double** m_N = init_N();

				double*** mhat_W = alloc_W();
				double** mhat_N = alloc_N();

				double*** v_W = init_W();
				double** v_N = init_N();

				double*** vhat_W = alloc_W();
				double** vhat_N = alloc_N();

	
				for (i=0;i<iterations;i++){
					train_batch_adam(my_net, inputs, outputs, batch_size*i, batch_size, m_W, mhat_W, m_N, mhat_N, v_W, vhat_W, v_N, vhat_N, i+1);
				    // int x, y, z;
				    // for (x=0;x<*size-1;x++){

				    //  for (y=0;y<network[x];y++){

				    //      for (z=0;z<network[x+1];z++){
				    //          printf("%lf, ",my_net->gradients_W[x][y][z]);
				    //      }
				    //      printf("ENDLINE\n");
				    //     }
				    //     printf("ENDMATRIX\n\n\n\n");
				    // }
					if (min_reached(my_net)){
						dealloc_W(m_W);
						dealloc_N(m_N);
						dealloc_W(mhat_W);
						dealloc_N(mhat_N);
						dealloc_W(v_W);
						dealloc_N(v_N);
						dealloc_W(vhat_W);
						dealloc_N(vhat_N);
						printf("adam algorithm has reached sufficient precision after %d iterations on epoch number %d \n", i+1, j+1);
						return;
					}
				}

				dealloc_W(m_W);
				dealloc_N(m_N);

				dealloc_W(mhat_W);
				dealloc_N(mhat_N);

				dealloc_W(v_W);
				dealloc_N(v_N);

				dealloc_W(vhat_W);
				dealloc_N(vhat_N);
			}
		}

		printf("algorithm has completed training of %d epochs\n", epochs);
	}
		
	else{

		for (j=0;j<epochs;j++){

			// mini batch SGD
			if (optimizer == 1){

				for (i=0;i<iterations-1;i++){
					train_batch(my_net, inputs, outputs, batch_size*i, batch_size);

					if (min_reached(my_net)){
						printf("minibatch algorithm has reached sufficient precision after %d iterations on epoch number %d \n", i+1, j+1);
						return;
					}
				}
				train_batch(my_net, inputs, outputs, batch_size*(iterations-1), remaining_samples);
			}

			// momentum
			else if (optimizer == 2){
				double*** v_W = init_W();
				double** v_N = init_N();

				for (i=0;i<iterations-1;i++){
					train_batch_momentum(my_net, inputs, outputs, batch_size*i, batch_size, v_W, v_N);

					if (min_reached(my_net)){
						dealloc_N(v_N);
						dealloc_W(v_W);
						printf("momentum algorithm has reached sufficient precision after %d iterations on epoch number %d \n", i+1, j+1);
						return;
					}
				}
				train_batch_momentum(my_net, inputs, outputs, batch_size*(iterations-1), remaining_samples, v_W, v_N);

				dealloc_W(v_W);
				dealloc_N(v_N);
			}

			// adagrad
			else if (optimizer == 3){
				double*** a_W = init_W();
				double** a_N = init_N();

				for (i=0;i<iterations-1;i++){
					train_batch_adagrad(my_net, inputs, outputs, batch_size*i, batch_size, a_W, a_N);

					if (min_reached(my_net)){
						dealloc_W(a_W);
						dealloc_N(a_N);	
						printf("adagrad algorithm has reached sufficient precision after %d iterations on epoch number %d \n", i+1, j+1);
						return;
					}
				}
				train_batch_adagrad(my_net, inputs, outputs, batch_size*(iterations-1), remaining_samples, a_W, a_N);

				dealloc_W(a_W);
				dealloc_N(a_N);						
			}

			// RMSprop
			else if (optimizer == 4){
				double*** s_W = init_W();
				double** s_N = init_N();

				for (i=0;i<iterations-1;i++){
					train_batch_RMSprop(my_net, inputs, outputs, batch_size*i, batch_size, s_W, s_N);

					if (min_reached(my_net)){
						dealloc_W(s_W);
						dealloc_N(s_N);	
						printf("RMSprop algorithm has reached sufficient precision after %d iterations on epoch number %d \n", i+1, j+1);
						return;
					}
				}
				train_batch_RMSprop(my_net, inputs, outputs, batch_size*(iterations-1), remaining_samples, s_W, s_N);

				dealloc_W(s_W);
				dealloc_N(s_N);	
			}

			// adadelta
			else if (optimizer == 5){
				double*** s_W = init_W();
				double** s_N = init_N();

				double*** D_W = init_W();
				double** D_N = init_N();

				double*** deltas_W = init_W();
				double** deltas_N = init_N();

	
				for (i=0;i<iterations-1;i++){
					// int x, y, z;
					// for (x=0;x<*size-1;x++){

					// 	for (y=0;y<network[x];y++){

					// 		for (z=0;z<network[x+1];z++){
					// 		    printf("%lf, ",my_net->gradients_W[x][y][z]);
					// 		}
					//     }        
					// }
					train_batch_adadelta(my_net, inputs, outputs, batch_size*i, batch_size, s_W, s_N, deltas_W, deltas_N, D_W, D_N);

					if (min_reached(my_net)){
						dealloc_W(s_W);
						dealloc_N(s_N);
						dealloc_W(D_W);
						dealloc_N(D_N);
						dealloc_W(deltas_W);
						dealloc_N(deltas_N);
						printf("adadelta algorithm has reached sufficient precision after %d iterations on epoch number %d \n", i+1, j+1);
						return;
					}
				}

				train_batch_adadelta(my_net, inputs, outputs, batch_size*(iterations-1), remaining_samples, s_W, s_N, deltas_W, deltas_N, D_W, D_N);

				dealloc_W(s_W);
				dealloc_N(s_N);

				dealloc_W(D_W);
				dealloc_N(D_N);

				dealloc_W(deltas_W);
				dealloc_N(deltas_N);
			}

			// adam
			else if (optimizer == 6){
				double*** m_W = init_W();
				double** m_N = init_N();

				double*** mhat_W = alloc_W();
				double** mhat_N = alloc_N();

				double*** v_W = init_W();
				double** v_N = init_N();

				double*** vhat_W = alloc_W();
				double** vhat_N = alloc_N();

	
				for (i=0;i<iterations-1;i++){
				    // int x, y, z;
				    // for (x=0;x<s-1;x++){

				    //  for (y=0;y<my_n[x];y++){

				    //      for (z=0;z<my_n[x+1];z++){
				    //          printf("%lf, ",my_net->gradients_W[x][y][z]);
				    //      }
				    //      printf("ENDLINE\n");
				    //     }
				    //     printf("ENDMATRIX\n\n\n\n");
				    // }
					train_batch_adam(my_net, inputs, outputs, batch_size*i, batch_size, m_W, mhat_W, m_N, mhat_N, v_W, vhat_W, v_N, vhat_N, i+1);

					if (min_reached(my_net)){
						dealloc_W(m_W);
						dealloc_N(m_N);
						dealloc_W(mhat_W);
						dealloc_N(mhat_N);
						dealloc_W(v_W);
						dealloc_N(v_N);
						dealloc_W(vhat_W);
						dealloc_N(vhat_N);
						printf("adam algorithm has reached sufficient precision after %d iterations on epoch number %d \n", i+1, j+1);
						return;
					}
				}
				
				train_batch_adam(my_net, inputs, outputs, batch_size*(iterations-1), remaining_samples, m_W, mhat_W, m_N, mhat_N, v_W, vhat_W, v_N, vhat_N, iterations);


				dealloc_W(m_W);
				dealloc_N(m_N);

				dealloc_W(mhat_W);
				dealloc_N(mhat_N);

				dealloc_W(v_W);
				dealloc_N(v_N);

				dealloc_W(vhat_W);
				dealloc_N(vhat_N);
			}
		}
		printf("algorithm has completed training of %d epochs\n", epochs);
	}
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// dealloc memory associated with network
void free_network(struct NEURAL_NET* my_net){
	dealloc_N(my_net->activations_N);
	dealloc_N(my_net->biases_N);
	dealloc_W(my_net->weights_W);
	free(my_net);
}

//---------------------------------------------------------------------------------------------
