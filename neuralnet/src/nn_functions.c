#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nn_functions.h"


// declare global pointers int* network and int* size
// *size is an int. Corresponds to number of layers in neural net (a.k.a the length of the array that network points to)
// network points to first element in an arry of int. Each entry corresponds to number of neurons in the neural net layer.
int* network;
int* size;

// declare a struct for the gradients to be computed in back_propagation
struct GRADIENTS{
	double*** W;  // an order three pointer. W[i][j][k] corresponds to the partial derivative of the cost function wrt weights[i][j][k] (defined in the struct in the header^).
	double** N;   // and order two pointer. N[i][j] corresponds to the partical derivative of the cost function wrt to biases[i][j] (defined in the struct in the header^).
};






//
// BEGIN FUNCTIONS OPERATING ON VARIABLES IN THE HEAP
//





// allocate memory in a shape of type N. N is usually a network of neurons.
// ptr_N[i][j] is the jth neuron in the ith layer
static double** alloc_N(){

   double** ptr_N = (double**)calloc(*size, sizeof(double*));

   for (int i=0;i<*size;i++){
      ptr_N[i] = (double*)calloc(network[i], sizeof(double));
   }

   return ptr_N;
}


// deallocate memory of a structure of type N.
static void dealloc_N(double** ptr_N){

   for (int i=0;i<*size;i++){
      free(ptr_N[i]);
   }

   free(ptr_N);
}


// allocate memory for a structure of type w. w is a matrix  that can hold the weights connecting two structures of type L.
// ptr_w[j][k] is the entry connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
static double** alloc_w(int i){

   double** ptr_w = (double**)calloc(network[i], sizeof(double*));

   for (int j=0;j<network[i];j++){
      ptr_w[j] = (double*)calloc(network[i+1], sizeof(double));
   }

   return ptr_w;
}


// deallocate memory that was previously allocated using alloc_w.
static void dealloc_w(double** ptr_w, int i){

   for (int j=0;j<network[i];j++){
      free(ptr_w[j]);
   }

   free(ptr_w);
}


// allocate memory for a structure of type W. W can be thought of as a list of structure type w.
// ptr_W[i][j][k] is the entry connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
static double*** alloc_W(){

   double*** ptr_W = (double***)calloc(*size-1, sizeof(double**));

   for (int i=0;i<*size-1;i++){
      ptr_W[i] = alloc_w(i);
   }

   return ptr_W;
}


// deallocate memory for a previously allocated structure of type W.
static void dealloc_W(double*** ptr_W){

   for (int i=0;i<*size-1;i++){
      dealloc_w(ptr_W[i], i);
   }

   free(ptr_W);
}


// dealloc memory associated with network
void free_network(NETWORK* my_net){
	dealloc_N(my_net->activations_N);
	dealloc_N(my_net->biases_N);
	dealloc_W(my_net->weights_W);
   dealloc_W(my_net->gradients_W);
	free(my_net);
}





//
// END FUNCTIONS OPERATING ON VARIABLES IN THE HEAP
//




//
// BEGIN FUNCTIONS TO INITIALIZE NETWORK
//




// generate a random value from a uniform distribution between 0 and 1
static double rand_gen() {
   // return a uniformly distributed random value
   return ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
}



// generate a random value from the standard normal distribution
static double normalRandom() {
   // return a normally distributed random value
   double v1=rand_gen();
   double v2=rand_gen();
   return cos(2*3.14*v2)*sqrt(-2.*log(v1));
}


// returns a random number from the range (min, max)
static double randfrom(double min, double max){
    double range = (max - min); 
    double div = RAND_MAX / range;

    return min + (rand() / div);
}


// allocate memory for a structure of type N. Initialize all entries
// ptr_N[i][j] is the entry for the jth neuron in the ith layer.
static double** init_N(char c){
   double** ptr_N = alloc_N();

   switch(c){
   	// initialize with random values
	   case 'r':{
		   for (int i=0;i<*size;i++){

		      for (int j=0;j<network[i];j++){
		         ptr_N[i][j] = randfrom(-0.01, 0.01);
		      }
		   }
		   break;
	   }

	   // use He intialization
	   case 'h':{
		   for (int i=0;i<*size;i++){

		      for (int j=0;j<network[i];j++){
		         ptr_N[i][j] = normalRandom()*sqrt(2.0/network[i]);
		      }
		   }
		   break;
	   }
	   // initialize all values to zero
	   case 'z':{
		   for (int i=0;i<*size;i++){

		      for (int j=0;j<network[i];j++){
		         ptr_N[i][j] = 0;
		      }
		   }
		   break;
		}
	}
   return ptr_N;
}

// allocate memory for a structure of type W and initialize all entries
// ptr_W[i][j][k] is the weight connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
static double*** init_W(char c){
   double*** ptr_W = alloc_W();

   switch(c){
	   case 'r':{
		   for (int i=0;i<*size-1;i++){

		      for (int j=0;j<network[i];j++){

		         for (int k=0;k<network[i+1];k++){
		            ptr_W[i][j][k] = randfrom(-0.01,0.01);
		         }
		      }
		   }   	
		   break;
	   }

	   case 'h':{
		   for (int i=0;i<*size-1;i++){

		      for (int j=0;j<network[i];j++){

		         for (int k=0;k<network[i+1];k++){
		            ptr_W[i][j][k] = normalRandom()*sqrt(2.0/network[i]);
		         }
		      }
		   }
		   break;
	   }

	   case 'z':{
		   for (int i=0;i<*size-1;i++){

		      for (int j=0;j<network[i];j++){

		         for (int k=0;k<network[i+1];k++){
		            ptr_W[i][j][k] = 0;
		         }
		      }
		   }
		   break;
		}
	}

   return ptr_W;
}




// initialize the network with random weights and biases.
NETWORK* initialize_network(int* n, int* s){
	network = n;
	size = s;
	
	NETWORK* net = malloc(sizeof*net);

	net->activations_N = alloc_N();   // net->activations_N[i][j] is the activation of the jth neuron in the ith layer
	net->biases_N = init_N('h');       // net->biases_N[i][j] is the bias of the jth neuron in the ith layer
	net->weights_W = init_W('h');      // net->weights_W[i][j][k] is the weight connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
	net->gradients_W = init_W('r');     // net->gradients_W[i][j][k] is the computed gradient of the cost function wrt net->weights_W[i][j][k] after the last training iteration of the network. It is initialized to random values to avoid premature training termination.

	return net;
}





//
// END FUNCTIONS TO INITIALIZE NETWORK
//



//
// BEGIN FUNCTIONS FOR HIDDEN AND OUTPUT LAYER ACTIVATIONS
//




// applies activation function to a structure of type L. I have chosen the relU function
static void activ(double* lyr_L, int len){

	for (int i=0;i<len;i++){

		if (lyr_L[i]<0){
			lyr_L[i] = 0.0;
		}

	}
}


// derivative of activation function for hidden layer. I have chosen the derivative of the relU function
static double activ_deriv(double x){

	if (x<0){
		return 0.0;
	}

	else{
		return 1.0;
	}
}


// applies activation function to output layer in neural network (which has structure type L). I have chosen the softmax activation function.
static void out_activ(double* lyr_L, int len){
	double sum = 0;
	int i;

	for (i=0;i<len;i++){
		sum += exp(lyr_L[i]);
	}

	for (i=0; i<len;i++){
		lyr_L[i] = (1/sum)*exp(lyr_L[i]);
	}
}


// given a layer lyr_L of structure type L and and index i, this function returns the derivative of the softmax function wrt to the variable in the (index)th position of lyr_L
static double out_activ_deriv(double* lyr_L, int len, int index){

	double sum = 0;
	double x = lyr_L[index];

	for (int i=0;i<len;i++){
		sum += exp(lyr_L[i]);
	}

	return (1/sum)*exp(x)*(1-(1/sum)*exp(x));
}






//
// END FUNCTIONS FOR HIDDEN AND OUTPUT LAYER ACTIVATIONS
//



//
// BEGIN FUNCTIONS FOR BACK PROPAGATION
//




// feed the activations in the ith layer forward and compute the resulting activations in the (i+1)th layer using the function (*f).
static void feed_fwd_H(int i, NETWORK* my_net, void (*f)(double*, int)){
	double* curr_L = my_net->activations_N[i];
	double* next_L = my_net->activations_N[i+1];
	double** wgt_matr_w = my_net->weights_W[i];

	for (int j=0;j<network[i+1];j++){
		double answer=0;

		for (int k=0;k<network[i];k++){
			answer += curr_L[k]*wgt_matr_w[k][j];
		}

		next_L[j] = answer + my_net->biases_N[i+1][j];
	}

	(*f)(next_L, network[i+1]);
}


// enter input data into first layer of network. Feed these activations forward throughout the rest of the network.
void feed_fwd(double input[], NETWORK* my_net){
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


// update a structure of type W by adding another structure of type W to it.
static void add_to_W(double*** cum_sum_W, double*** matr_W){
    
    for (int i=0;i<*size-1;i++){

    	for (int j=0;j<network[i];j++){

    		for (int k=0;k<network[i+1];k++){
    		    cum_sum_W[i][j][k] += matr_W[i][j][k];
    		}
    	}
    }
}


// update a structure of type N by adding another structure of type N to it.
static void add_to_N(double** cum_sum_N, double** matr_N){

	for (int i=0;i<*size;i++){

		for (int j=0;j<network[i];j++){
			cum_sum_N[i][j] += matr_N[i][j];
		}
	}
}


// same thing as feed_fwd_H but keeps track of intermediate values for layers.
static void comp_grad_H_H(int i, NETWORK* my_net, void (*f)(double*, int), double** itrmd_N){
	double* curr_L = my_net->activations_N[i];
	double* next_L = my_net->activations_N[i+1];
	double** wgt_matr_w = my_net->weights_W[i];

	for (int j=0;j<network[i+1];j++){
		double answer=0;

		for (int k=0;k<network[i];k++){
			answer += curr_L[k]*wgt_matr_w[k][j];
		}

		itrmd_N[i+1][j] = answer + my_net->biases_N[i+1][j];
		next_L[j] = answer + my_net->biases_N[i+1][j];
	}

	(*f)(next_L, network[i+1]);
}


// same thing as feed_fwd but keeps track of intermediate values for layers.
static void comp_grad_H(double input[], NETWORK* my_net, double** itrmd_N){
	int i;

	for (i=0;i<network[0];i++){
		my_net->activations_N[0][i] = input[i];
	}

	for (i=0;i<*size-2;i++){
		comp_grad_H_H(i, my_net, &activ, itrmd_N);
	}

	comp_grad_H_H(*size-2, my_net, &out_activ, itrmd_N);
}


// compute the gradient of the cost function wrt the weights and wrt to the biases
static struct GRADIENTS* comp_grad(double input[], double output[], NETWORK* my_net){
	int i, j, k;

	struct GRADIENTS* grad = malloc(sizeof*grad);

	double*** DCDW_W = alloc_W();
	double**  DCDB_N = alloc_N();
	double**  itrmd_N = alloc_N();
	double**  err_N = alloc_N();

	comp_grad_H(input, my_net, itrmd_N);

	for (i=0;i<network[*size-1];i++){
		err_N[*size-1][i] = (my_net->activations_N[*size-1][i] - output[i])*out_activ_deriv(itrmd_N[*size-1], network[*size-1], i);
	}

	for (i=*size-2;i>0;i--){

		for (j=0;j<network[i];j++){
			double answer = 0;

			for (k=0;k<network[i+1];k++){
				answer += my_net->weights_W[i][j][k]*err_N[i+1][k]*activ_deriv(itrmd_N[i][j]);
			}

			err_N[i][j] = answer;
		}
	}

	for (i=1;i<*size;i++){

		for(j=0;j<network[i];j++){
			DCDB_N[i][j] = err_N[i][j];
		}
	}

	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
				DCDW_W[i][j][k] = my_net->activations_N[i][j]*err_N[i+1][k];
			}
		}
	}

	grad->W = DCDW_W;   // grad->dcdw[i][j][k] is the partial derivative of the cost function wrt weights[i][j][k].
	grad->N = DCDB_N;   // grad->dcdb[i][j] is the partial derivative of the cost function wrt biases[i][j].

	dealloc_N(itrmd_N);
	dealloc_N(err_N);

	return grad;
}


// compute the element-wise sum of all the gradients in a batch of training samples.
static struct GRADIENTS* sum_of_grads(NETWORK* my_net, double** inputs, double** outputs, int start, int batch_size){
	struct GRADIENTS* b_grad = malloc(sizeof*b_grad);

	int j, k;
    int in_size = network[0];
    int out_size = network[*size-1];

    double*** summation_W = init_W('z');
    double** summation_N = init_N('z');

    // compute cumulative sum of gradients for the batch
    for (j=start;j<start+batch_size;j++){
        double input[in_size];
        double output[out_size];

        for (k=0;k<in_size;k++){
        	input[k] = inputs[j][k];
        }

        for (k=0;k<out_size;k++){
        	output[k] = outputs[j][k];
        }

        struct GRADIENTS* g = comp_grad(input, output, my_net);

        add_to_W(summation_W, g->W);
        add_to_N(summation_N, g->N);

        dealloc_W(g->W);
        dealloc_N(g->N);

        free(g);
    }
    b_grad->W = summation_W;
    b_grad->N = summation_N;

    return b_grad;
}





//
// END FUNCTIONS FOR BACK PROPAGATION
//





//
// BEGIN FUNCTIONS FOR NETWORK TRAINING
//


// check if the gradients in the network are all close to zero (below a set precision value). Return 1 if this is so. Return 0 otherwise.
static int min_reached(NETWORK* my_net){

	double precision = 0.00000001;

	for (int i=0;i<*size-1;i++){

		for (int j=0;j<network[i];j++){

			for (int k=0;k<network[i+1];k++){
			    if (my_net->gradients_W[i][j][k] > precision){
			    	return 0;
			    }
			}
	    }        
	}
	return 1;
}




// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses mini batch SGD.
static void train_batch(NETWORK* my_net, double** inputs, double** outputs, int start, int batch_size){
    int i, j, k;

    struct GRADIENTS* b_grad = sum_of_grads(my_net, inputs, outputs, start, batch_size);


    // update gradients_W in the NEURAL_NET struct 
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			    my_net->gradients_W[i][j][k] = (1.0/batch_size)*b_grad->W[i][j][k];
			}
	    }        
	}

    double alpha = 0.1; // learning rate

    // apply mini batch SGD to weights
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			   my_net->weights_W[i][j][k] -= alpha*(1.0/batch_size)*b_grad->W[i][j][k];
			}
	    }               
	}

    dealloc_W(b_grad->W);

    // apply mini batch SGD to biases
    for (i=1;i<*size;i++){

    	for (j=0;j<network[i];j++){
    		    my_net->biases_N[i][j] -= alpha*(1.0/batch_size)*b_grad->N[i][j];
    	}
    }

    dealloc_N(b_grad->N);
    free(b_grad);
}


// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses mini batch SGD with momentum.
static void train_batch_momentum(NETWORK* my_net, double** inputs, double** outputs, int start, int batch_size, double*** v_W, double** v_N){
    int i, j, k;

    struct GRADIENTS* b_grad = sum_of_grads(my_net, inputs, outputs, start, batch_size);

    double beta = 0.9;   // exponential average hyperparameter
    double alpha = 0.1;  // learning rate


	// apply momentum to weights
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
				v_W[i][j][k] = beta*v_W[i][j][k] + (1.0-beta)*(1.0/batch_size)*b_grad->W[i][j][k];  // update the "velocity" of the gradient wrt the weights
			    my_net->weights_W[i][j][k] -= alpha*v_W[i][j][k];											 // update weights
			}
	    }               
	}

	dealloc_W(b_grad->W);

    // apply momentum to biases
    for (i=1;i<*size;i++){

    	for (j=0;j<network[i];j++){
    		v_N[i][j] = beta*v_N[i][j] + (1-beta)*(1.0/batch_size)*b_grad->N[i][j];	// compute the "velocity" of the gradients wrt the biases
    		my_net->biases_N[i][j] -= alpha*v_N[i][j];											// update biases
    	}
    }

    dealloc_N(b_grad->N);
    free(b_grad);
}


// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses Adagrad optimizer.
static void train_batch_adagrad(NETWORK* my_net, double** inputs, double** outputs, int start, int batch_size, double*** a_W, double** a_N){
    int i, j, k;

    struct GRADIENTS* b_grad = sum_of_grads(my_net, inputs, outputs, start, batch_size);

    // update gradients_W in the NEURAL_NET struct
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			    my_net->gradients_W[i][j][k] = (1.0/batch_size)*b_grad->W[i][j][k];
			}
	    }        
	}

    double alpha = 0.01;      // laerning rate 
    double eps = 0.0000001; // parameter to prevent divide by zero error.

    // apply adagrad to weights
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
				a_W[i][j][k] += pow((1.0/batch_size)*b_grad->W[i][j][k], 2);												// compute the accumulated squared gradients wrt the weights
			    my_net->weights_W[i][j][k] -= alpha*(1.0/sqrt(a_W[i][j][k] + eps))*(1.0/batch_size)*b_grad->W[i][j][k];	// update weights
			}
	    }               
	}

	dealloc_W(b_grad->W);

    // apply adagrad to biases
    for (i=1;i<*size;i++){

    	for (j=0;j<network[i];j++){
    		a_N[i][j] +=  pow((1.0/batch_size)*b_grad->N[i][j], 2);										// compute the accumulated squared gradients wrt the biases
    		my_net->biases_N[i][j] -= alpha*(1.0/sqrt(a_N[i][j] + eps))*(1.0/batch_size)*b_grad->N[i][j];  // update biases
    	}
    }

    dealloc_N(b_grad->N);
    free(b_grad);
}


// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses RMSprop optimizer
static void train_batch_RMSprop(NETWORK* my_net, double** inputs, double** outputs, int start, int batch_size, double*** s_W, double** s_N){
    int i, j, k;

    struct GRADIENTS* b_grad = sum_of_grads(my_net, inputs, outputs, start, batch_size);

    // update gradients_W in the NEURAL_NET struct
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			    my_net->gradients_W[i][j][k] = (1.0/batch_size)*b_grad->W[i][j][k];
			}
	    }        
	}

    double alpha = 0.001;      // learning rate 
    double beta = 0.9;       // exponential average hyperparameter
    double eps = 0.000001; // parameter to prevent divide by zero error.

	// apply RMSprop to weights
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
				s_W[i][j][k] = beta*s_W[i][j][k] + (1.0 - beta)*pow((1.0/batch_size)*b_grad->W[i][j][k], 2); 				// update the exponential average of the squared gradients wrt the weights

			   	my_net->weights_W[i][j][k] -= alpha*(1.0/sqrt(s_W[i][j][k] + eps))*(1.0/batch_size)*b_grad->W[i][j][k] ;	// update the weights
			}
	    }               
	}

	dealloc_W(b_grad->W);

    // apply RMSprop to biases
    for (i=1;i<*size;i++){

    	for (j=0;j<network[i];j++){
    			s_N[i][j]  = beta*s_N[i][j] + (1.0 - beta)*pow((1.0/batch_size)*b_grad->N[i][j], 2);				// update the exponential average of the squared gradients wrt the biases
    		    my_net->biases_N[i][j] -= alpha*(1.0/sqrt(s_N[i][j] + eps))*(1.0/batch_size)*b_grad->N[i][j];		// update biases
    	}
    }

    dealloc_N(b_grad->N);
    free(b_grad);
}


// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses RMSprop optimizer
static void train_batch_adadelta(NETWORK* my_net, double** inputs, double** outputs, int start, int batch_size, double*** s_W, double** s_N, double*** deltas_W, double** deltas_N, double*** D_W, double** D_N){
    int i, j, k;

    struct GRADIENTS* b_grad = sum_of_grads(my_net, inputs, outputs, start, batch_size);

    // update gradients_W in the NEURAL_NET struct
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			    my_net->gradients_W[i][j][k] = (1.0/batch_size)*b_grad->W[i][j][k];
			}
	    }        
	}

    double beta = 0.95;       // exponential average hyperparameter
    double eps = 0.000001; // parameter to prevent divide by zero error.

   
	// apply adadelta to weights
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
				s_W[i][j][k] = beta*s_W[i][j][k] + (1.0 - beta)*pow((1.0/batch_size)*b_grad->W[i][j][k], 2);							// update the exponential average of the squared gradients wrt the weights

			    my_net->weights_W[i][j][k] -= (sqrt(D_W[i][j][k]+eps)/sqrt(s_W[i][j][k] + eps))*(1.0/batch_size)*b_grad->W[i][j][k] ;  // update the weights

			    deltas_W[i][j][k] = -(sqrt(D_W[i][j][k]+eps)/sqrt(s_W[i][j][k] + eps))*(1.0/batch_size)*b_grad->W[i][j][k];			// update the delta_weight for this iteration
			    D_W[i][j][k] = beta*D_W[i][j][k] + (1.0 - beta)*pow(deltas_W[i][j][k], 2);														// update the numerator that will multiply the gradient in the next iteration
			}
	    }               
	}

	dealloc_W(b_grad->W);

    // apply adadelta to biases
    for (i=1;i<*size;i++){

    	for (j=0;j<network[i];j++){
    			s_N[i][j]  = beta*s_N[i][j] + (1.0 - beta)*pow((1.0/batch_size)*b_grad->N[i][j], 2);						// update the exponential average of the squared gradients wrt the biases

    		    my_net->biases_N[i][j] -= (sqrt(D_N[i][j]+eps)/sqrt(s_N[i][j] + eps))*(1.0/batch_size)*b_grad->N[i][j];    // update biases

    		    deltas_N[i][j] = (sqrt(D_N[i][j]+eps)/sqrt(s_N[i][j] + eps))*(1.0/batch_size)*b_grad->N[i][j]; 			// update the delta_bias 
    		    D_N[i][j] = beta*D_N[i][j] + (1.0 -beta)*pow(deltas_N[i][j], 2);													// update the numerator that will multiply the gradient in the next iteration
    	}
    }

    dealloc_N(b_grad->N);
    free(b_grad);
}


// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses Adam optimizer
static void train_batch_adam(NETWORK* my_net, double** inputs, double** outputs, int start, int batch_size, double*** m_W, double*** mhat_W, double** m_N, double** mhat_N,  double*** v_W, double*** vhat_W, double** v_N, double** vhat_N, int iter_no){
    int i, j, k;

    struct GRADIENTS* b_grad = sum_of_grads(my_net, inputs, outputs, start, batch_size);

    // update gradients_W in the NEURAL_NET struct
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			    my_net->gradients_W[i][j][k] = (1.0/batch_size)*b_grad->W[i][j][k];
			}
	    }        
	}

    double alpha = 0.001;      // learning rate 
    double beta1 = 0.9;       // exponential average hyperparameter
    double beta2 = 0.999;      // exponential average hyperparameter
    double eps = 0.00000001;    // parameter to prevent divide by zero error.


    // apply the adam algorithm to the weights
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			    m_W[i][j][k] = beta1*m_W[i][j][k] + (1.0 - beta1)*(1.0/batch_size)*b_grad->W[i][j][k];         // update 1st moment vector
			    mhat_W[i][j][k] = m_W[i][j][k]/(1.0 - pow(beta1, iter_no));                                             // fill in mhat

			    v_W[i][j][k] = beta2*v_W[i][j][k] + (1.0 - beta2)*pow((1.0/batch_size)*b_grad->W[i][j][k], 2); // update 2nd moment vector
			    vhat_W[i][j][k] = v_W[i][j][k]/(1.0 - pow(beta2, iter_no));												// fill in vhat

			    my_net->weights_W[i][j][k] -= alpha*(1.0/(sqrt(vhat_W[i][j][k]) + eps))*mhat_W[i][j][k];				// update weights
			}
	    }        
	}


	dealloc_W(b_grad->W);

	// apply adam algorithm to biases
    for (i=1;i<*size;i++){

    	for (j=0;j<network[i];j++){
    		    m_N[i][j]  = beta1*m_N[i][j] + (1.0 - beta1)*(1.0/batch_size)*b_grad->N[i][j];				// update 1st moment 
    		    mhat_N[i][j]  = m_N[i][j]/(1.0 - pow(beta1, iter_no));												// fill in mhat

    		    v_N[i][j]  = beta2*v_N[i][j] + (1.0 - beta2)*pow((1.0/batch_size)*b_grad->N[i][j], 2);		// update 2nd moment
    		    vhat_N[i][j]  = v_N[i][j]/(1.0 - pow(beta2, iter_no));												// fill in vhat

    		    my_net->biases_N[i][j] -= alpha*(1.0/(sqrt(vhat_N[i][j]) + eps))*(1.0/batch_size)*mhat_N[i][j];		// update biases
    	}
    }


    dealloc_N(b_grad->N);
    free(b_grad);
}




// train the network
void train_network(NETWORK* my_net, double** inputs, double** outputs, int no_of_inputs, int batch_size, int epochs, int optimizer){
	int i,j;
	int iterations = ceil(no_of_inputs/batch_size);
	int remaining_samples = no_of_inputs - (batch_size*(iterations-1));


		
	for (j=0;j<epochs;j++){
      switch(optimizer){
			// mini batch SGD
			case 1:{

				for (i=0;i<iterations-1;i++){
					train_batch(my_net, inputs, outputs, batch_size*i, batch_size);

					if (min_reached(my_net)){
						printf("minibatch algorithm has reached sufficient precision after %d iterations on epoch number %d \n", i+1, j+1);
						return;
					}
				}
				train_batch(my_net, inputs, outputs, batch_size*(iterations-1), remaining_samples);
            break;
			}

			// momentum
			case 2:{
				double*** v_W = init_W('z');
				double** v_N = init_N('z');

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
            break;
			}

			// adagrad
			case 3:{
				double*** a_W = init_W('z');
				double** a_N = init_N('z');

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
            break;
			}

			// RMSprop
			case 4:{
				double*** s_W = init_W('z');
				double** s_N = init_N('z');

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
            break;
			}

			// adadelta
			case 5:{
				double*** s_W = init_W('z');
				double** s_N = init_N('z');

				double*** D_W = init_W('z');
				double** D_N = init_N('z');

				double*** deltas_W = init_W('z');
				double** deltas_N = init_N('z');

	
				for (i=0;i<iterations-1;i++){

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
            break;
			}

			// adam
			case 6:{
				double*** m_W = init_W('z');
				double** m_N = init_N('z');

				double*** mhat_W = alloc_W();
				double** mhat_N = alloc_N();

				double*** v_W = init_W('z');
				double** v_N = init_N('z');

				double*** vhat_W = alloc_W();
				double** vhat_N = alloc_N();

	
				for (i=0;i<iterations-1;i++){

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
            break;
			}
      }
	}

	printf("algorithm has completed training of %d epochs\n", epochs);
	return;
}




//
// END FUNCTIONS FOR NETWORK TRAINING
//



//
// BEGIN FUNCTIONS TO EXPORT/IMPORT NEURAL NET WEIGHTS/BIASES
//





// export weights to txt file of comma separated values
void export_weights(NETWORK* my_net, char* filename){
	int i, j, k;
	FILE *fp;
	fp = fopen(filename, "w+");

	for (i=0;i<*size-1;i++){
		fprintf(fp, "?\n");

		for (j=0;j<network[i];j++){
			for (k=0;k<network[i+1]-1;k++){
			    fprintf(fp, "%lf, ", my_net->weights_W[i][j][k]);
			}
			fprintf(fp, "%lf\n", my_net->weights_W[i][j][network[i+1]-1]);
	    }       
	}
	fclose(fp);
}


// export biases to txt file of comma separated values
void export_biases(NETWORK* my_net, char* filename){
	int i, j, k;
	FILE *fp;
	fp = fopen(filename, "w+");



	for (j=0;j<network[0]-1;j++){
		    fprintf(fp, "%lf, ", 0.0);
	}
	fprintf(fp, "%lf\n", 0.0);

	for (i=1;i<*size;i++){

    	for (j=0;j<network[i]-1;j++){
    		    fprintf(fp, "%lf, ", my_net->biases_N[i][j]);
    	}
    	fprintf(fp, "%lf\n", my_net->biases_N[i][j]);
    }

	fclose(fp);
}


// import weights from txt file of comma separated values
void import_weights(NETWORK* my_net, char* filename){
	int i,j, k;
	int buf_size = 10000;
	char buf[buf_size];
	FILE *fp;

	int* start_posns = (int*)calloc(1, sizeof(int));


	fp = fopen(filename, "r");


	int count = 0;
	
	do {
		fgets(buf, buf_size, fp);
		if (buf[0] == '?'){
			start_posns[count] = ftell(fp);
			start_posns = (int*)realloc(start_posns, sizeof(int));
			count+=1;
		}
	} while(!feof(fp));


	int matr_cols[count];

	for (i=0;i<count;i++){
		fseek(fp, start_posns[i], SEEK_SET);
		fgets(buf, buf_size, fp);

		j=0;
		char*ptr_1;
		char* tok_1 = strtok(buf, ",");

		while(tok_1 != NULL){
			j+=1;
			tok_1 = strtok(NULL, ",");
		}

		matr_cols[i] = j;
	}

	int matr_rows[count];

	for (i=0;i<count-1;i++){
		fseek(fp, start_posns[i], SEEK_SET);
		j=0;
		while (ftell(fp) != start_posns[i+1]){
			fgets(buf, buf_size, fp);
			j+=1;
		}
		matr_rows[i] = j-1;

	}

	fseek(fp, start_posns[count-1], SEEK_SET);
	j=0;
	while(!feof(fp)){
		fgets(buf, buf_size, fp);
		j+=1;
	}

	matr_rows[count-1] = j-1;


	double*** weights = (double***)calloc(count, sizeof(double**));

	for (i=0;i<count;i++){
		weights[i] = (double**)calloc(matr_rows[i], sizeof(double*));
		for (j=0;j<matr_rows[i];j++){
			weights[i][j] = (double*)calloc(matr_cols[i], sizeof(double));
		}
	}

	rewind(fp);

	char* ptr;
	char* tok;

	for (i=0;i<count;i++){
		fseek(fp, start_posns[i], SEEK_SET);
		for (j=0;j<matr_rows[i];j++){
			fgets(buf, buf_size, fp);
			tok = strtok(buf, ",");
			weights[i][j][0] = strtod(tok, &ptr);
			for (k=1;k<matr_cols[i];k++){
				tok = strtok(NULL, ",");
				weights[i][j][k] = strtod(tok, &ptr);
			}
			
		}
	}

	free(start_posns);
	fclose(fp);

	dealloc_W(my_net->weights_W);
	
	my_net->weights_W = weights;

	return;
}


// import weights from txt file of comma separated values
void import_biases(NETWORK* my_net, char* filename){

	int i, j, k;
	int buf_size = 10000;
	char buf[buf_size];
	FILE *fp;


	fp = fopen(filename, "r");

	int count = 0;
	while (!feof(fp)){
		fgets(buf, buf_size, fp);
		count+=1;
	}
	count -= 1;

	rewind(fp);

	int line_length[count];
	char* tok;

	for (i=0;i<count;i++){
		fgets(buf, buf_size, fp);

		j=0;

		tok = strtok(buf, ",");

		while (tok != NULL){
			j+=1;
			tok = strtok(NULL, ",");
		}

		line_length[i] = j;
	}

	double** biases = (double**)calloc(count, sizeof(double*));

	for (i=0; i<count; i++){
		biases[i] = (double*)calloc(line_length[i], sizeof(double));
	}

	rewind(fp);

	for (i=0;i<count;i++){
		fgets(buf, buf_size, fp);
		char* ptr;

		tok = strtok(buf, ",");
		biases[i][0] = strtod(tok, &ptr);

		for (j=1;j<line_length[i];j++){
			tok = strtok(NULL, ",");
			biases[i][j] = strtod(tok, &ptr);
		}
	}

	fclose(fp);

	dealloc_N(my_net->biases_N);

	my_net->biases_N = biases;
	return;
}






//
// END FUNCTIONS TO EXPORT/IMPORT NEURAL NET WEIGHTS/BIASES
//


// print the output activations of the neural network
void print_output(NETWORK* my_net){

	for (int i=0;i<network[*size-1];i++){
		printf("%lf, ", my_net->activations_N[*size-1][i]);
	}
}