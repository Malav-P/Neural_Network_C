#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
//*********************************************************

// applies activation function to a structure of type L
void activ(double* lyr_L, int len){
	int i;

	for (i=0;i<len;i++){

		if (lyr_L[i]<0){
			lyr_L[i] = 0.0;
		}

		else{
			lyr_L[i] = 0.001*lyr_L[i];
		}

	}
}

//---------------------------------------------------------
//*********************************************************

// derivative of activation function for hidden layer
double activ_deriv(double x){

	if (x<0){
		return 0;
	}

	else{
		return 1;
	}
}

//---------------------------------------------------------
//*********************************************************

// applies activation function to output layer in neural network (which has structure type L). I have chosen the activation function.
void out_activ(double* lyr_L, int len){
	double sum = 0;
	int i;

	for (i=0;i<len;i++){
		sum += exp(lyr_L[i]);
	}

	for (i=0; i<len;i++){
		lyr_L[i] = (1/sum)*exp(lyr_L[i]);
	}
}

//---------------------------------------------------------
//*********************************************************

// given a layer lyr_L of structure type L and and index i, this function returns the derivative of the softmax function wrt to the variable in the (index)th position of lyr_L
double out_activ_deriv(double* lyr_L, int len, int index){
	int i;
	double sum = 0;
	double x = lyr_L[index];

	for (i=0;i<len;i++){
		sum += exp(lyr_L[i]);
	}

	return (1/sum)*exp(x)*(1-(1/sum)*exp(x));
}

//---------------------------------------------------------
//*********************************************************

// returns a random number of type double from the range (min, max)
double randfrom(double min, double max){
    double range = (max - min); 
    double div = RAND_MAX / range;

    return min + (rand() / div);
}

//---------------------------------------------------------

// declare global pointers *network and *size
// *size is an int. Corresponds to number of layers in neural net (a.k.a the length of the array that network points to)
// network points to first element in an arry of int. Each entry corresponds to number of neurons in the neural net layer.
int* network;
int* size;


// declare a struct for the neural net.
struct NEURAL_NET{
	double** activations_N;  // an order two pointer. activations[i][j] is the activation for the jth neuron in the ith layer.
	double** biases_N;       // an order two pointer. biases[i][j] is the bias for the jth neuron in the ith layer.
	double*** weights_W;     // and order three pointer. weights[i][j][k] is the weight connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer
};

// declare a struct for the gradients to be computed in back_propagation
struct GRADIENTS{
	double*** dcdw_W;  // an order three pointer. dcdw[i][j][k] corresponds to the partial derivative of the cost function wrt weights[i][j][k] (defined in the struct above^).
	double** dcdb_N;   // and order two pointer. dcdb[i][j] corresponds to the partical derivative of the cost function wrt to biases[i][j] (defined in the struct above^).
};


//********************************************************************************************

// assign a pointer to an int and a pointer to the first element of an array of int that describe the number of layers and the number of neurons in each layer of the network.
/* 

n: a pointer to the first element of an array of int
s: a pointer to an int.

*/
void network_params(int* n, int* s){

	network = n;  // assign the global pointer network to the given pointer n.
	size = s;     // assign the global pointer size to the given pointer s.
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


// BEGIN FUNCTION DEFINITIONS FOR INITIALIZING NEURAL NETWORK LAYERS, BIASES, AND WEIGHTS.


//*********************************************************************************************

// allocate memory for a structure of type L. L is usually a layer of neurons holding an activation, intermediate value, or bias.
double* alloc_L(int len){
	double* ptr_L = (double*)calloc(len, sizeof(double));

	return ptr_L;
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory in a shape of type N. N is usually a network of neurons.
// ptr_N[i][j] is the jth neuron in the ith layer
double** alloc_N(){
	int i;
	double** ptr_N = (double**)calloc(*size, sizeof(double*));

	for (i=0;i<*size;i++){
		ptr_N[i] = alloc_L(network[i]);
	}

	return ptr_N;
}
//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type N. Initialize all entries to 0.
// ptr_N[i][j] is the entry for the jth neuron in the ith layer.
double** init_N(){
	int i, j;
	double** ptr_N = alloc_N();

	for (i=0;i<*size;i++){

		for (j=0;j<network[i];j++){
			ptr_N[i][j] = 0;
		}
	}

	return ptr_N;
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type N. Initialize all entries to to a random double between -0.01 and 0.01.
// ptr_N[i][j] is the entry for the jth neuron in the ith layer.
double** r_init_N(){
	int i, j;
	double** ptr_N = alloc_N();

	for (i=0;i<*size;i++){

		for (j=0;j<network[i];j++){
			ptr_N[i][j] = randfrom(-0.01, 0.01);
		}
	}

	return ptr_N;
}

//---------------------------------------------------------------------------------------------
//**********************************************************************************************

// deallocate memory of a structure of type N.
void dealloc_N(double** ptr_N){
	int i;

	for (i=0;i<*size;i++){
		free(ptr_N[i]);
	}

	free(ptr_N);
}
//----------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type w. w is a matrix  that can hold the weights connecting two structures of type L.
// ptr_w[j][k] is the entry connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
double** alloc_w(int i){
	int j;
	double** ptr_w = (double**)calloc(network[i], sizeof(double*));

	for (j=0;j<network[i];j++){
		ptr_w[j] = (double*)calloc(network[i+1], sizeof(double));
	}

	return ptr_w;
}
//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type W. W can be thought of as a list of structure type w.
// ptr_W[i][j][k] is the entry connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
double*** alloc_W(){
	int i, j, k;
	double*** ptr_W = (double***)calloc(*size-1, sizeof(double**));

	for (i=0;i<*size-1;i++){
		ptr_W[i] = alloc_w(i);
	}

	return ptr_W;
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type W and initialize all entries to 0.
// ptr_W[i][j][k] is the weight connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
double*** init_W(){
	int i, j, k;
	double*** ptr_W = alloc_W();

	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
				ptr_W[i][j][k] = 0;
			}
		}
	}

	return ptr_W;
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type W and intialize each entry with a random value between -1 and 1.
// ptr_W[i][j][k] is the entry connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
double*** r_init_W(){
	int i, j, k;
	double*** ptr_W = alloc_W();

	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
				ptr_W[i][j][k] = randfrom(-1,1);
			}
		}
	}

	return ptr_W;
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// deallocate memory that was previously allocated using alloc_w.
void dealloc_w(double** ptr_w, int i){
	int j;

	for (j=0;j<network[i];j++){
		free(ptr_w[j]);
	}

	free(ptr_w);

}
//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// deallocate memory for a previously allocated structure of type W.
void dealloc_W(double*** ptr_W){
	int i, j;

	for (i=0;i<*size-1;i++){
		dealloc_w(ptr_W[i], i);
	}

	free(ptr_W);
}
//---------------------------------------------------------------------------------------------

//*********************************************************************************************

// initialize the network with random weights and biases.
struct NEURAL_NET* initialize_network(int* n, int* s){
	network_params(n, s);
	struct NEURAL_NET* net = malloc(sizeof*net);

	net->activations_N = alloc_N();   // net->activations_N[i][j] is the activation of the jth neuron in the ith layer
	net->biases_N = r_init_N();       // net->biases_N[i][j] is the bias of the jth neuron in the ith layer
	net->weights_W = r_init_W();      // net->weights_W[i][j][k] is the weight connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.

	return net;
}

//---------------------------------------------------------------------------------------------

// BEGIN OUTLINING FUNCTIONS MEANT TO ALTER THE NEURAL NETWORK. THIS INCLUDES FEED FORWARD, BACK_PROP, TRAINING, ETC.


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

// same thing as feed_fwd_H but keeps track of intermediate values for layers.
void comp_grad_H_H(int i, struct NEURAL_NET* my_net, void (*f)(double*, int), double** itrmd_N){
	int j, k;
	double* curr_L = my_net->activations_N[i];
	double* next_L = my_net->activations_N[i+1];
	double** wgt_matr_w = my_net->weights_W[i];

	for (j=0;j<network[i+1];j++){
		double answer=0;

		for (k=0;k<network[i];k++){
			answer += curr_L[k]*wgt_matr_w[k][j];
		}

		itrmd_N[i+1][j] = answer + my_net->biases_N[i+1][j];
		next_L[j] = answer + my_net->biases_N[i+1][j];
	}

	(*f)(next_L, network[i+1]);
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// same thing as feed_fwd but keeps track of intermediate values for layers.
void comp_grad_H(double input[], struct NEURAL_NET* my_net, double** itrmd_N){
	int i;

	for (i=0;i<network[0];i++){
		my_net->activations_N[0][i] = input[i];
	}

	for (i=0;i<*size-2;i++){
		comp_grad_H_H(i, my_net, &activ, itrmd_N);
	}

	comp_grad_H_H(*size-2, my_net, &out_activ, itrmd_N);
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// compute the gradient of the cost function wrt the weights and wrt to the biases
struct GRADIENTS* comp_grad(double input[], double output[], struct NEURAL_NET* my_net){
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

	grad->dcdw_W = DCDW_W;   // grad->dcdw[i][j][k] is the partial derivative of the cost function wrt weights[i][j][k].
	grad->dcdb_N = DCDB_N;   // grad->dcdb[i][j] is the partial derivative of the cost function wrt biases[i][j].

	dealloc_N(itrmd_N);
	dealloc_N(err_N);

	return grad;
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// update a structure of type W by adding another structure of type W to it.
void add_to_W(double*** cum_sum_W, double*** matr_W){
	int i, j, k;
    
    for (i=0;i<*size-1;i++){

    	for (j=0;j<network[i];j++){

    		for (k=0;k<network[i+1];k++){
    		    cum_sum_W[i][j][k] += matr_W[i][j][k];
    		}
    	}
    }
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// update a structure of type N by adding another structure of type N to it.
void add_to_N(double** cum_sum_N, double** matr_N){
	int i, j;

	for (i=0;i<*size;i++){

		for (j=0;j<network[i];j++){
			cum_sum_N[i][j] += matr_N[i][j];
		}
	}
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses mini batch SGD.
void train_batch(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int start, int batch_size){
    int i, j, k;
    int in_size = network[0];
    int out_size = network[*size-1];

    double*** sum_dcdw_W = init_W();
    double** sum_dcdb_N = init_N();

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

        add_to_W(sum_dcdw_W, g->dcdw_W);
        add_to_N(sum_dcdb_N, g->dcdb_N);

        dealloc_W(g->dcdw_W);
        dealloc_N(g->dcdb_N);

        free(g);
    }

    double alpha = 0.1; // learning rate

	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			   my_net->weights_W[i][j][k] -= alpha*(1.0/batch_size)*sum_dcdw_W[i][j][k];
			}
	    }               
	}

    dealloc_W(sum_dcdw_W);


    for (i=1;i<*size;i++){

    	for (j=0;j<network[i];j++){
    		    my_net->biases_N[i][j] -= alpha*(1.0/batch_size)*sum_dcdb_N[i][j];
    	}
    }


    dealloc_N(sum_dcdb_N);
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses mini batch SGD with momentum.
void train_batch_momentum(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int start, int batch_size, double*** v_W, double** v_N){
    int i, j, k;
    int in_size = network[0];
    int out_size = network[*size-1];

    double*** sum_dcdw_W = init_W();
    double** sum_dcdb_N = init_N();

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

        add_to_W(sum_dcdw_W, g->dcdw_W);
        add_to_N(sum_dcdb_N, g->dcdb_N);

        dealloc_W(g->dcdw_W);
        dealloc_N(g->dcdb_N);

        free(g); 
    }

    double beta = 0.9;   // exponential average hyperparameter
    double alpha = 0.1;  // learning rate

 
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			    v_W[i][j][k] = beta*v_W[i][j][k] + (1.0-beta)*(1.0/batch_size)*sum_dcdw_W[i][j][k];
			}
	    }        
	}

	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			   my_net->weights_W[i][j][k] -= alpha*v_W[i][j][k];
			}
	    }               
	}

	dealloc_W(sum_dcdw_W);


    for (i=1;i<*size;i++){

    	for (j=0;j<network[i];j++){
    		    v_N[i][j] = beta*v_N[i][j] + (1-beta)*(1.0/batch_size)*sum_dcdb_N[i][j];
    	}
    }

    for (i=1;i<*size;i++){

    	for (j=0;j<network[i];j++){
    		    my_net->biases_N[i][j] -= alpha*v_N[i][j];
    	}
    }

    dealloc_N(sum_dcdb_N);
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// train the network
void train_network(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int no_of_inputs, int batch_size, int epochs, int optimizer){
	int i,j;
	if (optimizer == 1){
		int iterations = ceil(no_of_inputs/batch_size);
		int remaining_samples = no_of_inputs%batch_size;

		if (remaining_samples == 0){

			for (j=0;j<epochs;j++){

				for (i=0;i<iterations;i++){
					train_batch(my_net, inputs, outputs, batch_size*i, batch_size);
				}
			}

		}


		else{

			for (j=0;j<epochs;j++){

				for (i=0;i<iterations-1;i++){
					train_batch(my_net, inputs, outputs, batch_size*i, batch_size);
				}
				train_batch(my_net, inputs, outputs, batch_size*iterations, remaining_samples);
			}

		}
	}
	else if (optimizer == 2){
		int iterations = ceil(no_of_inputs/batch_size);
		int remaining_samples = no_of_inputs%batch_size;


		if (remaining_samples == 0){

			for (j=0;j<epochs;j++){
				double*** v_W = init_W();
				double** v_N = init_N();

				for (i=0;i<iterations;i++){
					train_batch_momentum(my_net, inputs, outputs, batch_size*i, batch_size, v_W, v_N);
				}

				dealloc_W(v_W);
				dealloc_N(v_N);
			}

		}


		else{

			for (j=0;j<epochs;j++){
				double*** v_W = init_W();
				double** v_N = init_N();

				for (i=0;i<iterations-1;i++){
					train_batch_momentum(my_net, inputs, outputs, batch_size*i, batch_size, v_W, v_N);
				}
				train_batch_momentum(my_net, inputs, outputs, batch_size*iterations, remaining_samples, v_W, v_N);

				dealloc_W(v_W);
				dealloc_N(v_N);
			}

		}
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
