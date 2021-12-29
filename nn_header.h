//*********************************************************

// returns a random number of type double from the range (min, max)
/*

min: double representing lower bound of return value
max: double representing upper bound of return value

*/
double randfrom(double min, double max);

//---------------------------------------------------------

// declare global pointers *network and *size
// *size is an int. Corresponds to number of layers in neural net (a.k.a the length of the array that network points to)
// network points to first element in an arry of int. Each entry corresponds to number of neurons in the neural net layer.
int* network;
int* size;


// declare a struct for the neural net.
struct NEURAL_NET{
	double** activations;  // an order two pointer. activations[i][j] is the activation for the jth neuron in the ith layer.
	double** biases;       // an order two pointer. biases[i][j] is the bias for the jth neuron in the ith layer.
	double*** weights;     // and order three pointer. weights[i][j][k] is the weight connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer
};

//*********************************************************************************************

// initialize the network with random weights and biases.
// return_var->activations[i][j] is the activation of the jth neuron in the ith layer
// return_var->biases[i][j] is the bias of the jth neuron in the ith layer
// return_var->weights[i][j][k] is the weight connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
struct NEURAL_NET* initialize_network(int* n, int* s);

//---------------------------------------------------------------------------------------------

//*********************************************************************************************

// compute the first layer's activations from the input. Feed these activations forward throughout the rest of the network.
void feed_fwd(double input[], struct NEURAL_NET* my_net);

//---------------------------------------------------------------------------------------------

//*********************************************************************************************

// print the output activations of the neural network
void print_output(struct NEURAL_NET* my_net);

//---------------------------------------------------------------------------------------------

//*********************************************************************************************

// train the network
void train_network(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int no_of_inputs, int batch_size, int epochs, int optimizer);

//---------------------------------------------------------------------------------------------

//*********************************************************************************************

// dealloc memory associated with network
void free_network(struct NEURAL_NET* my_net);

//---------------------------------------------------------------------------------------------