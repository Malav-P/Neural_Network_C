// declare global pointers *network and *size
// *size is an int. Corresponds to number of layers in neural net (a.k.a the length of the array that network points to)
// network points to first element in an arry of int. Each entry corresponds to number of neurons in the neural net layer.
int* network;
int* size;


// declare a struct for the neural net.
struct NEURAL_NET{
	double** activations_N;  // an order two pointer. activations_N[i][j] is the activation for the jth neuron in the ith layer.
	double** biases_N;       // an order two pointer. biases_N[i][j] is the bias for the jth neuron in the ith layer.
	double*** weights_W;     // an order three pointer. weights_W[i][j][k] is the weight connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer
	double*** gradients_W;   // an order three pointer. gradients_W[i][j][k] is the gradient of the cost function wrt weights_W[i][j][k] that was computed in the last iteration of training the network
};

// declare a struct for the gradients to be computed in back_propagation
struct GRADIENTS{
	double*** dcdw_W;  // an order three pointer. dcdw[i][j][k] corresponds to the partial derivative of the cost function wrt weights[i][j][k] (defined in the struct above^).
	double** dcdb_N;   // and order two pointer. dcdb[i][j] corresponds to the partical derivative of the cost function wrt to biases[i][j] (defined in the struct above^).
};

// declare a struct for batch gradients
struct BATCH_GRADIENTS{
	double*** sum_dcdw_W;
	double** sum_dcdb_N;
};
