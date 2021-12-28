# Neural_Network_C
An implementation of a standard feed-forward neural network written entirely in C.

# Overview

The files [nn_header.h](./nn_header.h) and [nn_functions.c](./nn_functions.c) contain all the necessary files to create and train a neural network. Please be aware that this code is not optimized for speed or size,
it is still a work in progress. 

# Walkthrough of the Repository

The following is a walkthrough of how the code is structured to better help the user understand how to work with this respository.

## Example Code

The [OLD_BUILD.zip](./OLD_BUILD.zip) file contains a working implementation of this network on the MNIST dataset. 

```C

int main(){
    int i,j;

    load_mnist();   // load the data from MNIST into 2D arrays.
    process_labels();  // format the data appropropriately before the function to train the network is called.

    printf("processing data complete, now training network...\n\n");

    int my_n[] = {784, 101, 101, 10};  
    int s = sizeof(my_n)/sizeof(int);

    struct NEURAL_NET* my_net = initialize_network(my_n, &s);

    train_network(my_net, train_image, label_train, no_of_train_pts, batch_size, epochs, "minibatch");
    test_network(my_net);

}

```
## Structure of the Neural Net

In the example code above, there is a reference to a `struct NEURAL_NET`. This is a user-defined datatype. For any instance `network` of this struct, we have:

`network.activations_N` : the activations of the neurons in the network. <br />
`network.biases_N` : the biases of the neurons in the network. <br />
`network.weights_W`: the weights between adjacent layers in the network. <br />

Accessing individual activations, biases, or weights is simple: <br />


`network.activations_N[i][j]` : the activation jth neuron in the ith layer of the network.  <br />
`network.biases_N[i][j]` : the bias of the jth neuron in the ith layer of the network <br />
`network.weights_W[i][j][k]`: the weight connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer <br />

## Initializing the Network

Before intializing the network, the size of the input and output layers as well the the hidden layers must be specified. In the example code above, `my_n[] = {784, 101, 101, 10}` specifies the following: <br />

- an input layer of 784 neurons <br />
- a hidden layer of 101 neurons <br />
- a hidden layer of 101 neurons <br />
- an output layer of 10 neurons <br />

Calling `initialize_network(my_n, &s)` initializes a network with the above structure. Biases are intialized with a random float value in (-0.01, 0.01) and weights are intialized with a random float value in (-1, 1).

## Training the Network

The function `train_network(...)` is a call to train the network. Each of the inputs is described below. <br />

- `my_net` : a pointer to an an instance of `NEURAL_NET`
- `train_image` : a 2D array of type `double` containing the training data. It is the user's responsibility to ensure that the dimensions of the array match the size of the input layer <br />
- `label_train` : a 2D array of type `double` containing the training labels. It is the user's responsibility to ensure that the dimesions of this array match the size of the output layer <br />
- `no_of_train_pts` : total number of training examples for this training session
- `batch_size' : the batch size of the data
- `epochs` : number of epochs for the the training session
- `optimizer` : the descent algorithm used to train the weights. For the time being mini-batch stochastic gradient descent is the only available optimizer. More to come
