# Neural_Network_C
An implementation of a standard feed-forward neural network written in C.

# Overview

The files [nn_functions.h](./nn_functions.h) and [nn_functions.c](./nn_functions.c) contain all the necessary files to create and train a neural network. Please be aware that this code is not optimized for speed or size,
it is still a work in progress. The neural net is designed as a classifier and has the following properties: <br />

- hidden layers use relU activation 
- output layer uses softmax activation
- back propagation uses mean-squared-error loss


# Walkthrough of the Repository

The following is a walkthrough of how the code is structured to better help the user understand how to work with this respository.

## Example Code

The [STABLE_BUILD.zip](./STABLE_BUILD.zip) file contains a working implementation of this network on the MNIST dataset. The following code snippet will help outline the general structure of the code. NOTE: running this code snippet alone will result in error. Please download [STABLE_BUILD.zip](./STABLE_BUILD.zip) and compile + run the example_mnist_dataset.c for a working implementation.

```C

int main(){
    int batch_size = 50;
    int epochs = 3;
    int optimizer = 3;

    struct DATA* my_data = load_mnist();

    int my_n[] = {784, 100, 100, 10};
    int s = sizeof(my_n)/sizeof(int);

    struct NEURAL_NET* my_net = initialize_network(my_n, &s);

    train_network(my_net, my_data->train_images, my_data->train_labels, N_TRAIN, batch_size, epochs, optimizer);
    test_network(my_net, my_data);

    export_weights(my_net, "weights.txt");
    export_biases(my_net, "biases.txt");

    free_network(my_net);
}

```
## Structure of the Neural Net

In the example code above, there is a reference to a `struct NEURAL_NET`. This is a user-defined datatype. For any instance `my_net` of this struct, we have:

`my_net.activations_N` : activations of the neurons in the network. <br />
`my_net.biases_N` : biases of the neurons in the network. <br />
`my_net.weights_W`: weights between adjacent layers in the network. <br />

Accessing individual activations, biases, or weights is simple: <br />


`my_net.activations_N[i][j]` : activation of the jth neuron in the ith layer of the network.  <br />
`my_net.biases_N[i][j]` : bias of the jth neuron in the ith layer of the network <br />
`my_net.weights_W[i][j][k]`: weight connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer <br />

## Initializing the Network

Before intializing the network, the size of the input and output layers as well the the hidden layers must be specified. In the example code above, `my_n[] = {784, 101, 101, 10}` specifies the following: <br />

- an input layer of 784 neurons <br />
- a hidden layer of 101 neurons <br />
- a hidden layer of 101 neurons <br />
- an output layer of 10 neurons <br />

Calling `initialize_network(my_n, &s)` initializes a network with the above structure. Biases and weights are He initialized.

## Training the Network

The function `train_network(...)` is a call to train the network. Each of the inputs is described below. <br />

- `my_net` : a pointer to an an instance of `NEURAL_NET`
- `my_data->train_images` : a 2D array of type `double` containing the training data. It is the user's responsibility to ensure that the dimensions of the array match the size of the input layer
- `my_data->train_labels` : a 2D array of type `double` containing the training labels. It is the user's responsibility to ensure that the dimesions of this array match the size of the output layer
- `N_TRAIN` : total number of training examples for this training session
- `batch_size` : subset size of the training sample the algorithm uses when completing a singular update of the weights and biases
- `epochs` : number of epochs for the the training session
- `optimizer` : the descent algorithm used to train the weights.

### More on optimizers

Currently, the source code supports only the following gradient descent algorithms. An optimizer is called by using an integer specifier, specified below.

- 1 : Mini-batch SGD
- 2 : Momentum
- 3 : Adagrad
- 4 : RMSProp
- 5 : Adadelta
- 6 : Adam

## Exporting/Importing Training Data Functions

After a network is trained, one can export the weights and biases to `.txt` files use `export_weights` and `export_biases`. The functions require two arguments:

- a pointer to the network whose weights/biases are to be exported
- a string representing the file name to be exported to

See the example code above for the implementation. <br />
Similarly one can do the reverse and import weights and biases using `import_weights` and `import biases`. These functions require that a network be initialized and take the same two arguments as above.
