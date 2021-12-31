
//**********************************************************

// generate a random value from a uniform distribution between 0 and 1
double rand_gen();

//----------------------------------------------------------
//*********************************************************

// generate a random value from the standard normal distribution
double normalRandom();

//----------------------------------------------------------
//*********************************************************

// returns a random number from the range (min, max)
double randfrom(double min, double max);

//---------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type L. L is usually a layer of neurons holding an activation, intermediate value, or bias.
double* alloc_L(int len);

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory in a shape of type N. N is usually a network of neurons.
// ptr_N[i][j] is the jth neuron in the ith layer
double** alloc_N();
//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type N. Initialize all entries to 0.
// ptr_N[i][j] is the entry for the jth neuron in the ith layer.
double** init_N();

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type N. Initialize all entries to 1.0.
// ptr_N[i][j] is the entry for the jth neuron in the ith layer.
double** one_init_N();

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type N. Initialize all entries to to a random double between -0.01 and 0.01.
// ptr_N[i][j] is the entry for the jth neuron in the ith layer.
double** r_init_N();

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type N. Initialize all entries according to He initialization procedure
// ptr_N[i][j] is the entry for the jth neuron in the ith layer.
double** he_init_N();

//---------------------------------------------------------------------------------------------
//**********************************************************************************************

// deallocate memory of a structure of type N.
void dealloc_N(double** ptr_N);
//----------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type w. w is a matrix  that can hold the weights connecting two structures of type L.
// ptr_w[j][k] is the entry connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
double** alloc_w(int i);
//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type W. W can be thought of as a list of structure type w.
// ptr_W[i][j][k] is the entry connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
double*** alloc_W();

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type W and initialize all entries to 0.
// ptr_W[i][j][k] is the weight connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
double*** init_W();

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type W and initialize all entries to 1.0.
// ptr_W[i][j][k] is the weight connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
double*** one_init_W();

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type W and intialize each entry with a random value between -0.01 and 0.01.
// ptr_W[i][j][k] is the entry connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
double*** r_init_W();

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type W and intialize each entry according to He initialization procedure
// ptr_W[i][j][k] is the entry connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
double*** he_init_W();

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// deallocate memory that was previously allocated using alloc_w.
void dealloc_w(double** ptr_w, int i);
//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// deallocate memory for a previously allocated structure of type W.
void dealloc_W(double*** ptr_W);
//---------------------------------------------------------------------------------------------