
//*********************************************************

// applies activation function to a structure of type L. I have chosen the relU function
void activ(double* lyr_L, int len);

//---------------------------------------------------------
//*********************************************************

// derivative of activation function for hidden layer. I have chosen the derivative of the relU function
double activ_deriv(double x);

//---------------------------------------------------------
//*********************************************************

// applies activation function to output layer in neural network (which has structure type L). I have chosen the softmax activation function.
void out_activ(double* lyr_L, int len);

//---------------------------------------------------------
//*********************************************************

// given a layer lyr_L of structure type L and and index i, this function returns the derivative of the softmax function wrt to the variable in the (index)th position of lyr_L
double out_activ_deriv(double* lyr_L, int len, int index);

//---------------------------------------------------------