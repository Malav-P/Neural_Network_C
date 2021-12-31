#include <math.h>
//*********************************************************

// applies activation function to a structure of type L. I have chosen the relU function
void activ(double* lyr_L, int len){
	int i;

	for (i=0;i<len;i++){

		if (lyr_L[i]<0){
			lyr_L[i] = 0.0;
		}

	}
}

//---------------------------------------------------------
//*********************************************************

// derivative of activation function for hidden layer. I have chosen the derivative of the relU function
double activ_deriv(double x){

	if (x<0){
		return 0.0;
	}

	else{
		return 1.0;
	}
}

//---------------------------------------------------------
//*********************************************************

// applies activation function to output layer in neural network (which has structure type L). I have chosen the softmax activation function.
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