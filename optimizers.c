#include "structs_and_globals.h"
#include "alloc_dealloc.h"
#include "activations.h"
#include <stdlib.h>
#include <math.h>


// BEGIN HELPER FUNCTION DEFINITIONS-------------------------------------------



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

// compute the elemenwise sum of all the gradients in a batch of training samples.
struct BATCH_GRADIENTS* sum_of_grads(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int start, int batch_size){
	struct BATCH_GRADIENTS* b_grad = malloc(sizeof*b_grad);

	int j, k;
    int in_size = network[0];
    int out_size = network[*size-1];

    double*** summation_dcdw_W = init_W();
    double** summation_dcdb_N = init_N();

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

        add_to_W(summation_dcdw_W, g->dcdw_W);
        add_to_N(summation_dcdb_N, g->dcdb_N);

        dealloc_W(g->dcdw_W);
        dealloc_N(g->dcdb_N);

        free(g);
    }
    b_grad->sum_dcdw_W = summation_dcdw_W;
    b_grad->sum_dcdb_N = summation_dcdb_N;

    return b_grad;
}

//---------------------------------------------------------------------------------------------




// END HELPER FUNCTION DEFINTIONS------------------------------------------------




//*********************************************************************************************

// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses mini batch SGD.
void train_batch(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int start, int batch_size){
    int i, j, k;

    struct BATCH_GRADIENTS* b_grad = sum_of_grads(my_net, inputs, outputs, start, batch_size);


    // update gradients_W in the NEURAL_NET struct 
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			    my_net->gradients_W[i][j][k] = (1.0/batch_size)*b_grad->sum_dcdw_W[i][j][k];
			}
	    }        
	}

    double alpha = 0.1; // learning rate

    // apply mini batch SGD to weights
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			   my_net->weights_W[i][j][k] -= alpha*(1.0/batch_size)*b_grad->sum_dcdw_W[i][j][k];
			}
	    }               
	}

    dealloc_W(b_grad->sum_dcdw_W);

    // apply mini batch SGD to biases
    for (i=1;i<*size;i++){

    	for (j=0;j<network[i];j++){
    		    my_net->biases_N[i][j] -= alpha*(1.0/batch_size)*b_grad->sum_dcdb_N[i][j];
    	}
    }

    dealloc_N(b_grad->sum_dcdb_N);
    free(b_grad);
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses mini batch SGD with momentum.
void train_batch_momentum(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int start, int batch_size, double*** v_W, double** v_N){
    int i, j, k;

    struct BATCH_GRADIENTS* b_grad = sum_of_grads(my_net, inputs, outputs, start, batch_size);

    double beta = 0.9;   // exponential average hyperparameter
    double alpha = 0.1;  // learning rate


	// apply momentum to weights
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
				v_W[i][j][k] = beta*v_W[i][j][k] + (1.0-beta)*(1.0/batch_size)*b_grad->sum_dcdw_W[i][j][k];  // update the "velocity" of the gradient wrt the weights
			    my_net->weights_W[i][j][k] -= alpha*v_W[i][j][k];											 // update weights
			}
	    }               
	}

	dealloc_W(b_grad->sum_dcdw_W);

    // apply momentum to biases
    for (i=1;i<*size;i++){

    	for (j=0;j<network[i];j++){
    		v_N[i][j] = beta*v_N[i][j] + (1-beta)*(1.0/batch_size)*b_grad->sum_dcdb_N[i][j];	// compute the "velocity" of the gradients wrt the biases
    		my_net->biases_N[i][j] -= alpha*v_N[i][j];											// update biases
    	}
    }

    dealloc_N(b_grad->sum_dcdb_N);
    free(b_grad);
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses Adagrad optimizer.
void train_batch_adagrad(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int start, int batch_size, double*** a_W, double** a_N){
    int i, j, k;

    struct BATCH_GRADIENTS* b_grad = sum_of_grads(my_net, inputs, outputs, start, batch_size);

    // update gradients_W in the NEURAL_NET struct
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			    my_net->gradients_W[i][j][k] = (1.0/batch_size)*b_grad->sum_dcdw_W[i][j][k];
			}
	    }        
	}

    double alpha = 0.01;      // laerning rate 
    double eps = 0.0000001; // parameter to prevent divide by zero error.

    // apply adagrad to weights
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
				a_W[i][j][k] += pow((1.0/batch_size)*b_grad->sum_dcdw_W[i][j][k], 2);												// compute the accumulated squared gradients wrt the weights
			    my_net->weights_W[i][j][k] -= alpha*(1.0/sqrt(a_W[i][j][k] + eps))*(1.0/batch_size)*b_grad->sum_dcdw_W[i][j][k];	// update weights
			}
	    }               
	}

	dealloc_W(b_grad->sum_dcdw_W);

    // apply adagrad to biases
    for (i=1;i<*size;i++){

    	for (j=0;j<network[i];j++){
    		a_N[i][j] +=  pow((1.0/batch_size)*b_grad->sum_dcdb_N[i][j], 2);										// compute the accumulated squared gradients wrt the biases
    		my_net->biases_N[i][j] -= alpha*(1.0/sqrt(a_N[i][j] + eps))*(1.0/batch_size)*b_grad->sum_dcdb_N[i][j];  // update biases
    	}
    }

    dealloc_N(b_grad->sum_dcdb_N);
    free(b_grad);
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses RMSprop optimizer
void train_batch_RMSprop(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int start, int batch_size, double*** s_W, double** s_N){
    int i, j, k;

    struct BATCH_GRADIENTS* b_grad = sum_of_grads(my_net, inputs, outputs, start, batch_size);

    // update gradients_W in the NEURAL_NET struct
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			    my_net->gradients_W[i][j][k] = (1.0/batch_size)*b_grad->sum_dcdw_W[i][j][k];
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
				s_W[i][j][k] = beta*s_W[i][j][k] + (1.0 - beta)*pow((1.0/batch_size)*b_grad->sum_dcdw_W[i][j][k], 2); 				// update the exponential average of the squared gradients wrt the weights

			   	my_net->weights_W[i][j][k] -= alpha*(1.0/sqrt(s_W[i][j][k] + eps))*(1.0/batch_size)*b_grad->sum_dcdw_W[i][j][k] ;	// update the weights
			}
	    }               
	}

	dealloc_W(b_grad->sum_dcdw_W);

    // apply RMSprop to biases
    for (i=1;i<*size;i++){

    	for (j=0;j<network[i];j++){
    			s_N[i][j]  = beta*s_N[i][j] + (1.0 - beta)*pow((1.0/batch_size)*b_grad->sum_dcdb_N[i][j], 2);				// update the exponential average of the squared gradients wrt the biases
    		    my_net->biases_N[i][j] -= alpha*(1.0/sqrt(s_N[i][j] + eps))*(1.0/batch_size)*b_grad->sum_dcdb_N[i][j];		// update biases
    	}
    }

    dealloc_N(b_grad->sum_dcdb_N);
    free(b_grad);
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses RMSprop optimizer
void train_batch_adadelta(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int start, int batch_size, double*** s_W, double** s_N, double*** deltas_W, double** deltas_N, double*** D_W, double** D_N){
    int i, j, k;

    struct BATCH_GRADIENTS* b_grad = sum_of_grads(my_net, inputs, outputs, start, batch_size);

    // update gradients_W in the NEURAL_NET struct
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			    my_net->gradients_W[i][j][k] = (1.0/batch_size)*b_grad->sum_dcdw_W[i][j][k];
			}
	    }        
	}

    double beta = 0.95;       // exponential average hyperparameter
    double eps = 0.000001; // parameter to prevent divide by zero error.

   
	// apply adadelta to weights
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
				s_W[i][j][k] = beta*s_W[i][j][k] + (1.0 - beta)*pow((1.0/batch_size)*b_grad->sum_dcdw_W[i][j][k], 2);							// update the exponential average of the squared gradients wrt the weights

			    my_net->weights_W[i][j][k] -= (sqrt(D_W[i][j][k]+eps)/sqrt(s_W[i][j][k] + eps))*(1.0/batch_size)*b_grad->sum_dcdw_W[i][j][k] ;  // update the weights

			    deltas_W[i][j][k] = -(sqrt(D_W[i][j][k]+eps)/sqrt(s_W[i][j][k] + eps))*(1.0/batch_size)*b_grad->sum_dcdw_W[i][j][k];			// update the delta_weight for this iteration
			    D_W[i][j][k] = beta*D_W[i][j][k] + (1.0 - beta)*pow(deltas_W[i][j][k], 2);														// update the numerator that will multiply the gradient in the next iteration
			}
	    }               
	}

	dealloc_W(b_grad->sum_dcdw_W);

    // apply adadelta to biases
    for (i=1;i<*size;i++){

    	for (j=0;j<network[i];j++){
    			s_N[i][j]  = beta*s_N[i][j] + (1.0 - beta)*pow((1.0/batch_size)*b_grad->sum_dcdb_N[i][j], 2);						// update the exponential average of the squared gradients wrt the biases

    		    my_net->biases_N[i][j] -= (sqrt(D_N[i][j]+eps)/sqrt(s_N[i][j] + eps))*(1.0/batch_size)*b_grad->sum_dcdb_N[i][j];    // update biases

    		    deltas_N[i][j] = (sqrt(D_N[i][j]+eps)/sqrt(s_N[i][j] + eps))*(1.0/batch_size)*b_grad->sum_dcdb_N[i][j]; 			// update the delta_bias 
    		    D_N[i][j] = beta*D_N[i][j] + (1.0 -beta)*pow(deltas_N[i][j], 2);													// update the numerator that will multiply the gradient in the next iteration
    	}
    }

    dealloc_N(b_grad->sum_dcdb_N);
    free(b_grad);
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses Adam optimizer
void train_batch_adam(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int start, int batch_size, double*** m_W, double*** mhat_W, double** m_N, double** mhat_N,  double*** v_W, double*** vhat_W, double** v_N, double** vhat_N, int iter_no){
    int i, j, k;

    struct BATCH_GRADIENTS* b_grad = sum_of_grads(my_net, inputs, outputs, start, batch_size);

    // update gradients_W in the NEURAL_NET struct
	for (i=0;i<*size-1;i++){

		for (j=0;j<network[i];j++){

			for (k=0;k<network[i+1];k++){
			    my_net->gradients_W[i][j][k] = (1.0/batch_size)*b_grad->sum_dcdw_W[i][j][k];
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
			    m_W[i][j][k] = beta1*m_W[i][j][k] + (1.0 - beta1)*(1.0/batch_size)*b_grad->sum_dcdw_W[i][j][k];         // update 1st moment vector
			    mhat_W[i][j][k] = m_W[i][j][k]/(1.0 - pow(beta1, iter_no));                                             // fill in mhat

			    v_W[i][j][k] = beta2*v_W[i][j][k] + (1.0 - beta2)*pow((1.0/batch_size)*b_grad->sum_dcdw_W[i][j][k], 2); // update 2nd moment vector
			    vhat_W[i][j][k] = v_W[i][j][k]/(1.0 - pow(beta2, iter_no));												// fill in vhat

			    my_net->weights_W[i][j][k] -= alpha*(1.0/(sqrt(vhat_W[i][j][k]) + eps))*mhat_W[i][j][k];				// update weights
			}
	    }        
	}


	dealloc_W(b_grad->sum_dcdw_W);

	// apply adam algorithm to biases
    for (i=1;i<*size;i++){

    	for (j=0;j<network[i];j++){
    		    m_N[i][j]  = beta1*m_N[i][j] + (1.0 - beta1)*(1.0/batch_size)*b_grad->sum_dcdb_N[i][j];				// update 1st moment 
    		    mhat_N[i][j]  = m_N[i][j]/(1.0 - pow(beta1, iter_no));												// fill in mhat

    		    v_N[i][j]  = beta2*v_N[i][j] + (1.0 - beta2)*pow((1.0/batch_size)*b_grad->sum_dcdb_N[i][j], 2);		// update 2nd moment
    		    vhat_N[i][j]  = v_N[i][j]/(1.0 - pow(beta2, iter_no));												// fill in vhat

    		    my_net->biases_N[i][j] -= alpha*(1.0/(sqrt(vhat_N[i][j]) + eps))*(1.0/batch_size)*mhat_N[i][j];		// update biases
    	}
    }


    dealloc_N(b_grad->sum_dcdb_N);
    free(b_grad);
}

//---------------------------------------------------------------------------------------------