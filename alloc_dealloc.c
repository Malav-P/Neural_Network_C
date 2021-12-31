#include <math.h>
#include <stdlib.h>
#include "structs_and_globals.h"
//**********************************************************

// generate a random value from a uniform distribution between 0 and 1
double rand_gen() {
   // return a uniformly distributed random value
   return ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
}

//----------------------------------------------------------
//*********************************************************

// generate a random value from the standard normal distribution
double normalRandom() {
   // return a normally distributed random value
   double v1=rand_gen();
   double v2=rand_gen();
   return cos(2*3.14*v2)*sqrt(-2.*log(v1));
}

//----------------------------------------------------------
//*********************************************************

// returns a random number from the range (min, max)
double randfrom(double min, double max){
    double range = (max - min); 
    double div = RAND_MAX / range;

    return min + (rand() / div);
}

//---------------------------------------------------------
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

// allocate memory for a structure of type N. Initialize all entries to 1.0.
// ptr_N[i][j] is the entry for the jth neuron in the ith layer.
double** one_init_N(){
   int i, j;
   double** ptr_N = alloc_N();

   for (i=0;i<*size;i++){

      for (j=0;j<network[i];j++){
         ptr_N[i][j] = 1.0;
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
//*********************************************************************************************

// allocate memory for a structure of type N. Initialize all entries according to He initialization procedure
// ptr_N[i][j] is the entry for the jth neuron in the ith layer.
double** he_init_N(){
   int i, j;
   double** ptr_N = alloc_N();

   for (i=0;i<*size;i++){

      for (j=0;j<network[i];j++){
         ptr_N[i][j] = normalRandom()*sqrt(2.0/network[i]);
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

// allocate memory for a structure of type W and initialize all entries to 1.0.
// ptr_W[i][j][k] is the weight connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
double*** one_init_W(){
   int i, j, k;
   double*** ptr_W = alloc_W();

   for (i=0;i<*size-1;i++){

      for (j=0;j<network[i];j++){

         for (k=0;k<network[i+1];k++){
            ptr_W[i][j][k] = 1.0;
         }
      }
   }

   return ptr_W;
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type W and intialize each entry with a random value between -0.01 and 0.01.
// ptr_W[i][j][k] is the entry connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
double*** r_init_W(){
   int i, j, k;
   double*** ptr_W = alloc_W();

   for (i=0;i<*size-1;i++){

      for (j=0;j<network[i];j++){

         for (k=0;k<network[i+1];k++){
            ptr_W[i][j][k] = randfrom(-0.01,0.01);
         }
      }
   }

   return ptr_W;
}

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// allocate memory for a structure of type W and intialize each entry according to He initialization procedure
// ptr_W[i][j][k] is the entry connecting the kth neuron in the (i+1)th layer to the jth neuron in the ith layer.
double*** he_init_W(){
   int i, j, k;
   double*** ptr_W = alloc_W();

   for (i=0;i<*size-1;i++){

      for (j=0;j<network[i];j++){

         for (k=0;k<network[i+1];k++){
            ptr_W[i][j][k] = normalRandom()*sqrt(2.0/network[i]);
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