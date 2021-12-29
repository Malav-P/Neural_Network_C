#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "nn_header.h"

#define BUFFER_SIZE 10000

#define no_of_test_pts 1000
#define no_of_train_pts 20000

int train_label[no_of_train_pts];
int test_label[no_of_test_pts];
double train_image[no_of_train_pts][784];
double test_image[no_of_test_pts][784];

double label_train[no_of_train_pts][10];
double label_test[no_of_test_pts][10];

void load_mnist(){
    FILE *fp;
    char buf[BUFFER_SIZE];
    double ret;
    int row = 0;
    int col=0;

    fp = fopen("mnist_test.csv","r");

    if(fp == NULL) {
      perror("Error opening file");
      exit(1);
   }

    while(row<no_of_test_pts){

    col = 0;

   fgets(buf, sizeof(buf), fp);

   char* ptr;
   char *tok = strtok(buf, ",");


   test_label[row] = atoi(tok);
   col+=1;
   tok = strtok(NULL, ",");

   while (col<785){
    test_image[row][col] = strtod(tok, &ptr);
    col +=1;
    tok = strtok(NULL, ",");
    
   }
   
    row+=1;
    }


    row = 0;

    fp = fopen("mnist_train.csv","r");

    if(fp == NULL) {
      perror("Error opening file");
      exit(1);
   }

   while(row < no_of_train_pts){

    col = 0;

   fgets(buf, sizeof(buf), fp);

   char* ptr;
   char *tok = strtok(buf, ",");

   
   train_label[row] = atoi(tok);
   col+=1;
   tok = strtok(NULL, ",");

   while (col<785){
    train_image[row][col] = strtod(tok, &ptr);
    col +=1;
    tok = strtok(NULL, ",");
    
   }
   
    row+=1;
    }
}

void process_labels(){
    int i, j;

    for (i=0;i<no_of_train_pts;i++){
        for (j=0;j<10;j++){
            if (j==train_label[i]){
                label_train[i][j] = 1;
            }
            else{
                label_train[i][j] = 0;
            }
        }
    }

    for (i=0;i<no_of_test_pts;i++){
        for (j=0;j<10;j++){
            if (j==test_label[i]){
                label_test[i][j] = 1;
            }
            else{
                label_test[i][j] = 0;
            }
        }
    }
}

int largest(double* arr, int n){
    int i;
    
    // Initialize maximum element
    int max = 0;
 
    // Traverse array elements from second and
    // compare every element with current max 
    for (i = 1; i < n; i++)
        if (arr[i] > arr[max]){
            max = i;
        }
 
    return max;
}

void test_network(struct NEURAL_NET* my_net){
    int i,j;

    int errors = 0;

    for (i=0;i<no_of_test_pts;i++){
        double input[784];
        for (j=0;j<784;j++){
            input[j] = test_image[i][j];
        }
        feed_fwd(input, my_net);

        int prediction = largest(my_net->activations[*size-1], network[*size-1]);

        if (prediction != test_label[i]){
            errors+=1;
        }


    }

    printf("number of errors: %d\n", errors);

    double error_rate = (double) 100*errors/no_of_test_pts;

    printf("error rate: %f %%\n\n", error_rate);
}



int main(){
    int i,j;
    int batch_size = 50;
    int epochs = 3;

    load_mnist();
    process_labels();

    printf("processing data complete, now training network...\n\n");

    int my_n[] = {784, 100, 100, 10};
    int s = sizeof(my_n)/sizeof(int);

    struct NEURAL_NET* my_net = initialize_network(my_n, &s);

    train_network(my_net, train_image, label_train, no_of_train_pts, batch_size, epochs, 3);
    test_network(my_net);

}