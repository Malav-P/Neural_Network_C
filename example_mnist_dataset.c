#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "nn_functions.h"


#define N_TEST 1000
#define N_TRAIN 5000

int train_label[N_TRAIN];
int test_label[N_TEST];
double train_image[N_TRAIN][784];
double test_image[N_TEST][784];

double train_lbl_dbl[N_TRAIN][10];
double test_lbl_dbl[N_TEST][10];

void load_mnist(){
    FILE *fp;

    int row, col;

    char buf[10000];
    char* ptr;
    char* tok;

    fp = fopen("mnist_test.csv","r");

    if(fp == NULL){
      perror("Error opening file");
      exit(1);
    }

    for(row = 0;row<N_TEST;row++){

        col = 0;

        fgets(buf, sizeof(buf), fp);

        tok = strtok(buf, ",");
        test_label[row] = atoi(tok);

        col+=1;
        tok = strtok(NULL, ",");
        while (tok != NULL){
            test_image[row][col] = strtod(tok, &ptr)/255.0;
            col +=1;
            tok = strtok(NULL, ",");
        }
    }

    fclose(fp);


    fp = fopen("mnist_train.csv","r");

    if(fp == NULL) {
      perror("Error opening file");
      exit(1);
    }
    
    for(row=0;row < N_TRAIN;row++){

        col = 0;

        fgets(buf, sizeof(buf), fp);

        tok = strtok(buf, ",");
        train_label[row] = atoi(tok);

        col+=1;
        tok = strtok(NULL, ",");
        while (tok != NULL){
            train_image[row][col] = strtod(tok, &ptr)/255.0;
            col +=1;
            tok = strtok(NULL, ",");
    
        }
    }

    fclose(fp);

    int i, j;

    for (i=0;i<N_TRAIN;i++){
        for (j=0;j<10;j++){
            if (j==train_label[i]){
                train_lbl_dbl[i][j] = 1;
            }
            else{
                train_lbl_dbl[i][j] = 0;
            }
        }
    }

    for (i=0;i<N_TEST;i++){
        for (j=0;j<10;j++){
            if (j==test_label[i]){
                test_lbl_dbl[i][j] = 1;
            }
            else{
                test_lbl_dbl[i][j] = 0;
            }
        }
    }
    return;
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

    for (i=0;i<N_TEST;i++){
        double input[784];
        for (j=0;j<784;j++){
            input[j] = test_image[i][j];
        }
        feed_fwd(input, my_net);

        int prediction = largest(my_net->activations_N[3], 10);

        if (prediction != test_label[i]){
            errors+=1;
        }
    }

    printf("number of errors: %d\n", errors);

    double error_rate = (double) 100*errors/N_TEST;

    printf("error rate: %f %%\n\n", error_rate);
}


int main(){
    int batch_size = 50;
    int epochs = 3;
    int optimizer = 3;

    load_mnist();

    int my_n[] = {784, 100, 100, 10};
    int s = sizeof(my_n)/sizeof(int);

    struct NEURAL_NET* my_net = initialize_network(my_n, &s);

    train_network(my_net, train_image, train_lbl_dbl, N_TRAIN, batch_size, epochs, optimizer);
    test_network(my_net);

    export_weights(my_net, "weights.txt");
    export_biases(my_net, "biases.txt");
}