#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "nn_functions.h"


#define N_TEST 1000
#define N_TRAIN 20000

// declare a struct for mnist data
struct DATA{
    double** train_images;
    double** train_labels;
    double** test_images;
    int* test_labels;
};

struct DATA* load_mnist(){
    int train_label[N_TRAIN];

    struct DATA* my_data = malloc(sizeof*my_data);

    double** train_image_ptr = (double**)calloc(N_TRAIN, sizeof(double*));
    for(int i =0;i<N_TRAIN;i++){
        train_image_ptr[i] = (double*)calloc(784, sizeof(double));
    }

    double** train_label_ptr = (double**)calloc(N_TRAIN, sizeof(double*));
    for(int i=0;i<N_TRAIN;i++){
        train_label_ptr[i] = (double*)calloc(10, sizeof(double));
    }

    double** test_image_ptr = (double**)calloc(N_TRAIN, sizeof(double*));
    for(int i =0;i<N_TRAIN;i++){
        test_image_ptr[i] = (double*)calloc(784, sizeof(double));
    }

    int* test_label_ptr = (int*)calloc(N_TEST, sizeof(int));


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
        test_label_ptr[row] = atoi(tok);


        tok = strtok(NULL, ",");
        while (tok != NULL){
            test_image_ptr[row][col] = strtod(tok, &ptr)/255.0;
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


        tok = strtok(NULL, ",");
        while (tok != NULL){
            train_image_ptr[row][col] = strtod(tok, &ptr)/255.0;
            col +=1;
            tok = strtok(NULL, ",");
    
        }
    }

    fclose(fp);

    int i, j;

    for (i=0;i<N_TRAIN;i++){
        for (j=0;j<10;j++){
            if (j==train_label[i]){
                train_label_ptr[i][j] = 1;
            }
            else{
                train_label_ptr[i][j] = 0;
            }
        }
    }

    my_data->test_images = test_image_ptr;
    my_data->test_labels = test_label_ptr;
    my_data->train_images = train_image_ptr;
    my_data->train_labels = train_label_ptr;
    return my_data;
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

void test_network(struct NEURAL_NET* my_net, struct DATA* my_data){
    int i,j;

    int errors = 0;

    for (i=0;i<N_TEST;i++){
        double input[784];
        for (j=0;j<784;j++){
            input[j] = my_data->test_images[i][j];
        }
        feed_fwd(input, my_net);

        int prediction = largest(my_net->activations_N[3], 10);

        if (prediction != my_data->test_labels[i]){
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