//*********************************************************************************************

// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses mini batch SGD.
void train_batch(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int start, int batch_size);

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses mini batch SGD with momentum.
void train_batch_momentum(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int start, int batch_size, double*** v_W, double** v_N);

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses Adagrad optimizer.
void train_batch_adagrad(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int start, int batch_size, double*** a_W, double** a_N);

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses RMSprop optimizer
void train_batch_RMSprop(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int start, int batch_size, double*** s_W, double** s_N);

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

// train the network on a subset window of the training data. This subset window is defined as the (start)th input to the (start+batch_size)th input. This uses RMSprop optimizer
void train_batch_adadelta(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int start, int batch_size, double*** s_W, double** s_N, double*** deltas_W, double** deltas_N, double*** D_W, double** D_N);

//---------------------------------------------------------------------------------------------
//*********************************************************************************************

void train_batch_adam(struct NEURAL_NET* my_net, double inputs[][network[0]], double outputs[][network[*size-1]], int start, int batch_size, double*** m_W, double*** mhat_W, double** m_N, double** mhat_N,  double*** v_W, double*** vhat_W, double** v_N, double** vhat_N, int iter_no);

//---------------------------------------------------------------------------------------------