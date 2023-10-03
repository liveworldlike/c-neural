#include <stdio.h>
#include "net.h"
#include <assert.h>
#include <stdlib.h>
#include <math.h>

Net *create_net(int layers_count, int *layer_sizes) {
  void *space = malloc(sizeof(Net));
  assert(space != NULL);
  Net *net = (Net *)space;
  init_net(net, layers_count, layer_sizes);
  return net;
}

void init_net(Net *net, int layers_count, int *layer_sizes) {
  net->layers_count = layers_count;
  assert(net->layers_count == layers_count);
  void *layerspace = malloc((layers_count + 1) * sizeof(int));
  assert(layerspace != NULL);
  net->layer_sizes = layerspace;
  for (int i = 0; i <= layers_count; i++) {
    net->layer_sizes[i] = layer_sizes[i];
  }

  void *Wspace = malloc(layers_count * sizeof(Matrix *));
  assert(Wspace != NULL);
  net->W = (Matrix **)Wspace;
  
  void *Wgradspace = malloc(layers_count * sizeof(Matrix *));
  assert(Wgradspace != NULL);
  net->Wgrad = (Matrix **)Wgradspace;

  void *Bspace = malloc(layers_count * sizeof(Matrix *));
  assert(Bspace != NULL);
  net->B = (Matrix **)Bspace;

  void *Bgradspace = malloc(layers_count * sizeof(Matrix *));
  assert(Bgradspace != NULL);
  net->Bgrad = (Matrix **)Bgradspace;

  for (int i = 0; i < layers_count; i++){
    net->W[i] = create_matrix(layer_sizes[i], layer_sizes[i + 1]);
    net->Wgrad[i] = create_matrix(layer_sizes[i], layer_sizes[i + 1]);
    net->B[i] = create_matrix(1, layer_sizes[i + 1]);
    net->Bgrad[i] = create_matrix(1, layer_sizes[i + 1]);
  }
  init_weights(net);
}

double uniform_dist() {
  return (double)rand() / (double)RAND_MAX;
}

double normal_dist() {
  // Marsaglia polar method - https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution
  while (1) {
    double u = uniform_dist();
    double v = uniform_dist();
    double s = u * u + v * v;
    if (s >= 1) {
      continue;
    }
    double x = u * sqrt(-2 * log(s) / s);
    return x;
  }
}

void init_weights(Net *net) {
  srand(42);
  for (int i = 0; i < net->layers_count; i++) {
    double std = 1. / net->W[i]->rows;
    double K = sqrt(std);
    for (int j = 0; j < net->W[i]->rows; j++) {
      for (int k = 0; k < net->W[i]->columns; k++) {
        net->W[i]->array[j][k] = uniform_dist() * 2 * K - K;
      }
    }
    for (int j = 0; j < net->B[i]->rows; j++) {
      for (int k = 0; k < net->B[i]->columns; k++) {
        net->B[i]->array[j][k] = uniform_dist() * 2 * K - K;
      }
    }
  }
}

void apply_fc(Matrix *x, Matrix *W, Matrix *y) {
  dot_matrix(y, x, W);
}

void apply_bias(Matrix *x, Matrix *B) {
  elementwise_add_matrix(x, x, B);
}

void ReLU(Matrix *x) {
  for (int i = 0; i < x->rows; i++) {
    for (int j = 0; j < x->columns; j++) {
      if (x->array[i][j] < 0) {
        x->array[i][j] *= .1;
      }
    }
  }
}

void forward_net(Net *net, Matrix *inputs, Matrix *outputs) {
  Matrix *x = create_matrix(inputs->rows, inputs->columns);
  copy_matrix(x, inputs);
  Matrix *y;
  for (int i = 0; i < net->layers_count; i++) {
    y = create_matrix(x->rows, net->W[i]->columns);
    apply_fc(x, net->W[i], y);
    free_matrix(x);
    x = y;
    apply_bias(x, net->B[i]);
    if (i < net->layers_count - 1) {
      ReLU(x);
    }
  }
  copy_matrix(outputs, x);
  free_matrix(x);
}

void update_net(Net *net, double c) {
  for (int i = 0; i < net->layers_count; i++) {
    multiply_scalar_matrix(net->Wgrad[i], net->Wgrad[i], c);
    multiply_scalar_matrix(net->Bgrad[i], net->Bgrad[i], c);
    elementwise_add_matrix(net->W[i], net->W[i], net->Wgrad[i]);
    elementwise_add_matrix(net->B[i], net->B[i], net->Bgrad[i]);
    multiply_scalar_matrix(net->Wgrad[i], net->Wgrad[i], 1 / c);
    multiply_scalar_matrix(net->Bgrad[i], net->Bgrad[i], 1 / c);
  }
}

void ReLUbackward(Matrix *activations, Matrix *gradients) {
  return;
  for (int i = 0; i < activations->rows; i++) {
    for (int j = 0; j < activations->columns; j++) {
      if (activations->array[i][j] < 0) {
        gradients->array[i][j] *= .1;
      }
    }
  }
}

void backward_net(Net *net, Matrix *inputs, Matrix *gradients) {
  void *wspace = malloc(net->layers_count * sizeof(Matrix));
  assert(wspace != NULL);
  Matrix **wActivations = (Matrix **)wspace;

  void *bspace = malloc(net->layers_count * sizeof(Matrix));
  assert(bspace != NULL);
  Matrix **bActivations = (Matrix **)bspace;

  void *aspace = malloc(net->layers_count * sizeof(Matrix));
  assert(aspace != NULL);
  Matrix **aActivations = (Matrix **)aspace;

  int batch_size = inputs->rows;
  for(int i = 0; i < net->layers_count; i++) {
    wActivations[i] = create_matrix(batch_size, net->layer_sizes[i]);
    bActivations[i] = create_matrix(batch_size, net->layer_sizes[i + 1]);
    aActivations[i] = create_matrix(batch_size, net->layer_sizes[i + 1]);
  }
  Matrix *x = create_matrix(inputs->rows, inputs->columns);
  copy_matrix(x, inputs);
  Matrix *y;

  for (int i = 0; i < net->layers_count; i++) {
    y = create_matrix(x->rows, net->W[i]->columns);
    copy_matrix(wActivations[i], x);
    apply_fc(x, net->W[i], y);
    free_matrix(x);
    x = y;
    copy_matrix(bActivations[i], x);
    apply_bias(x, net->B[i]);
    copy_matrix(aActivations[i], x);
    if (i < net->layers_count - 1) {
      ReLU(x);
    }
  }
  free_matrix(x);

  x = create_matrix(gradients->rows, gradients->columns);
  copy_matrix(x, gradients);
  for (int i = net->layers_count - 1; i >= 0; i--){
    if (i < net->layers_count - 1) {
      ReLUbackward(aActivations[i], x);
    }
    copy_matrix(net->Bgrad[i], x);
    Matrix *AT = create_matrix(wActivations[i]->columns, wActivations[i]->rows);
    transpose_matrix(AT, wActivations[i]);
    dot_matrix(net->Wgrad[i], AT, x);
    free_matrix(AT);
    Matrix *WT = create_matrix(net->W[i]->columns, net->W[i]->rows);
    transpose_matrix(WT, net->W[i]);
    y = create_matrix(x->rows, WT->columns);
    dot_matrix(y, x, WT);
    free_matrix(WT);
    free_matrix(x);
    x = y;
  }
  for(int i = 0; i < net->layers_count; i++) {
    free_matrix(wActivations[i]);
    free_matrix(bActivations[i]);
    free_matrix(aActivations[i]);
  }
  free(wActivations);
  free(bActivations);
  free(aActivations);
  free_matrix(x);
}

void zero_grad(Net *net) {
  for (int i = 0; i < net->layers_count; i++) {
    for (int j = 0; j < net->Wgrad[i]->rows; j++) {
      for (int k = 0; k < net->Wgrad[i]->columns; k++) {
        net->Wgrad[i]->array[j][k] = 0;
      }
    }
    for (int j = 0; j < net->Bgrad[i]->rows; j++) {
      for (int k = 0; k < net->Bgrad[i]->columns; k++) {
        net->Bgrad[i]->array[j][k] = 0;
      }
    }
  }
}

void free_net(Net *net) {
  free_net_weights(net);
  free(net->layer_sizes);
  free(net);
}

void free_net_weights(Net *net) {
  for (int i = 0; i < net->layers_count; i++) {
    free_matrix(net->W[i]);
    free_matrix(net->Wgrad[i]);
    free_matrix(net->B[i]);
    free_matrix(net->Bgrad[i]);
  }
  free(net->B);
  free(net->Bgrad);
  free(net->W);
  free(net->Wgrad);
}