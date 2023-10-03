#pragma once
#include "matrix.h"
struct _Net {
  int layers_count;
  int *layer_sizes;
  Matrix **W;
  Matrix **B;
  Matrix **Wgrad;
  Matrix **Bgrad;
};

typedef struct _Net Net;

Net *create_net(int layers_count, int *layer_sizes);
void init_net(Net *net, int layers_count, int *layer_sizes);
void init_weights(Net *net);
void forward_net(Net *net, Matrix *inputs, Matrix *outputs);
void update_net(Net *net, double c);
void backward_net(Net *net, Matrix *inputs, Matrix *gradients);
void zero_grad(Net *net);
void free_net(Net *net);
void free_net_weights(Net *net);

