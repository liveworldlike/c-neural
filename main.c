#include <stdio.h>
#include "mnist.h"
#include "net.h"
#include "matrix.h"
#include <assert.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>

void get_image(Dataset *dataset, int index, Matrix *buffer) {
    assert(buffer->columns == dataset->image_len);
    assert(buffer->rows == 1);
    for (int i = 0; i < dataset->image_len; i++) {
        buffer->array[0][i] = dataset->images[index][i];
    }
}

void calculate_gradients(Matrix *pred, Matrix *temp, Matrix *gradients, int label, double *loss) {
    // Calculates cross-entropy gradients from logits
    // temp is used as a temporary storage for intermediate results
    double es = 0;
    for (int j = 0; j < 10; j++) {
        temp->array[0][j] = exp(pred->array[0][j]);
        es += temp->array[0][j];
    }
    double z = log(es) - pred->array[0][label];
    *loss = z;
    double common = -temp->array[0][label] * z / es / es;
    double labelled = z  / es;
    for (int j = 0; j < 10; j++) {
        gradients->array[0][j] = common * temp->array[0][j];
    }
    gradients->array[0][label] += labelled * temp->array[0][label];
}

int main() {
    Dataset *train_dataset = load_dataset("dataset/train-images-idx3-ubyte", "dataset/train-labels-idx1-ubyte");
    int layers[] = {28 * 28 * 1, 128, 10};
    Net *net = create_net(2, layers);
    Matrix *x = create_matrix(1, train_dataset->image_len);
    Matrix *pred = create_matrix(1, 10);
    Matrix *gradients = create_matrix(1, 10);
    Matrix *temp = create_matrix(1, 10);
    double eta = 1e-2;
    for (int epoch = 0; epoch < 10; epoch++) {
        double loss_sum = 0;
        for (int i = 0; i < train_dataset->dataset_len; i++) {
            double loss;
            get_image(train_dataset, i, x);
            forward_net(net, x, pred);
            int label = train_dataset->labels[i];
            calculate_gradients(pred, temp, gradients, label, &loss);
            backward_net(net, x, gradients);
            loss_sum += loss;
            if (i && i % 50 == 0) {
                update_net(net, eta);
            }
        }
        eta *= .7;    
        printf("Epoch %d finished.\n", epoch);
        printf("Mean loss: %.4f\n", loss_sum / train_dataset->dataset_len);
    }
    free_dataset(train_dataset);
    Dataset *test_dataset = load_dataset("dataset/t10k-images-idx3-ubyte", "dataset/t10k-labels-idx1-ubyte");
    int correct = 0;
    for (int i = 0; i < test_dataset->dataset_len; i++) {
        get_image(test_dataset, i, x);
        forward_net(net, x, pred);
        double max = -1000;
        int plabel = 0;
        for (int j = 0; j < 10; j++){
            if (pred->array[0][j] > max) {
                max = pred->array[0][j];
                plabel = j;
            }
        }
        int label = test_dataset->labels[i];
        if (plabel == label) {
            correct++;
        }
    }
    free_dataset(test_dataset);
    free_net(net);
    free_matrix(x);
    free_matrix(temp);
    free_matrix(gradients);
    free_matrix(pred);
    printf("Accuracy: %.2f%%\n", (double)correct / test_dataset->dataset_len * 100);
}