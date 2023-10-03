#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mnist.h"

int read_int(FILE *file) {
    int result = 0;
    for (int i = 24; i >= 0; i -= 8) {
        byte temp = getc(file);
        result |= temp << i;
    } 
    return result;
}

byte read_byte(FILE *file) {
    return getc(file);
}

double preprocess_pixel(byte pixel) {
    return (double)pixel / 127.5 - 1;
}

Dataset *load_dataset(const char *images_filename, const char *labels_filename) {
    void *space = malloc(sizeof(Dataset));
    assert(space != NULL);
    Dataset *dataset = (Dataset *)space;

    FILE *images_file = fopen(images_filename, "rb");
    int images_header = read_int(images_file);
    assert(images_header == 0x0803);
    int images_count = read_int(images_file);
    int image_rows = read_int(images_file);
    int image_columns = read_int(images_file);
    int image_len = image_rows * image_columns;
    dataset->image_len = image_len;

    void *imspace = malloc(images_count * sizeof(byte *));
    assert(imspace != NULL);
    dataset->images = (double **)imspace;

    for (int i = 0; i < images_count; i++) {
        dataset->images[i] = (double *)malloc(image_len * sizeof(double)); 
        assert(dataset->images[i] != NULL);
        for (int j = 0; j < image_len; j++) {
            dataset->images[i][j] = preprocess_pixel(read_byte(images_file));
        }
    }
    fclose(images_file);

    FILE *labels_file = fopen(labels_filename, "rb");
    int labels_header = read_int(labels_file);
    assert(labels_header == 0x0801);
    int labels_count = read_int(labels_file);

    void *lbspace = malloc(labels_count * sizeof(int));
    assert(lbspace != NULL);
    dataset->labels = (int *)lbspace;

    for (int i = 0; i < labels_count; i++) {
        dataset->labels[i] = (int)read_byte(labels_file);
    }
    fclose(labels_file);
    assert(labels_count == images_count);
    dataset->dataset_len = images_count;
    return dataset;
}

void free_dataset(Dataset *dataset) {
    for (int i = 0; i < dataset->dataset_len; i++){
        free(dataset->images[i]);
    }
    free(dataset->images);
    free(dataset->labels);
}