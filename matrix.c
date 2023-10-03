#include "matrix.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void init_matrix(Matrix *matrix, int rows, int columns) {
  matrix->rows = rows;
  matrix->columns = columns;
  void *array_space = malloc(rows * sizeof(double *));
  assert(array_space != NULL);
  matrix->array = (double **)array_space;
  for (int i = 0; i < rows; i++) {
    void *row_space = malloc(columns * sizeof(double));
    assert(row_space != NULL);
    matrix->array[i] = row_space;
  }
}

void init_matrix_with_initial_value(Matrix *matrix, int rows, int columns,
                                    double init) {
  init_matrix(matrix, rows, columns);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      matrix->array[i][j] = init;
    }
  }
}

double *index_matrix(Matrix *matrix, int row, int column) {
  if (row < 0 || row >= matrix->rows) {
    return NULL;
  }
  if (column < 0 || column >= matrix->columns) {
    return NULL;
  }
  return matrix->array[row] + column;
}

void elementwise_add_matrix(Matrix *target, Matrix *a, Matrix *b) {
  assert(a->rows == b->rows && a->rows == target->rows);
  assert(a->columns == b->columns && a->columns == target->columns);
  for (int i = 0; i < target->rows; i++) {
    for (int j = 0; j < target->columns; j++) {
      target->array[i][j] = a->array[i][j] + b->array[i][j];
    }
  }
}

void add_scalar_matrix(Matrix *target, Matrix *a, double b) {
  assert(a->rows == target->rows);
  assert(a->columns == target->columns);
  for (int i = 0; i < target->rows; i++) {
    for (int j = 0; j < target->columns; j++) {
      target->array[i][j] = a->array[i][j] + b;
    }
  }
}

void elementwise_product_matrix(Matrix *target, Matrix *a, Matrix *b) {
  assert(a->rows == b->rows && a->rows == target->rows);
  assert(a->columns == b->columns && a->columns == target->columns);
  for (int i = 0; i < target->rows; i++) {
    for (int j = 0; j < target->columns; j++) {
      target->array[i][j] = a->array[i][j] * b->array[i][j];
    }
  }
}

void multiply_scalar_matrix(Matrix *target, Matrix *a, double b) {
  assert(a->rows == target->rows);
  assert(a->columns == target->columns);
  for (int i = 0; i < target->rows; i++) {
    for (int j = 0; j < target->columns; j++) {
      target->array[i][j] = a->array[i][j] * b;
    }
  }
}

void dot_matrix(Matrix *target, Matrix *a, Matrix *b) {
  assert(target->rows == a->rows);
  assert(target->columns == b->columns);
  assert(a->columns == b->rows);
  for (int i = 0; i < target->rows; i++) {
    for (int j = 0; j < target->columns; j++) {
      target->array[i][j] = 0;
      for (int k = 0; k < a->columns; k++) {
        target->array[i][j] += a->array[i][k] * b->array[k][j];
      }
    }
  }
}

void print_matrix(Matrix *matrix) {
  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->columns; j++) {
      printf("%.2f ", matrix->array[i][j]);
    }
    printf("\n");
  }
}

void free_matrix_array(Matrix *matrix) {
  for (int i = 0; i < matrix->rows; i++) {
    free(matrix->array[i]);
  }
  free(matrix->array);
}

void free_matrix(Matrix *matrix) {
  free_matrix_array(matrix);
  free(matrix);
}

Matrix *create_matrix(int rows, int columns) {
  void *space = malloc(sizeof(Matrix));
  assert(space != NULL);
  Matrix *result = (Matrix *)space;
  init_matrix(result, rows, columns);
  return result;
}

Matrix *create_matrix_with_initial_value(int rows, int columns, double init) {
  void *space = malloc(sizeof(Matrix));
  assert(space != NULL);
  Matrix *result = (Matrix *)space;
  init_matrix_with_initial_value(result, rows, columns, init);
  return result;
}

void copy_matrix(Matrix *dst, Matrix *src) {
  assert(dst->rows == src->rows);
  assert(dst->columns == src->columns);
  for (int i = 0; i < dst->rows; i++) {
    for (int j = 0; j < dst->columns; j++) {
      dst->array[i][j] = src->array[i][j];
    }
  }
}

void transpose_matrix(Matrix *dst, Matrix *src) {
  assert(dst->columns == src->rows);
  assert(dst->rows == src->columns);
  for (int i = 0; i < dst->rows; i++) {
    for (int j = 0; j < dst->columns; j++) {
      dst->array[i][j] = src->array[j][i];
    }
  }
}