#pragma once
struct _Matrix {
  int rows;
  int columns;
  double **array;
};

typedef struct _Matrix Matrix;

void init_matrix(Matrix *matrix, int rows, int columns);
void init_matrix_with_initial_value(Matrix *matrix, int rows, int columns,
                                    double init);
Matrix *create_matrix(int rows, int columns);
Matrix *create_matrix_with_initial_value(int rows, int columns, double init);
double *index_matrix(Matrix *matrix, int row, int column);
void elementwise_add_matrix(Matrix *target, Matrix *a, Matrix *b);
void add_scalar_matrix(Matrix *target, Matrix *a, double b);
void elementwise_product_matrix(Matrix *target, Matrix *a, Matrix *b);
void multiply_scalar_matrix(Matrix *target, Matrix *a, double b);
void dot_matrix(Matrix *target, Matrix *a, Matrix *b);
void print_matrix(Matrix *matrix);
void free_matrix(Matrix *matrix);
void free_matrix_array(Matrix *matrix);
void copy_matrix(Matrix *dst, Matrix *src);
void transpose_matrix(Matrix *dst, Matrix *src);
