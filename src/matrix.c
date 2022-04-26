#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * Assume `row` and `col` are valid. 
 * Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    // gets an element from the matrix data at the specified row and column (zero-indexed)
    int index = (mat->cols) * row + col;
    return mat->data[index];
}

/*
 * Sets the value at the given row and column to val. 
 * Assume `row` and `col` are valid. 
 * Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    // puts the given value in the matrix data at the specified row and column
    int index = (mat->cols) * row + col;
    mat->data[index] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. 
 * Allocates memory for the data array and initialize all entries to be zeros. 
 * Sets `parent` to NULL to indicate that this matrix is not a slice. 
 * Sets `ref_cnt` to 1.
 * Returns -1 if either `rows` or `cols` or both have invalid values. 
 * Returns -2 if any call to allocate memory in this function fails.
 * Returns 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // allocates space for a new matrix struct with the provided number of rows and columns
    if (!(rows > 0 && cols > 0)) {
        return -1;
    }
    struct matrix *new_matrix = (matrix *) malloc(sizeof(struct matrix));
    if (new_matrix == NULL) {
        return -2;
    }
    new_matrix->data = (double*) calloc(rows * cols, sizeof(double));
    if (new_matrix->data == NULL) {
        return -2;
    }
    new_matrix->rows = rows;
    new_matrix->cols = cols;
    new_matrix->parent = NULL;
    new_matrix->ref_cnt = 1;
    *mat = new_matrix;
    return 0;
}

/*
 * Deallocates space of the matrix mat. 
 */
void deallocate_matrix(matrix *mat) {
    // frees a matrix struct and, if no other structs are pointing at the data, frees the data as well
    if (mat == NULL) {
        return;
    }
    if (mat->parent == NULL) {
        mat->ref_cnt -= 1;
        if (mat->ref_cnt == 0) {
            free(mat->data);
            free(mat);
            return;
        } else {
            deallocate_matrix(mat->parent);
            free(mat);
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data points to the `offset`th entry of `from`'s data for the data field. 
 * Sets `parent` to `from` to indicate this matrix is a slice of `from` 
 * and incrementts the reference counter for `from`. 
 * Returns -1 if either `rows` or `cols` or both have invalid values. 
 * Returns -2 if any call to allocate memory in this function fails.
 * Returns 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // allocates space for a matrix struct created by slicing an existing matrix struct
    if (!(rows > 0 && cols > 0)) {
        return -1;
    }
    struct matrix *new_matrix = (matrix *) malloc(sizeof(struct matrix));
    if (new_matrix == NULL) {
        return -2;
    }
    new_matrix->data = from->data + offset;
    new_matrix->rows = rows;
    new_matrix->cols = cols;
    new_matrix->parent = from;
    from->ref_cnt += 1;
    *mat = new_matrix;
    return 0;
}

/*
 * Sets all entries in mat to val. 
 * Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    int row = mat->rows;
    int col = mat->cols;
    __m256d fill_vector = _mm256_set1_pd(val);
    #pragma omp parallel for
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col / 16 * 16; j += 16) {
            _mm256_storeu_pd(&mat->data[col * i + j], fill_vector);
            _mm256_storeu_pd(&mat->data[col * i + j + 4], fill_vector);
            _mm256_storeu_pd(&mat->data[col * i + j + 8], fill_vector);
            _mm256_storeu_pd(&mat->data[col * i + j + 12], fill_vector);
        }
        for (int j = col / 16 * 16; j < col; j++) {
            mat->data[col * i + j] = val;
        }
    }
    /*
    #pragma omp parallel for
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col / 4 * 4; j += 4) {
            _mm256_storeu_pd(&mat->data[col * i + j], fill_vector);
        }
        for (int j = col / 4 * 4; j < col; j++) {
            mat->data[col * i + j] = val;
        }
    }
    */
    return;
}

/*
 * Stores the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    int row = mat->rows;
    int col = mat->cols;
    #pragma omp parallel for
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col / 16 * 16; j += 16) {
            __m256d sub_vector1 = _mm256_set1_pd(0.0);
            __m256d sub_vector2 = _mm256_set1_pd(0.0);
            __m256d sub_vector3 = _mm256_set1_pd(0.0);
            __m256d sub_vector4 = _mm256_set1_pd(0.0);
            __m256d orig_vector1 = _mm256_loadu_pd(&mat->data[col * i + j]);
            __m256d orig_vector2 = _mm256_loadu_pd(&mat->data[col * i + j + 4]);
            __m256d orig_vector3 = _mm256_loadu_pd(&mat->data[col * i + j] + 8);
            __m256d orig_vector4 = _mm256_loadu_pd(&mat->data[col * i + j] + 12);
            sub_vector1 = _mm256_max_pd(_mm256_sub_pd(sub_vector1, orig_vector1), orig_vector1);
            sub_vector1 = _mm256_max_pd(_mm256_sub_pd(sub_vector2, orig_vector2), orig_vector2);
            sub_vector1 = _mm256_max_pd(_mm256_sub_pd(sub_vector3, orig_vector3), orig_vector3);
            sub_vector1 = _mm256_max_pd(_mm256_sub_pd(sub_vector4, orig_vector4), orig_vector4);
            _mm256_storeu_pd(&result->data[col * i + j], sub_vector1);
            _mm256_storeu_pd(&result->data[col * i + j + 4], sub_vector2);
            _mm256_storeu_pd(&result->data[col * i + j + 8], sub_vector3);
            _mm256_storeu_pd(&result->data[col * i + j + 12], sub_vector4);
        }
        for (int j = col / 16 * 16; j < col; j++) {
            int index = col * i + j;
            double value = mat->data[index];
            if (value < 0) {
                value *= -1;
            }
            result->data[col * i + j] = value;
        }
    }
    /*
    #pragma omp parallel for
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col / 4 * 4; j += 4) {
            __m256d sub_vector = _mm256_set1_pd(0.0);
            __m256d orig_vector = _mm256_loadu_pd(&mat->data[col * i + j]);
            sub_vector = _mm256_max_pd(_mm256_sub_pd(sub_vector, orig_vector), orig_vector);
            _mm256_storeu_pd(&result->data[col * i + j], sub_vector);
        }
        for (int j = col / 4 * 4; j < col; j++) {
            int index = col * i + j;
            double value = mat->data[index];
            if (value < 0) {
                value *= -1;
            }
            result->data[col * i + j] = value;
        }
    }
    */
    return 0;
}

/*
 * Stores the result of element-wise negating mat's entries to `result`.
 * Returns 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    int row = mat->rows;
    int col = mat->cols;
    #pragma omp parallel for
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col / 4 * 4; j += 4) {
            __m256d neg_vector = _mm256_set1_pd(0.0);
            __m256d orig_vector = _mm256_loadu_pd(&mat->data[col * i + j]);
            neg_vector = _mm256_sub_pd(neg_vector, orig_vector);
            _mm256_storeu_pd(&result->data[col * i + j], neg_vector);
        }
        for (int j = col / 4 * 4; j < col; j++) {
            int index = col * i + j;
            double value = mat->data[index];
            result->data[col * i + j] = -value;
        }
    }
    return 0;
}

/*
 * Stores the result of adding mat1 and mat2 to `result`.
 * Returns 0 upon success.
 * Assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    int row = mat1->rows;
    int col = mat1->cols;
    #pragma omp parallel for
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col / 16 * 16; j += 16) {
            __m256d sum_vector1 = _mm256_loadu_pd(&mat1->data[col * i + j]);
            __m256d sum_vector2 = _mm256_loadu_pd(&mat1->data[col * i + j + 4]);
            __m256d sum_vector3 = _mm256_loadu_pd(&mat1->data[col * i + j + 8]);
            __m256d sum_vector4 = _mm256_loadu_pd(&mat1->data[col * i + j + 12]);
            sum_vector1 = _mm256_add_pd(sum_vector1, _mm256_loadu_pd(&mat2->data[col * i + j]));
            sum_vector2 = _mm256_add_pd(sum_vector2, _mm256_loadu_pd(&mat2->data[col * i + j + 4]));
            sum_vector3 = _mm256_add_pd(sum_vector3, _mm256_loadu_pd(&mat2->data[col * i + j + 8]));
            sum_vector4 = _mm256_add_pd(sum_vector4, _mm256_loadu_pd(&mat2->data[col * i + j + 12]));
            _mm256_storeu_pd(&result->data[col * i + j], sum_vector1);
            _mm256_storeu_pd(&result->data[col * i + j + 4], sum_vector2);
            _mm256_storeu_pd(&result->data[col * i + j + 8], sum_vector3);
            _mm256_storeu_pd(&result->data[col * i + j + 12], sum_vector4);
        }
        for (int j = col / 16 * 16; j < col; j++) {
            result->data[col * i + j] = mat1->data[col * i + j] + mat2->data[col * i + j];
        }
    }
    /*
    #pragma omp parallel for
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col / 4 * 4; j += 4) {
            __m256d sum_vector = _mm256_loadu_pd(&mat1->data[col * i + j]);
            sum_vector = _mm256_add_pd(sum_vector, _mm256_loadu_pd(&mat2->data[col * i + j]));
            _mm256_storeu_pd(&result->data[col * i + j], sum_vector);
        }
        for (int j = col / 4 * 4; j < col; j++) {
            result->data[col * i + j] = mat1->data[col * i + j] + mat2->data[col * i + j];
        }
    }
    */
    return 0;
}

/*
 * Stores the result of subtracting mat2 from mat1 to `result`.
 * Returns 0 upon success.
 * Assumes `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    int row = mat1->rows;
    int col = mat1->cols;
    #pragma omp parallel for
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col / 4 * 4; j += 4) {
            __m256d sub_vector = _mm256_loadu_pd(&mat1->data[col * i + j]);
            sub_vector = _mm256_sub_pd(sub_vector, _mm256_loadu_pd(&mat2->data[col * i + j]));
            _mm256_storeu_pd(&result->data[col * i + j], sub_vector);
        }
        for (int j = col / 4 * 4; j < col; j++) {
            result->data[col * i + j] = mat1->data[col * i + j] - mat2->data[col * i + j];
        }
    }
    return 0;
}

/*
 * Stores the result of multiplying mat1 and mat2 to `result`.
 * Returns 0 upon success.
 * Assumes `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    matrix *trans = trans_matrix(mat2);
    int row = mat1->rows;
    int num = mat1->cols;
    int col = trans->rows;
    #pragma omp parallel for
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            double sum = 0.0;
            __m256d mul_vector1 = _mm256_set1_pd(0.0);
            __m256d mul_vector2 = _mm256_set1_pd(0.0);
            __m256d mul_vector3 = _mm256_set1_pd(0.0);
            __m256d mul_vector4 = _mm256_set1_pd(0.0);
            for (int k = 0; k < num / 16 * 16; k += 16) {
                mul_vector1 = _mm256_fmadd_pd(_mm256_loadu_pd(&mat1->data[num * i + k]), _mm256_loadu_pd(&trans->data[num * j + k]), mul_vector1);
                mul_vector2 = _mm256_fmadd_pd(_mm256_loadu_pd(&mat1->data[num * i + k + 4]), _mm256_loadu_pd(&trans->data[num * j + k + 4]), mul_vector2);
                mul_vector3 = _mm256_fmadd_pd(_mm256_loadu_pd(&mat1->data[num * i + k + 8]), _mm256_loadu_pd(&trans->data[num * j + k + 8]), mul_vector3);
                mul_vector4 = _mm256_fmadd_pd(_mm256_loadu_pd(&mat1->data[num * i + k + 12]), _mm256_loadu_pd(&trans->data[num * j + k + 12]), mul_vector4);
            }
            double temp[16] = {};
            _mm256_storeu_pd(temp, mul_vector1);
            _mm256_storeu_pd(temp + 4, mul_vector2);
            _mm256_storeu_pd(temp + 8, mul_vector3);
            _mm256_storeu_pd(temp + 12, mul_vector4);
            for (int i = 0; i < 16; i += 4) {
                sum += temp[i];
                sum += temp[i + 1];
                sum += temp[i + 2];
                sum += temp[i + 3];
            }
            for (int k = num / 16 * 16; k < num; k++) {
                sum += mat1->data[num * i + k] * trans->data[num * j + k];
            }
            result->data[col * i + j] = sum;
        }
    }
    /*
    #pragma omp parallel for
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            double sum = 0.0;
            __m256d mul_vector = _mm256_set1_pd(0.0);
            for (int k = 0; k < num / 4 * 4; k += 4) {
                mul_vector = _mm256_fmadd_pd(_mm256_loadu_pd(&mat1->data[num * i + k]), _mm256_loadu_pd(&trans->data[num * j + k]), mul_vector);
            }
            double temp[4] = {};
            _mm256_storeu_pd(temp, mul_vector);
            sum += (temp[0] + temp[1] + temp[2] + temp[3]);
            for (int k = num / 4 * 4; k < num; k++) {
                sum += mat1->data[num * i + k] * trans->data[num * j + k];
            }
            result->data[col * i + j] = sum;
        }
    }
    */
    deallocate_matrix(trans);
    return 0;
}

/*
 * Transposes mat1.
 * Returns transposed matrix of mat1.
 * Note that the matrix is in row-major order.
 */
matrix* trans_matrix(matrix *mat) {
    matrix *trans;
    allocate_matrix(&trans, mat->cols, mat->rows);
    int row = mat->rows;
    int col = mat->cols;
    #pragma omp parallel for
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            trans->data[row * j + i] = mat->data[col * i + j];
        }
    }
    return trans;
}

/*
 * Stores the result of raising mat to the (pow)th power to `result`.
 * Returns 0 upon success.
 * Assumes `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    // Task 1.6 TODO
    if (pow == 0) {
        // identity matrix
        fill_matrix(result, 0.0);
        int row = mat->rows;
        int col = mat->cols;
        #pragma omp parallel for
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (i == j) {
                    set(result, i, j, 1.0);
                }
            }
        }
    } else if (pow == 1) {
        int row = mat->rows;
        int col = mat->cols;
        #pragma omp parallel for
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                int index = col * i + j;
                double value = mat->data[index];
                result->data[index] = value;
            }
        }
    } else {
        matrix *store;
        allocate_matrix(&store, mat->rows, mat->cols);
        mul_matrix(store, mat, mat);
        if (pow % 2 == 0) {
            return pow_matrix(result, store, pow / 2);
        } else if (pow % 2 == 1) {
            pow_matrix(result, store, (pow - 1) / 2);
            memcpy(store->data, result->data, sizeof(double) * result->rows * result->cols);
            return mul_matrix(result, store, mat);
        }
        deallocate_matrix(store);
        /*
        mul_matrix(result, mat, mat);
        matrix *store;
        allocate_matrix(&store, mat->rows, mat->cols);
        for (int i = 0; i < pow - 2; i++) {
            memcpy(store->data, result->data, sizeof(double) * result->rows * result->cols);
            mul_matrix(result, store, mat);
        }
        deallocate_matrix(store);
        */
    }
    return 0;
}

