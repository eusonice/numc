# numc

## Spec:
> The full spec can be read [here](https://cs61c.org/sp22/projects/proj4/).

A simple version of NumPy, a Python library for performing mathematical and logical operations on arrays and matrices, written in C. 

## Notes: 
The library performs the following:
- **basic operations**: fill_matrix, abs_matrix, neg_matrix, add_matrix, sub_matrix
- **extra operations**: mul_matrix, pow_matrix

Optimized performance in speed with the following methods:
- **SIMD instructions**: parallelizing using AVX2
- **multi threading**: parallelizing using OpenMP
- **loop unrolling**
- **cache blocking**
- **repeated squaring**: efficient exponentiation algorithm

Achieved speed-up in matrix multiplications (mul_matrix) by 60.03 and speed-up in powers in matrix (pow_matrix) by 912.25.
