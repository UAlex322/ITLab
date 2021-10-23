#pragma once
#include "Matrix.h"
#include <cmath>

#define MTX_TYPE double
#define eps 1e-12

void LU(MTX_TYPE* A, MTX_TYPE* L, MTX_TYPE* U, size_t n);

template <typename T>
void LU1(Matrix<T> &A, Matrix<T> &L, Matrix<T> &U) {
	U = A;
	size_t n = A.rows();
	for (size_t i = 0; i < n; ++i) {
		L(i,i) = 1.0;
		for (size_t j = i+1; j < n; ++j)
			L(i,j) = 0.0;
	}

	T mult;
	for (size_t j = 0; j < n-1; ++j) {
		for (size_t i = j+1; i < n; ++i) {
			mult = U(i,j)/U(j,j);
			L(i,j) = mult;
			for (size_t k = j+1; k < n; ++k) {
				U(i,k) -= mult*U(j,k);
				if (abs(U(i,k)) < eps) U(i,k) = 0.0;
			}
		}
	}
}

template <typename T>
Matrix<T> LU2(Matrix<T> &A) {
	Matrix<T> U = A;
	T mult;
	size_t n = A.rows();

	for (size_t j = 0; j < n-1; ++j) {
		for (size_t i = j+1; i < n; ++i) {
			mult = U(i,j)/U(j,j);
			U(i,j) = mult;
			for (size_t k = j+1; k < n; ++k) {
				U(i,k) -= mult*U(j,k);
				if (abs(U(i,k)) < eps) U(i,k) = 0.0;
			}
		}
	}

	return U;
}