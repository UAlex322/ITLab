#pragma once

#include "LU.h"

void LU1(MTX_TYPE *A, MTX_TYPE *L, MTX_TYPE *U, size_t n) {
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < n; ++j)
			U[i*n+j] = A[i*n+j];

	for (size_t i = 0; i < n; ++i) {
		L[i*(n+1)] = 1.0;
		for (size_t j = i+1; j < n; ++j)
			L[i*n+j] = 0.0;
	}

	MTX_TYPE mult;
	for (size_t j = 0; j < n-1; ++j) {
		for (size_t i = j+1; i < n; ++i) {
			mult = U[i*n+j]/U[j*(n+1)];
			L[i*n+j] = mult;
#pragma omp parallel for
			for (size_t k = j; k < n; ++k) {
				U[i*n+k] -= mult*U[j*n+k];
				if (abs(U[i*n+k]) < eps) U[i*n+k] = 0.0;
			}
		}
	}
}