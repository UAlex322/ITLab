#pragma once
#include <iostream>
#include <random>
#include <omp.h>
using namespace std;

template <typename T>
class Matrix {
public:

	friend void check_correct(const Matrix &M1, const Matrix &M2);

	Matrix(int rows = 0, int cols = 0): m(rows), n(cols) {
		data = new T[m*n];
	}

	Matrix(const Matrix &mtx): m(mtx.m), n(mtx.n) {
		data = new T[m*n];
		for (int i = 0; i < m*n; ++i)
			data[i] = mtx.data[i];
	}

	Matrix& operator=(const Matrix &mtx) {
		if (this == &mtx) return *this;

		if (m != mtx.m || n != mtx.n) {
			delete[] data;
			data = new T[mtx.m*mtx.n];
			m = mtx.m;
			n = mtx.n;
		}

		for (int i = 0; i < m*n; ++i)
			data[i] = mtx.data[i];

		return *this;
	}

	~Matrix() {
		delete[] data;
		m = 0; n = 0;
	}
	
	inline void generate_random_matrix() {
		mt19937 gen(random_device{}());
		uniform_real_distribution<T> dist(-5.0, 5.0);
		#pragma omp parallel for
		for (int i = 0; i < m*n; ++i)
			data[i] = dist(gen);
	}

	inline void generate_well_conditioned_matrix(const T &avg_diag_val, const T &max_val) {
		mt19937 gen(random_device{}());
		uniform_real_distribution<float> d(-max_val, max_val);

		for (int i = 0; i < m*n; ++i)
			data[i] = d(gen);
		for (int i = 0; i < m; ++i)
			data[i*(m+1)] = (gen() % 2 ? -1 : 1) * (avg_diag_val + gen() % 10000 * 0.001);
	}

	inline const T& operator()(int row, int col) const {
		return data[row*n + col];
	}

	inline T& operator()(int row, int col) {
		return data[row*n + col];
	}

	void enter() {
		for (int i = 0; i < m; ++i)
			for (int j = 0; j < n; ++j)
				std::cin >> data[i*n + j];
	}

	void print() {
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j)
				std::cout << std::fixed << data[i*n + j] << " ";
			std::cout << '\n';
		}
		std::cout << '\n';
	}

	Matrix &operator+=(const Matrix &mtx) {
		T *m_ptr = mtx.data;

		for (int i = 0; i < m*n; ++i)
			data[i] += m_ptr[i];

		return *this;
	}

	Matrix &operator-=(const Matrix &mtx) {
		T *m_ptr = mtx.data;

		for (int i = 0; i < m*n; ++i)
			data[i] -= m_ptr[i];

		return *this;
	}

	Matrix operator*(const Matrix &b) {
		int p = b.n;
		Matrix<T> c(m, p);

		memset(c.data, 0, m*p*sizeof(T));

		mult(data, b.data, c.data, n, p, p, m, n, b.n);

		return c;
	}



	// Обычный алгоритм (последовательная версия)
	void lu_trivial_sequential() {
		T *ptr1 = data, *ptr2;
		T mult;

		for (int j = 0; j < n-1; ++j, ptr1 += n) {
			ptr2 = ptr1 + n;

			for (int i = j+1; i < n; ++i, ptr2 += n) {
				mult = ptr2[j]/ptr1[j];
				ptr2[j] = mult;

				for (int k = j+1; k < n; ++k) {
					ptr2[k] -= mult*ptr1[k];
				}
			}
		}
	}
	
	// Обычный алгоритм (параллельная версия)
	void lu_trivial_parallel_omp() {
		T *ptr1 = data;

		for (int j = 0; j < n-1; ++j, ptr1 += n) {
		#pragma omp parallel for
			for (int i = j+1; i < n; ++i) {
				T *ptr2 = data + i*n;
				T mult = ptr2[j]/ptr1[j];
				ptr2[j] = mult;

				for (int k = j+1; k < n; ++k) {
					ptr2[k] -= mult*ptr1[k];
				}
			}
		}
	}

	// Блочный алгоритм (параллельная версия)
	void lu_block_parallel_omp(const int ls, const int mbs) {
		T *dptr = data;
		const int bs = ls;		// размер блока
		int bi = 0, nbi = bs;	// индексы начала текущего и следующего блоков

		T *buffer = new T[mbs*bs*omp_get_max_threads()]; // буфер для блоков матрицы B в матричном умножении

		for (; nbi < n; dptr += bs*(n+1), bi += bs, nbi += bs) {
			LU(dptr, n, bs);

			LSolve(dptr, dptr + bs, n, bs, n - nbi);
			USolve(dptr, dptr + bs*n, n, n - nbi, bs);
			
			FMMS(dptr + bs*n, dptr + bs, dptr + bs*(n + 1), buffer, n, n, n, n - nbi, bs, n - nbi, bs, mbs);
			//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n-nbi, bs, n-nbi, -1.0, dptr + bs*n, n, dptr + bs, n, 1.0, dptr + bs*(n+1), n);
		}
		LU(dptr, n, n-bi);

		delete[] buffer;
	}

private:

	// Общий алгоритм умножения матриц 
	void mult(const T* a_ptr, const T* b_ptr, T *c_ptr, const int lda, const int ldb, const int ldc, const int m, const int n, const int p) {

	#pragma omp parallel for
		for (int i = 0; i < m; ++i) {
			T a_elem, *c_curr = c_ptr + i*ldc;
			const T *b_curr = b_ptr;

			for (int k = 0; k < n; ++k, b_curr += ldb) {
				a_elem = a_ptr[i*lda + k];

				for (int j = 0; j < p; ++j)
					c_curr[j] += a_elem * b_curr[j];
			}
		}
	}

	// Оптимизированная версия FMMS с транспонированием блоков
	void FMMS(const T *a_ptr, const T *b_ptr, T *c_ptr, T *buffer, const int lda, const int ldb, const int ldc, const int m, const int n, const int p, const int bs, const int mbs) {
		const int num_of_blocks = (m+mbs-1)/mbs;
		int *pos = new int[num_of_blocks+1];
		for (int i = 0; i < num_of_blocks; ++i)
			pos[i] = i*mbs;
		pos[num_of_blocks] = m;

	#pragma omp parallel for
		for (int jb = 0; jb < num_of_blocks; ++jb) {
			const int bb_len = pos[jb+1]-pos[jb];
			const T *nb_ptr = b_ptr + pos[jb],
					*a_curr = a_ptr,
					*b_curr;
				  T *c_curr = c_ptr + pos[jb],
					*nbt_ptr = buffer + mbs*bs*omp_get_thread_num(),
					 sum;

			for (int i = 0; i < bs; ++i)
				for (int j = 0; j < bb_len; ++j)
					nbt_ptr[j*bs + i] = nb_ptr[i*ldb + j];

			for (int i = 0; i < m; ++i, a_curr += lda, c_curr += ldc) {
				b_curr = nbt_ptr;
				for (int j = 0; j < bb_len; ++j, b_curr += bs) {
					sum = 0.0;
					for (int k = 0; k < bs; ++k)
						sum += a_curr[k] * b_curr[k]; //a_curr[i*lda + k] * nbt_ptr[j*bs + k];
					c_curr[j] -= sum;
				}
			}
		}
	}

	// Общее LU-разложение (применяется для блока в матрице)
	void LU(T *a_ptr, const int lda, const int n) {
		T *ptr1 = a_ptr; // указатель на текущую вычитаемую строку

		for (int j = 0; j < n-1; ++j, ptr1 += lda) {
		#pragma omp parallel for
			for (int i = j+1; i < n; ++i) {
				T *ptr2 = data + i*n;		// указатель на текущую уменьшаемую строку
				T mult = ptr2[j]/ptr1[j];	// множитель, на который умножается вычитаемая строка
				ptr2[j] = mult;

				for (int k = j+1; k < n; ++k) {
					ptr2[k] -= mult*ptr1[k];
				}
			}
		}
	}

	// Решение системы LX = B; L - верхнетреугольная матрица, X,B - подходящие прямоугольные матрицы (одинакового размера)
	void LSolve(const T *l_ptr, T *a_ptr, const int lda, const int m, const int n) {
		// m - высота обеих матриц, n - длина искомой матрицы
		const int block_size = m, num_of_blocks = (n+m-1)/m;

	#pragma omp parallel for
		for (int it = 0; it < num_of_blocks; ++it) {
			int block_len = (it+1 == num_of_blocks) ? ((n % block_size != 0) ? n % block_size : block_size) : block_size; // длина текущего блока
			T *na_ptr = a_ptr + block_size*it, // указатель на начало текущего блока
			  *ptr1 = na_ptr,				   // указатель на текущую вычитаемую строку
			  *ptr2,						   // указатель на текущую уменьшаемую строку
			  mult;							   // множитель, на который умножается вычитаемая строка

			for (int j = 0; j < block_size - 1; ++j, ptr1 += lda) {
				ptr2 = ptr1 + lda;

				for (int i = j + 1; i < block_size; ++i, ptr2 += lda) {
					mult = -l_ptr[i*lda + j];

					for (int k = 0; k < block_len; ++k) {
						ptr2[k] += mult*ptr1[k];
					}
				}
			}
		}
	}

	// Решение системы XU = B; U - верхнетреугольная матрица, X,B - подходящие прямоугольные матрицы (одинакового размера)
	void USolve(const T *u_ptr, T *a_ptr, const int lda, const int m, const int n) {
		// m - высота искомой матрицы, n - длина обеих матриц
		const int block_size = n, num_of_blocks = (m + n - 1)/n;

	#pragma omp parallel for
		for (int it = 0; it < num_of_blocks; ++it) {
			int block_len = (it + 1 == num_of_blocks) ? ((m % block_size != 0) ? m % block_size : block_size) : block_size; // высота текущего блока
			const T *u_curr;					   // указатель на текущую строку верхнетреугольной матрицы
			T *na_ptr = a_ptr + block_size*it*lda, // указатель на начало блока
			  *a_curr = na_ptr,					   // указатель на текущую строку матрицы в правой части
			   mult;							   // множитель, на который умножается вычитаемый элемент

			for (int k = 0; k < block_len; ++k, a_curr += lda) {
				u_curr = u_ptr;
				for (int i = 0; i < block_size-1; ++i, u_curr += lda) {
					mult = -a_curr[i]/u_curr[i];
					for (int j = i + 1; j < block_size; ++j) {
						a_curr[j] += mult*u_curr[j];
					}
				}
			}

			for (int j = 0; j < block_size; ++j) {
				mult = u_ptr[j*(lda + 1)];
				for (int k = 0; k < block_len; ++k)
					na_ptr[k*lda + j] /= mult;
			}
		}
	}

	// Без транспонирования
	void FMMS1(const T *a_ptr, const T *b_ptr, T *c_ptr, const int lda, const int ldb, const int ldc, const int m, const int n, const int p, const int bs) {
		const int num_of_blocks = (m+bs-1)/bs;
		int *pos = new int[num_of_blocks+1];
		for (int i = 0; i < num_of_blocks; ++i)
			pos[i] = i*bs;
		pos[num_of_blocks] = m;

	#pragma omp parallel for
		for (int ib = 0; ib < num_of_blocks; ++ib) {
			for (int jb = 0; jb < num_of_blocks; ++jb) {
				//for (int it = 0; it < num_of_blocks*num_of_blocks; ++it) {
				const int //ib = it/num_of_blocks,
						  //jb = it % num_of_blocks,
					ba_len = pos[ib+1]-pos[ib],
					bb_len = pos[jb+1]-pos[jb];
				const T *na_ptr = a_ptr + pos[ib]*lda,
					*nb_ptr = b_ptr + pos[jb];
				T		*nc_ptr = c_ptr + pos[ib]*ldc + pos[jb],
					a_elem;

				for (int i = 0; i < ba_len; ++i, nc_ptr += ldc, na_ptr += lda) {
					const T *b_curr = nb_ptr;

					for (int k = 0; k < bs; ++k, b_curr += ldb) {
						a_elem = na_ptr[k];

						for (int j = 0; j < bb_len; ++j)
							nc_ptr[j] -= a_elem * b_curr[j];
					}
				}
			}
		}
	}

	
	// 
	T* data;
	int m, n; // Число строк и столбцов
};

void check_correct(const Matrix<double> &M1, const Matrix<double> &M2) {
	if (M1.m != M2.m || M1.n != M2.n)
		throw "Matrices have different sizes";

	double *ptr1 = M1.data, *ptr2 = M2.data;
	double ae = 0.0, re = 0.0;

#pragma omp parallel for
	for (int i = 0; i < M1.m * M1.n; ++i) {
		ae = std::max(ae, abs(ptr2[i] - ptr1[i]));
		re = std::max(re, (ptr1[i] == 0.0) ? 0.0 : abs(ptr2[i]/ptr1[i] - 1.0));
	}
	cout << "Absolute error: " << ae << '\n' << "Relative error: " << re << endl;
}

void check_correct(const Matrix<float> &M1, const Matrix<float> &M2) {
	if (M1.m != M2.m || M1.n != M2.n)
		throw "Matrices have different sizes";

	float *ptr1 = M1.data, *ptr2 = M2.data;
	float ae = 0.0, re = 0.0;

#pragma omp parallel for
	for (int i = 0; i < M1.m * M1.n; ++i) {
		ae = std::max(ae, abs(ptr2[i] - ptr1[i]));
		re = std::max(re, (ptr1[i] == 0.0f) ? 0.0f : abs(ptr2[i]/ptr1[i] - 1.0f));
	}
	cout << "Absolute error: " << ae << '\n' << "Relative error: " << re << std::endl;
}

