#include <iostream>
#include <omp.h>

template <typename T>
class Matrix {
public:

	Matrix(int rows = 0, int cols = 0): m(rows), n(cols) {
		data = new T[rows*cols];
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

	void generate_random_matrix() {
		srand(time(0));

		for (int i = 0; i < m; ++i)
			for (int j = 0; j < n; ++j)
				data[i*n+j] = (rand() - 16384) / 2048.0;
	}

	void generate_well_conditioned_matrix(T &mean_value) {
		srand(time(0));

		for (int i = 0; i < m; ++i)
			for (int j = 0; j < n; ++j) {
				data[i*n+j] = (rand() - 16384) / 2048.0;
			}
		for (int i = 0; i < m; ++i)
			data[i*(m+1)] = (rand() % 2 ? -1 : 1) * (mean_value + (rand() % 10000) * 0.0001);
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
		Matrix<T> c(m, b.n);
		int p = b.n;

		for (int i = 0; i < n; ++i)
			for (int j = 0; j < p; ++j)
				c.data[i*p + j] = 0.0;

		Mult(data, b.data, c.data, n, p, p, m, n, b.n);

		return c;
	}

	void LU1(Matrix &L, Matrix &U) {
		U = *this;
		T *l_ptr = L.data, *u_ptr = U.data, *ptr1, *ptr2;

		for (int i = 0; i < n; ++i) {
			l_ptr[i*(n+1)] = 1.0;
		}
		for (int i = 0; i < n; ++i)
			for (int j = i+1; j < n; ++j)
				l_ptr[i*n + j] = 0.0;

		T mult;
		for (int j = 0; j < n-1; ++j) {
			ptr1 = u_ptr + j*n;

			for (int i = j+1; i < n; ++i) {
				ptr2 = u_ptr + i*n;
				mult = ptr2[j]/ptr1[j];
				l_ptr[i*n + j] = mult;
				
			//#pragma omp parallel for
				for (int k = j; k < n; ++k) {
					ptr2[k] -= mult*ptr1[k];
				}
			}
		}
	}

	// Обычный алгоритм (последовательная версия)
	void LU2_sequential() {
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
	void LU2_parallel() {
		T *ptr1 = data, *ptr2;
		T mult;

		for (int j = 0; j < n-1; ++j, ptr1 += n) {
			ptr2 = ptr1 + n;

			for (int i = j+1; i < n; ++i, ptr2 += n) {
				mult = ptr2[j]/ptr1[j];
				ptr2[j] = mult;

			#pragma omp parallel for
				for (int k = j+1; k < n; ++k) {
					ptr2[k] -= mult*ptr1[k];
				}
			}
		}
	}

	// Блочный алгоритм (параллельная версия)
	void LU3_block() {
		T *dptr = data;
		const int bs = 100;	  // размер блока
		int bi = 0, nbi = bs; // индексы начала текущего и следующего блоков

		for (; nbi < n; dptr += bs*(n+1), bi += bs, nbi += bs) {
			LU(dptr, n, bs);

			LSolve(dptr, dptr + bs, n, bs, n - nbi);
			USolve(dptr, dptr + bs*n, n, n - nbi, bs);

			FMMS1(dptr + bs*n, dptr + bs, dptr + bs*(n + 1), n, n, n, n - nbi, bs, n - nbi, bs);
		}
		LU(dptr, n, n-bi);
	}

private:

	// Общий алгоритм умножения матриц 
	void Mult(const T* a_ptr, const T* b_ptr, T *c_ptr, const int lda, const int ldb, const int ldc, const int m, const int n, const int p) {
		//int bp, cp;

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

	// "Fused Matrix Multiply - Subtract" - "C = C - A*B"
	// Занимает >90% общего времени работы
	void FMMS(const T *a_ptr, const T *b_ptr, T *c_ptr, const int lda, const int ldb, const int ldc, const int m, const int n, const int p) {

	#pragma omp parallel for
		for (int i = 0; i < m; ++i) {
			const T *b_curr = b_ptr;
			T a_elem,
			 *c_curr = c_ptr + i*ldc; 

			for (int k = 0; k < n; ++k, b_curr += ldb) {
				a_elem = -a_ptr[i*lda + k];

				for (int j = 0; j < p; ++j)
					c_curr[j] += a_elem * b_curr[j];
			}
		}
	}

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

	// Общее LU-разложение (применяется для блока в матрице)
	void LU(T *a_ptr, const int lda, const int n) {
		T *ptr1 = a_ptr, // указатель на текущую вычитаемую строку 
		  *ptr2,		 // указатель на текущую уменьшаемую строку
		  mult;			 // множитель, на который умножается вычитаемая строка

		for (int j = 0; j < n - 1; ++j, ptr1 += lda) {
			ptr2 = ptr1 + lda;

			for (int i = j + 1; i < n; ++i, ptr2 += lda) {
				mult = ptr2[j]/ptr1[j];
				ptr2[j] = mult;

			//#pragma omp parallel for
				for (int k = j + 1; k < n; ++k) {
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

	
	// 
	T* data;
	int m, n; // Number of rows and columns
};