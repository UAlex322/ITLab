#include <iostream>
#include <omp.h>
#define eps 1e-17

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
		for (int i = 0; i < m; ++i)
			for (int j = 0; j < n; ++j)
				data[i*n + j] += m_ptr[i*n + j];

		return *this;
	}

	Matrix &operator-=(const Matrix &mtx) {
		T *m_ptr = mtx.data;
		for (int i = 0; i < m; ++i)
			for (int j = 0; j < n; ++j)
				data[i*n + j] -= m_ptr[i*n + j];

		return *this;
	}

	Matrix operator*(const Matrix &b) {
		Matrix<T> c(m, b.n);

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

	void LU3_block() {
		T *dptr = data;
		const int bs = 100;	  // ������ �����
		int bi = 0, nbi = bs; // ������� ������ �������� � ���������� ������

		for (; nbi < n; dptr += bs*(n+1), bi += bs, nbi += bs) {
			LU(dptr, n, bs);

			LSolve(dptr, dptr + bs, n, bs, n - nbi);
			USolve(dptr, dptr + bs*n, n, n - nbi, bs);

			FMS(dptr + bs*n, dptr + bs, dptr + bs*(n + 1), n, n, n, n - nbi, bs, n - nbi);
		}
		LU(dptr, n, n-bi);
	}

private:
	void Mult(const T* a_ptr, const T* b_ptr, T *c_ptr, const int lda, const int ldb, const int ldc, const int m, const int n, const int p) {
		//int bp, cp;

	#pragma omp parallel for
		for (int i = 0; i < m; ++i) {
			T a_elem, *c_curr = c_ptr + i*ldc, *b_curr = b_ptr;

			for (int k = 0; k < n; ++k, b_curr += ldb) {
				a_elem = a_ptr[i*lda + k];

				for (int j = 0; j < p; ++j)
					c_curr[j] += a_elem * b_curr[j];
			}
		}
	}

	// �������� >90% ������ ������� ������
	void FMS(const T *a_ptr, const T *b_ptr, T *c_ptr, const int lda, const int ldb, const int ldc, const int m, const int n, const int p) {

	#pragma omp parallel for
		for (int i = 0; i < m; ++i) {
			const T *b_curr = b_ptr;
			T a_elem,
			 *c_curr = c_ptr + i*ldc; 

			for (int k = 0; k < n; ++k, b_curr += ldb) {
				a_elem = a_ptr[i*lda + k];

				for (int j = 0; j < p; ++j)
					c_curr[j] -= a_elem * b_curr[j];
			}
		}
	}

	void LU(T *a_ptr, const int lda, const int n) {
		T *ptr1 = a_ptr, // ��������� �� ������� ���������� ������ 
		  *ptr2,		 // ��������� �� ������� ����������� ������
		  mult;			 // ���������, �� ������� ���������� ���������� ������

		for (int j = 0; j < n - 1; ++j, ptr1 += lda) {
			ptr2 = ptr1 + lda;

			for (int i = j + 1; i < n; ++i, ptr2 += lda) {
				mult = ptr2[j]/ptr1[j];
				ptr2[j] = mult;

			#pragma omp parallel for
				for (int k = j + 1; k < n; ++k) {
					ptr2[k] -= mult*ptr1[k];
				}
			}
		}
	}

	void LSolve(const T *l_ptr, T *a_ptr, const int lda, const int m, const int n) {
		// m - ������ ����� ������, n - ����� ������� �������
		const int block_size = m, num_of_blocks = (n+m-1)/m;

	#pragma omp parallel for
		for (int it = 0; it < num_of_blocks; ++it) {
			int block_len = (it+1 == num_of_blocks) ? ((n % block_size != 0) ? n % block_size : block_size) : block_size; // ����� �������� �����
			T *na_ptr = a_ptr + block_size*it, // ��������� �� ������ �������� �����
			  *ptr1 = na_ptr,				   // ��������� �� ������� ���������� ������
			  *ptr2,						   // ��������� �� ������� ����������� ������
			  mult;							   // ���������, �� ������� ���������� ���������� ������

			for (int j = 0; j < block_size - 1; ++j, ptr1 += lda) {
				ptr2 = ptr1 + lda;

				for (int i = j + 1; i < block_size; ++i, ptr2 += lda) {
					mult = l_ptr[i*lda + j];

					for (int k = 0; k < block_len; ++k) {
						ptr2[k] -= mult*ptr1[k];
					}
				}
			}
		}
	}

	void USolve(const T *u_ptr, T *a_ptr, const int lda, const int m, const int n) {
		// m - ������ ������� �������, n - ����� ����� ������
		const int block_size = n, num_of_blocks = (m + n - 1)/n;

	#pragma omp parallel for
		for (int it = 0; it < num_of_blocks; ++it) {
			int block_len = (it + 1 == num_of_blocks) ? ((m % block_size != 0) ? m % block_size : block_size) : block_size; // ������ �������� �����
			const T *u_curr;					   // ��������� �� ������� ������ ����������������� �������
			T *na_ptr = a_ptr + block_size*it*lda, // ��������� �� ������ �����
			  *a_curr = na_ptr,					   // ��������� �� ������� ������ ������� � ������ �����
			   mult;							   // ���������, �� ������� ���������� ���������� �������

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