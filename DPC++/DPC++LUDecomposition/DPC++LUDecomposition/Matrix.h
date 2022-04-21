#pragma once
#include <iostream>
#include <random>
#include <CL/sycl.hpp>
#include <omp.h>
using namespace sycl;
using namespace std;

class Matrix {
public:

	friend void check_correct(const Matrix &M1, const Matrix &M2, int m, int n);

	Matrix(int rows = 0, int cols = 0): m(rows), n(cols) {
		data = new float[m*n];
	}

	Matrix(const Matrix &mtx): m(mtx.m), n(mtx.n) {
		data = new float[m*n];
		for (int i = 0; i < m*n; ++i)
			data[i] = mtx.data[i];
	}

	Matrix& operator=(const Matrix &mtx) {
		if (this == &mtx) return *this;

		if (m != mtx.m || n != mtx.n) {
			delete[] data;
			data = new float[mtx.m*mtx.n];
			m = mtx.m;
			n = mtx.n;
		}

		for (int i = 0; i < m*n; ++i)
			data[i] = mtx.data[i];

		return *this;
	}

	~Matrix() {
		delete[] data;
	}

	void generate_random_matrix(const float max_val) {
		mt19937 gen(random_device{}());
		uniform_real_distribution<float> d(-max_val, max_val);

		for (int i = 0; i < m*n; ++i)
				data[i] = d(gen);
	}

	void generate_well_conditioned_matrix(const float avg_diag_val, const float max_val) {
		mt19937 gen(random_device{}());
		uniform_real_distribution<float> d(-max_val, max_val);

		for (int i = 0; i < m*n; ++i)
			data[i] = d(gen);
		for (int i = 0; i < m; ++i)
			data[i*(m+1)] = (gen() % 2 ? -1 : 1) * (avg_diag_val + gen() % 10000 * 0.001);
	}

	inline const float& operator()(int row, int col) const {
		return data[row*n + col];
	}

	inline float& operator()(int row, int col) {
		return data[row*n + col];
	}

	void enter() {
		for (int i = 0; i < m; ++i)
			for (int j = 0; j < n; ++j)
				std::cin >> data[i*n + j];
	}

	void print() const {
		std::cout.precision(8);
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j)
				std::cout << std::fixed << data[i*n + j] << " ";
			std::cout << '\n';
		}
		std::cout << '\n';
	}

	Matrix &operator+=(const Matrix &mtx) {
		float *m_ptr = mtx.data;

		for (int i = 0; i < m*n; ++i)
			data[i] += m_ptr[i];

		return *this;
	}

	Matrix &operator-=(const Matrix &mtx) {
		float *m_ptr = mtx.data;

		for (int i = 0; i < m*n; ++i)
			data[i] -= m_ptr[i];

		return *this;
	}

	Matrix operator*(const Matrix &b) {
		Matrix c(m, b.n);
		int p = b.n;

		memset(c.data, 0, m*p*sizeof(float));

		Mult(data, b.data, c.data, n, p, p, m, n, b.n);

		return c;
	}



	// Обычный алгоритм (последовательная версия)
	void lu_trivial_sequential() {
		float *ptr1 = data, *ptr2;
		float mult;

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
		float *ptr1 = data, *ptr2;
		float mult;

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
	void lu_block_parallel_dpc(const int ls, const int mbs) {
		float *dptr = data;
		const int bs = ls;		// размер блока
		int bi = 0, nbi = bs;	// индексы начала текущего и следующего блоков

		//float *buffer = new float[bs*n]; // буфер для блоков матрицы B в матричном умножении

		for (; nbi < n; dptr += bs*(n+1), bi += bs, nbi += bs) {
			LU(dptr, n, bs);

			LSolve(dptr, dptr + bs, n, bs, n - nbi);
			USolve(dptr, dptr + bs*n, n, n - nbi, bs);

			//std::cout << n - nbi << std::endl;
			FMMS(dptr + bs*n, dptr + bs, n, n - nbi, bs, mbs);

			//cblas_dgemm(CblasRowMajor, CblasNofloatrans, CblasNofloatrans, n-nbi, bs, n-nbi, -1.0, dptr + bs*n, n, dptr + bs, n, 1.0, dptr + bs*(n+1), n);
		}
		LU(dptr, n, n-bi);

		//delete[] buffer;
	}

	TYPE* get_ptr() const {
		return data;
	}

private:
	// Общий алгоритм умножения матриц 
	void Mult(const float* a_ptr, const float* b_ptr, float *c_ptr, const int lda, const int ldb, const int ldc, const int m, const int n, const int p) {
		//int bp, cp;

	#pragma omp parallel for
		for (int i = 0; i < m; ++i) {
			float a_elem, *c_curr = c_ptr + i*ldc;
			const float *b_curr = b_ptr;

			for (int k = 0; k < n; ++k, b_curr += ldb) {
				a_elem = a_ptr[i*lda + k];

				for (int j = 0; j < p; ++j)
					c_curr[j] += a_elem * b_curr[j];
			}
		}
	}

	void FMMS(float *a_ptr, float *b_ptr, const int lda, const int size, const int bs, const int mbs) {
		//std::cout << "Call FMMS" << std::endl;

		{
			buffer<float,1> buf1(a_ptr, range<1>{size*lda-lda+size}); // буфер для матриц A и C
			buffer<float,1> buf2(b_ptr, range<1>{bs*lda-lda+bs}); // буфер для B

			
			sycl::event event = q.submit([&](handler& cgh) {

				sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> buffer(2*bs*bs, cgh);

				auto a = buf1.get_access<sycl::access::mode::read>(cgh);
				auto b = buf2.get_access<sycl::access::mode::read>(cgh);
				auto c = buf1.get_access<sycl::access::mode::write>(cgh);

				cgh.parallel_for<class _Mult>(nd_range<2>(range<2>(size,size), range<2>(mbs,mbs)), [=](nd_item<2> item) {
					float* block_a = buffer.get_pointer();
					float* block_b = block_a + bs*bs;

					size_t li = item.get_local_id(0);			//локальный индекс в группе (строка)
					//size_t shift = li*bs;
					size_t lj = item.get_local_id(1);
					uint32_t gi = item.get_global_id(0);
					uint32_t gj = item.get_global_id(1);
					//uint32_t gi = bs*item.get_group(0) + li;	//начало номера группы по строке 
					//uint32_t gj = bs*item.get_group(1) + lj;

					block_a[li*bs + lj] = a[gi*lda + lj];
					block_b[li*bs + lj] = b[li*lda + gj];
					item.barrier(sycl::access::fence_space::local_space);
					
					float sum = 0.0f;
					for (int k = 0; k < bs; ++k) {
						sum += block_a[li*bs + k] * block_b[k*bs + lj];
					}
					c[gi*lda + bs + gj] -= sum;
					item.barrier(sycl::access::fence_space::global_space);
				});
			});
			event.wait();
			
			
			/*
			q.submit([&](handler &cgh) {
				auto a_acc = buf1.get_access<sycl::access::mode::read>(cgh); // Матрица А
				auto b_acc = buf2.get_access<sycl::access::mode::read>(cgh); // Матрица B
				auto c_acc = buf1.get_access<sycl::access::mode::write>(cgh);
				accessor<float, 1, access::mode::read_write, access::target::local> block_a_acc(mbs*bs, cgh);

				cgh.parallel_for<class Mult>(nd_range<2>(range<2>(size,size), range<2>(mbs,mbs)), [=](nd_item<2> item) {
					int a_old_shift = item.get_global_id(0)*lda,
						a_new_shift = item.get_local_id(0)*bs,
						b_idx		= item.get_global_id(1);

					block_a_acc[a_new_shift + item.get_local_id(1)] = a_acc[a_old_shift + item.get_local_id(1)];
					item.barrier(access::fence_space::local_space);
					
					float sum = 0.0f;
					for (int j = 0; j < bs; ++j)
						sum += block_a_acc[a_new_shift + j] * b_acc[j*lda + b_idx];
					c_acc[a_old_shift + bs + item.get_global_id(1)] -= sum;
					item.barrier(access::fence_space::local_space);
				});
			}).wait();
			*/

			/*
			q.submit([&](handler &cgh) {
				auto a_acc = buf1.get_access<sycl::access::mode::read_write>(cgh); // Матрицы А и С
				auto b_acc = buf2.get_access<sycl::access::mode::read>(cgh); // Матрица B
				accessor<float, 1, access::mode::read_write, access::target::local> block_a_acc(mbs*bs, cgh); // Матрица A

				cgh.parallel_for<class Mult>(nd_range<1>(range<1>(size), range<1>(mbs)), [=](nd_item<1> item) {
					int a_old_shift = item.get_global_id(0)*lda,
						a_new_shift = item.get_local_id(0)*bs,
						b_shift = 0;

					for (int j = 0; j < bs; ++j)
						block_a_acc[a_new_shift + j] = a_acc[a_old_shift + j];
					item.barrier(access::fence_space::local_space);

					float sum;
					for (int i = 0; i < size; ++i, b_shift += bs) {
						sum = 0.0f;
						for (int j = 0; j < bs; ++j)
							sum += block_a_acc[a_new_shift + j] * b_acc[b_shift + j];
						a_acc[a_old_shift + bs + i] -= sum;
					}
				});
			}).wait();
			*/
			
			//std::cout << "Call submit" << std::endl;
			/*
			q.submit([&](handler &cgh) {
				auto a_acc = buf1.get_access<sycl::access::mode::read_write>(cgh);
				auto b_acc = buf2.get_access<sycl::access::mode::read>(cgh);

				//std::cout << "Call parallel_for" << std::endl;
				cgh.parallel_for(range<2>(size,size), [=](item<2> item) {
					int a0 = item.get_id(0)*lda,
						b0 = item.get_id(1),
						bn = b0*n;

						float sum = 0.0f;
						for (int k = 0; k < bs; ++k)
							sum += a_acc[a0 + k] * b_acc[bn + k];
						a_acc[a0 + b0] -= sum;
				});
				//std::cout << "End of parallel_for" << std::endl;
			});
			*/

			/*
			q.submit([&](handler &cgh) {
				auto a_acc = buf1.get_access<sycl::access::mode::read_write>(cgh);
				auto b_acc = buf2.get_access<sycl::access::mode::read>(cgh);
				accessor<float, 2, access::mode::read_write, access::target::local> buffer(range(bs,mbs),cgh);
				//std::cout << "Call parallel_for" << std::endl;
				cgh.parallel_for(nd_range<2>(range<2>(size,size), range<2>(mbs,mbs)), [=](nd_item<2> item) {
					int a0 = item.get_global_id(0)*lda,
						b0 = item.get_global_id(1),
						bn = b0*n;

					float sum = 0.0f;
					for (int k = 0; k < bs; ++k)
						sum += a_acc[a0 + k] * b_acc[bn + k];
					a_acc[a0 + bs + b0] -= sum;
				});
				//std::cout << "End of parallel_for" << std::endl;
			}).wait();
			*/
			
			

			//std::cout << "End of submit" << std::endl;
		}
		//std::cout << "End of FMMS" << std::endl;
	}

	// Общее LU-разложение (применяется для блока в матрице)
	void LU(float *a_ptr, const int lda, const int n) {
		float *ptr1 = a_ptr, // указатель на текущую вычитаемую строку 
			  *ptr2,		 // указатель на текущую уменьшаемую строку
			   mult;		 // множитель, на который умножается вычитаемая строка

		for (int j = 0; j < n-1; ++j, ptr1 += lda) {
			ptr2 = ptr1 + lda;

			for (int i = j+1; i < n; ++i, ptr2 += lda) {
				mult = ptr2[j]/ptr1[j];
				ptr2[j] = mult;

				//#pragma omp parallel for
				for (int k = j+1; k < n; ++k) {
					ptr2[k] -= mult*ptr1[k];
				}
			}
		}
	}

	// Решение системы LX = B; L - верхнетреугольная матрица, X,B - подходящие прямоугольные матрицы (одинакового размера)
	void LSolve(const float *l_ptr, float *a_ptr, const int lda, const int m, const int n) {
		// m - высота обеих матриц, n - длина искомой матрицы
		const int block_size = m, num_of_blocks = (n+m-1)/m;

	#pragma omp parallel for
		for (int it = 0; it < num_of_blocks; ++it) {
			int block_len = (it+1 == num_of_blocks) ? ((n % block_size != 0) ? n % block_size : block_size) : block_size; // длина текущего блока
			float *na_ptr = a_ptr + block_size*it, // указатель на начало текущего блока
				  *ptr1 = na_ptr,				   // указатель на текущую вычитаемую строку
				  *ptr2,						   // указатель на текущую уменьшаемую строку
				   mult;						   // множитель, на который умножается вычитаемая строка

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
	void USolve(const float *u_ptr, float *a_ptr, const int lda, const int m, const int n) {
		// m - высота искомой матрицы, n - длина обеих матриц
		const int block_size = n, num_of_blocks = (m + n - 1)/n;

	#pragma omp parallel for
		for (int it = 0; it < num_of_blocks; ++it) {
			int block_len = (it + 1 == num_of_blocks) ? ((m % block_size != 0) ? m % block_size : block_size) : block_size; // высота текущего блока
			const float *u_curr;					   // указатель на текущую строку верхнетреугольной матрицы
			float *na_ptr = a_ptr + block_size*it*lda, // указатель на начало блока
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

	float* data;
	int m, n; // Число строк и столбцов
	sycl::queue q{cpu_selector{}, property::queue::in_order()};
};

void check_correct(const Matrix &M1, const Matrix &M2, int m, int n) {
	TYPE *ptr1 = M1.get_ptr(), *ptr2 = M2.get_ptr();
	TYPE ae = 0.0, re = 0.0;

#pragma omp parallel for
	for (int i = 0; i < m*n; ++i) {
		ae = std::max(ae, abs(ptr2[i] - ptr1[i]));
		re = std::max(re, (ptr1[i] == 0.0f) ? 0.0f : abs(ptr2[i]/ptr1[i] - 1.0f));
	}
	cout << "Absolute error: " << ae << '\n' << "Relative error: " << re << std::endl;
}