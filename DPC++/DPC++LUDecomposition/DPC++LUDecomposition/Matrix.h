#pragma once
#include <iostream>
#include <CL/sycl.hpp>
using namespace sycl;

class Matrix {
public:

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
		m = 0; n = 0;
	}

	void generate_random_matrix() {
		srand(time(0));

		for (int i = 0; i < m; ++i)
			for (int j = 0; j < n; ++j)
				data[i*n+j] = (rand() - 16384) / 2048.0;
	}

	void generate_well_conditioned_matrix(const float &mean_value) {
		srand(time(0));

		for (int i = 0; i < m; ++i)
			for (int j = 0; j < n; ++j) {
				data[i*n+j] = (rand() - 16384) / 2048.0;
			}
		for (int i = 0; i < m; ++i)
			data[i*(m+1)] = (rand() % 2 ? -1 : 1) * (mean_value + (rand() % 10000) * 0.0001);
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

	void print() {
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
	void LU2_sequential() {
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
	void LU2_parallel() {
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
	void LU3_block(const int ls, const int mbs) {
		float *dptr = data;
		const int bs = ls;		// размер блока
		int bi = 0, nbi = bs;	// индексы начала текущего и следующего блоков

		float *buffer = new float[bs*n]; // буфер для блоков матрицы B в матричном умножении
		sycl::queue q{cpu_selector{}}; // sycl::queue для параллелизма

		//std::cout << "Max workgroup size: " << q.get_device().get_info<sycl::info::device::max_work_group_size>() << std::endl;
		//std::cout << "Max workitem dim: " << q.get_device().get_info<info::device::max_work_item_dimensions>() << std::endl;
		for (; nbi < n; dptr += bs*(n+1), bi += bs, nbi += bs) {
			//std::cout << "Matrix size: " << n << 'x' << n << ", block size: " << mbs << 'x' << bs << std::endl;
			LU(dptr, n, bs);

			LSolve(dptr, dptr + bs, n, bs, n - nbi);
			USolve(dptr, dptr + bs*n, n, n - nbi, bs);

			std::cout << n - nbi << std::endl;
			FMMS2(dptr + bs*n, dptr + bs, dptr + bs*(n + 1), buffer, n, n, n, n - nbi, bs, n - nbi, bs, mbs, q);
			//cblas_dgemm(CblasRowMajor, CblasNofloatrans, CblasNofloatrans, n-nbi, bs, n-nbi, -1.0, dptr + bs*n, n, dptr + bs, n, 1.0, dptr + bs*(n+1), n);
		}
		LU(dptr, n, n-bi);

		delete[] buffer;
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

	// Оптимизированная версия FMMS с транспонированием блоков
	void FMMS2(float *a_ptr, float *b_ptr, float *c_ptr, float *buf, const int lda, const int ldb, const int ldc, const int m, const int n, const int p, const int bs, const int mbs, sycl::queue &q) {
		//std::cout << "Call FMMS" << std::endl;
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < p; ++j)
				buf[j*n + i] = b_ptr[i*ldb + j];
		{
			buffer<float,1> buf1(a_ptr, range<1>{m*lda-lda+m}); // буфер для подматриц A и C
			buffer<float,1> buf2(buf, range<1>{n*p}); // буфер для B^T

			q.submit([&](handler &cgh) {
				auto a_acc = buf1.get_access<sycl::access::mode::read_write>(cgh); // Матрица С
				auto b_acc = buf2.get_access<sycl::access::mode::read>(cgh); // Матрица B
				accessor<float, 1, access::mode::read_write, access::target::local> block_a_acc(mbs*bs, cgh); // Матрица A

				cgh.parallel_for<class Mult>(nd_range<2>(range<2>(m,p), range<2>(mbs,mbs)), [=](nd_item<2> item) {
						int a_old_shift = item.get_global_id(0)*lda,
							a_new_shift = item.get_local_id(0)*bs,
							b_shift		= item.get_global_id(1)*bs;

					for (int j = 0; j < bs; ++j)
						block_a_acc[a_new_shift + j] = a_acc[a_old_shift + j];
					item.barrier(access::fence_space::local_space);

					float sum = 0.0f;
					for (int j = 0; j < bs; ++j)
						sum += block_a_acc[a_new_shift + j] * b_acc[b_shift + j];
					a_acc[a_old_shift + bs + item.get_global_id(1)] -= sum;
				});
			}).wait();
			
			
			//std::cout << "Call submit" << std::endl;
			/*
			q.submit([&](handler &cgh) {
				auto a_acc = buf1.get_access<sycl::access::mode::read_write>(cgh);
				auto b_acc = buf2.get_access<sycl::access::mode::read>(cgh);

				//std::cout << "Call parallel_for" << std::endl;
				cgh.parallel_for(range<2>(m,p), [=](item<2> item) {
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

			
			
			q.submit([&](handler &cgh) {
				auto a_acc = buf1.get_access<sycl::access::mode::read_write>(cgh);
				auto b_acc = buf2.get_access<sycl::access::mode::read>(cgh);
				accessor<float, 2, access::mode::read_write, access::target::local> buffer(range(bs,mbs),cgh);
				//std::cout << "Call parallel_for" << std::endl;
				cgh.parallel_for(nd_range<2>(range<2>(m,p), range<2>(mbs,mbs)), [=](nd_item<2> item) {
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
			
			
			

			//std::cout << "End of submit" << std::endl;
		}
		//std::cout << "End of FMMS" << std::endl;
	}

	// Без транспонирования
	void FMMS1(const float *a_ptr, const float *b_ptr, float *c_ptr, const int lda, const int ldb, const int ldc, const int m, const int n, const int p, const int bs) {
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
				const float *na_ptr = a_ptr + pos[ib]*lda,
						*nb_ptr = b_ptr + pos[jb];
					  float	*nc_ptr = c_ptr + pos[ib]*ldc + pos[jb],
					a_elem;

				for (int i = 0; i < ba_len; ++i, nc_ptr += ldc, na_ptr += lda) {
					const float *b_curr = nb_ptr;

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
	void LU(float *a_ptr, const int lda, const int n) {
		float *ptr1 = a_ptr, // указатель на текущую вычитаемую строку 
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
	void LSolve(const float *l_ptr, float *a_ptr, const int lda, const int m, const int n) {
		// m - высота обеих матриц, n - длина искомой матрицы
		const int block_size = m, num_of_blocks = (n+m-1)/m;

	#pragma omp parallel for
		for (int it = 0; it < num_of_blocks; ++it) {
			int block_len = (it+1 == num_of_blocks) ? ((n % block_size != 0) ? n % block_size : block_size) : block_size; // длина текущего блока
			float *na_ptr = a_ptr + block_size*it, // указатель на начало текущего блока
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


	// 
	float* data;
	int m, n; // Число строк и столбцов
};