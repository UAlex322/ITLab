#pragma once
#include <iostream>
#include <random>
#include <CL/sycl.hpp>
#include <omp.h>
using namespace sycl;
using namespace std;

class Matrix {
public:

	friend void check_correct(const Matrix &M1, const Matrix &M2);
	
	Matrix(int rows = 0, int cols = 0, const sycl::queue &queue = sycl::queue{cpu_selector{}}): m(rows), n(cols), q(queue) {
		data = new float[m*n];
	}

	Matrix(const Matrix &mtx): m(mtx.m), n(mtx.n), q(mtx.q) {
		data = new float[m*n];
		for (int i = 0; i < m*n; ++i)
			data[i] = mtx.data[i];
	}

	Matrix& operator=(const Matrix &mtx) {
		if (this == &mtx) return *this;

		q = mtx.q;
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

	void generate_well_conditioned_matrix(const float avg_diag_val, const float max_deviation) {
		mt19937 gen(random_device{}());
		uniform_real_distribution<float> d(-max_deviation, max_deviation);

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

	MTX_TYPE* get_ptr() const {
		return data;
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
		float *ptr1 = data;

		for (int j = 0; j < n-1; ++j, ptr1 += n) {
		#pragma omp parallel for
			for (int i = j+1; i < n; ++i) {
				float *ptr2 = data + i*n;
				float mult = ptr2[j]/ptr1[j];
				ptr2[j] = mult;

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

		buffer<float,1> buf(dptr, range<1>{n*n});

		for (; nbi < n; dptr += bs*(n+1), bi += bs, nbi += bs) {
			LU(buf, n, bs, bi);
			LSolve(buf, n, bs, n-nbi, bi);
			USolve(buf, n, n - nbi, bs, bi);

			//std::cout << n - nbi << std::endl;
			FMMS(buf, n, n-nbi, bi, bs, mbs);
		}
		LU(buf, n, n-bi, bi);
	}


private:
	// Общий алгоритм умножения матриц 
	void Mult(const float* a_ptr, const float* b_ptr, float *c_ptr, const int lda, const int ldb, const int ldc, const int m, const int n, const int p) {
		//int bp, cp;

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
	
	void FMMS(buffer<float,1> &buf, const int lda, const int size, const int shift, const int bs, const int mbs) {
		sycl::event event = q.submit([&](handler &cgh) {

			//sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> buffer(2*bs*bs, cgh);
			sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> block_a(bs*bs, cgh),
																								  block_b(bs*bs, cgh);

			auto a = buf.get_access<sycl::access::mode::read_write>(cgh, range<1>{lda*lda}, id<1>(shift*(lda+1) + bs*lda));
			auto b = buf.get_access<sycl::access::mode::read_write>(cgh, range<1>{lda*lda}, id<1>{shift*(lda+1) + bs});

			cgh.parallel_for<class Mult>(nd_range<2>(range<2>(size,size), range<2>(mbs,mbs)), [=](nd_item<2> item) {
				//float* block_a = buffer.get_pointer();
				//float* block_b = block_a + bs*bs;

				int li = item.get_local_id(0);			//локальный индекс в группе (строка)						
				int lj = item.get_local_id(1);
				int gi = item.get_global_id(0);
				int gj = item.get_global_id(1);

				block_a[li*bs + lj] = a[gi*lda + lj];
				block_b[li*bs + lj] = b[li*lda + gj];
				item.barrier(sycl::access::fence_space::local_space);

				float sum = 0.0f;
				for (int k = 0; k < bs; ++k) {
					sum += block_a[li*bs + k] * block_b[k*bs + lj];
				}
				a[gi*lda + bs + gj] -= sum;
				item.barrier(sycl::access::fence_space::global_space);
			});
		});
		event.wait();
	}

	void LU(buffer<float,1> &buf, const int lda, const int bs, const int shift) {
		for (int j = 0; j < bs-1; ++j) {

			q.submit([&](handler &cgh) {
				auto a = buf.get_access<access::mode::read_write>(cgh, range<1>{lda*lda - (shift+j)*(lda+1)}, id<1>{(shift+j)*(lda+1)});

				cgh.parallel_for(range<1>(bs-1-j), [=](item<1> it) {
					a[(1+it.get_id(0))*lda] /= a[0];
				});
			}).wait();

			q.submit([&](handler &cgh) {
				auto a = buf.get_access<access::mode::read_write>(cgh, range<1>{lda*lda - (shift+j)*(lda+1)}, id<1>{(shift+j)*(lda+1)});

				cgh.parallel_for(range<2>(bs-1-j, bs-1-j), [=](item<2> it) {
					a[(1+it.get_id(0))*lda + 1+it.get_id(1)] -= a[(1+it.get_id(0))*lda] * a[1+it.get_id(1)];
				});
			}).wait();

		}
	}

	void LSolve(buffer<float,1> &buf, const int lda, const int bs, const int n, const int shift) {
		
		q.submit([&](handler &cgh) {
			auto l = buf.get_access<access::mode::read_write>(cgh, range<1>{bs*lda}, id<1>{shift*(lda+1)});
			auto a = buf.get_access<access::mode::read_write>(cgh, range<1>{bs*lda}, id<1>{shift*(lda+1) + bs});

			cgh.parallel_for(nd_range<1>{range<1>{n}, range<1>{bs}}, [=](nd_item<1> it) {
				int col_shift = it.get_global_id();

				for (int j = 0; j < bs-1; ++j) {
					for (int i = j+1; i < bs; ++i)
						a[col_shift + i*lda] -= l[i*lda + j] * a[col_shift + j*lda];
				}
			});
		});
	}

	void USolve(buffer<float,1> &buf, const int lda, const int m, const int bs, const int shift) {
		
		q.submit([&](handler &cgh) {
			auto u = buf.get_access<access::mode::read_write>(cgh, range<1>{bs*lda}, id<1>{shift*(lda+1)});
			auto a = buf.get_access<access::mode::read_write>(cgh, range<1>{m*lda}, id<1>{shift*(lda+1) + bs*lda});

			cgh.parallel_for(nd_range<1>{range<1>{m}, range<1>{bs}}, [=](nd_item<1> it) {
				int row_shift = it.get_global_id()*lda;

				for (int i = 0; i < bs-1; ++i)
					for (int j = i+1; j < bs; ++j)
						a[row_shift + j] -= u[i*lda + j] / u[i*(lda+1)] * a[row_shift + i];
			});
		});

		q.submit([&](handler &cgh) {
			auto u = buf.get_access<access::mode::read_write>(cgh, range<1>{bs*lda}, id<1>{shift*(lda+1)});
			auto a = buf.get_access<access::mode::read_write>(cgh, range<1>{m*lda}, id<1>{shift*(lda+1) + bs*lda});

			cgh.parallel_for(nd_range<1>{range<1>{m}, range<1>{bs}}, [=](nd_item<1> it) {
				int row_shift = it.get_global_id()*lda;

				for (int i = 0; i < bs; ++i)
					a[row_shift + i] /= u[i*(lda+1)];
			});
		});
	}

	// DATA

	float* data;
	int m, n; // Число строк и столбцов
	sycl::queue q;
};

void check_correct(const Matrix &M1, const Matrix &M2) {
	if (M1.m != M2.m || M1.n != M2.n)
		throw "Matrices have different sizes";

	MTX_TYPE *ptr1 = M1.data, *ptr2 = M2.data;
	MTX_TYPE ae = 0.0, re = 0.0;

#pragma omp parallel for
	for (int i = 0; i < M1.m * M1.n; ++i) {
		ae = std::max(ae, abs(ptr2[i] - ptr1[i]));
		re = std::max(re, (ptr1[i] == 0.0f) ? 0.0f : abs(ptr2[i]/ptr1[i] - 1.0f));
	}
	cout << "Absolute error: " << ae << '\n' << "Relative error: " << re << std::endl;
}