#include <iostream>

template <typename T>
class Matrix {
public:

	Matrix(size_t rows = 0, size_t cols = 0): m(rows), n(cols) {
		data = new T[rows*cols];
	}

	Matrix(const Matrix &mtx): m(mtx.m), n(mtx.n) {
		data = new T[m*n];
		for (size_t i = 0; i < m*n; ++i)
			data[i] = mtx.data[i];
	}

	Matrix &operator=(const Matrix &mtx) {
		if (this == &mtx) return *this;

		if (m != mtx.m || n != mtx.n) {
			delete[] data;
			data = new T[mtx.m*mtx.n];
			m = mtx.m;
			n = mtx.n;
		}

		for (size_t i = 0; i < m*n; ++i)
			data[i] = mtx.data[i];
	}

	~Matrix() {
		delete[] data;
	}

	void fill_matrix_random_elements() {
		for (size_t i = 0; i < m; ++i)
			for (size_t j = 0; j < n; ++j)
				data[i*n+j] = (rand() - 16384) / 10000.0;
	}

	Matrix operator*(const Matrix &b) {
		Matrix<T> c(m, b.n);
		size_t p = b.n;
		T *a_ptr = data, *b_ptr = b.data, *c_ptr = c.data,
		*b_curr, *c_curr = c_ptr, a_elem;

		//for (int i = 0; i < m; ++i)
			//for (int j = 0; j < b.n; ++j) {
			//	T dp = 0;
			//	for (int k = 0; k < n; ++k)
			//		dp += (*this)(i,k) * b(k,j);
			//	c(i,j) = dp;
		//	}

		for (int i = 0; i < m; ++i, c_curr += p) {
			b_curr = b_ptr;
			for (int k = 0; k < n; ++k, b_curr += p) {
				a_elem = a_ptr[i*n + k];
				for (int j = 0; j < p; ++j)
					c_curr[j] += a_elem * b_curr[j];
			}
		}

		return c;
	}

	inline const T& operator()(int row, int col) const {
		return data[row*n + col];
	}

	inline T& operator()(int row, int col) {
		return data[row*n + col];
	}

	size_t rows() {return m;}
	size_t columns() {return n;}

	void print() {
		for (size_t i = 0; i < m; ++i) {
			for (size_t j = 0; j < n; ++j)
				std::cout << data[i*n + j] << " ";
			std::cout << '\n';
		}
	}

private:
	T* data;
	size_t m, n; // Number of rows and columns
};