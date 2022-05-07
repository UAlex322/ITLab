// ����������� ������������� ��� ������::
// PARALLEL - ������������ ������������ �������� (�� ��������� - ����������������)
// BLOCK - ������������ ������� �������� (�� ��������� - �������)

//#define PARALLEL
#define BLOCK
#define MTX_TYPE float

#include <iostream>
#include <algorithm>
#include <string>
#include <limits>
#include <cstdlib>
#include <numeric>
#include <chrono>
#include <mkl.h>
#include "Matrix.h"
using namespace std;
using namespace chrono;


int main(int argc, char **argv) {
	size_t size,
		   lu_block,
		   mtx_product_block;

	// ���� ���������� �� ��������� ������:
	// argv[1] - ����� �������
	// argv[2] - ����� ����� � LU-����������
	// argv[3] - ����� ����� � ��������� ���������
	// � ��������� ��� ��������� �� ����� 'argv[2] x argv[3]'
	if (argc > 1) {
		size = stoll(argv[1]);
		
		if (argc > 2) {
			lu_block = stoll(argv[2]);
			mtx_product_block = stoll(argv[3]);
		}
		else
			lu_block = mtx_product_block = 128;
	}
	else
		size = 8192;


	cout << "Sizes: (" << size << ',' << lu_block << ',' << mtx_product_block << "), ";

	Matrix<MTX_TYPE> A(size, size);
	A.generate_well_conditioned_matrix((float)size * 0.25, 0.04);

	auto start_time = steady_clock::now();
	{
	#if !defined(BLOCK) && !defined(PARALLEL)
		A.lu_trivial_sequential();
	#elif !defined(BLOCK) && defined(PARALLEL)
		//A.lu_trivial_parallel_omp();
	#elif defined(BLOCK)
		A.lu_block(lu_block, mtx_product_block);
	#endif
	}
	auto end_time = steady_clock::now();

	cout << "time: " << duration_cast<milliseconds>(end_time-start_time).count() << " ms" << endl << endl;

	return 0;
}