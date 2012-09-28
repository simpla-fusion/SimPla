/*
 * pic_init_mkl.h
 *
 *  Created on: 2012-2-14
 *      Author: salmon
 */

#ifndef PIC_INIT_MKL_H_
#define PIC_INIT_MKL_H_
/** @file detail/initial_ramdom_load.h
 *  This is an internal header file, included by other library headers.
 *  Do not attempt to use it directly. @headername{pic.h}
 */
#include "include/simpla_defs.h"
#include <cmath>
#include <stddef.h>
#include <mkl_vsl.h>

namespace simpla
{
namespace pic
{

int dUpdateFunc(VSLStreamStatePtr stream, int* n = NULL, double dbuf[] = NULL,
		int* nmin = NULL, int* nmax = NULL, int* idx = NULL)
{
	// Function: PIC::distribute
	static double *data;
	if (n == NULL)
	{
		data = dbuf;
		return (0);
	}
	else
	{
		int num = *n;
		for (int s = 0; s < *nmax; ++s)
		{
			int i = (*idx + s) % (*n);
			dbuf[i] = data[6 * (i / 3) + 3 + i % 3];
		}
		return (*nmax);
	}
}

template<typename Point_s, typename TPool>
void RandomLoad(TPool pool)
{
	size_t num_of_elements = pool->get_num_of_elements();

	size_t buff_length = 1000;

	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_NIEDERR, 6);

	double *buff, *bufft, *buffv;
	try
	{
		buff = new double[buff_length * 6];
		bufft = new double[buff_length * 3];
		buffv = new double[buff_length * 3];
	} catch (std::bad_alloc const & error)
	{
		ERROR_BAD_ALLOC_MEMORY(buff_length * 6 * sizeof(double), error);
	}

	double a1[3];
	double C1[3 * 3];
	for (int i = 0; i < 3; ++i)
	{
		a1[i] = 0;
		for (int j = 0; j < 3; ++j)
		{
			C1[i * 3 + j] = 0;
		}
		C1[i * 3 + i] = 1.0;
	}

	for (size_t sbegin = 0; sbegin < num_of_elements; sbegin += buff_length)
	{
		size_t num =
				(sbegin + buff_length < num_of_elements) ?
						buff_length : num_of_elements - sbegin;

		vdRngUniform(VSL_METHOD_DUNIFORM_STD, stream, num * 6, buff, 0.0, 1.0);

		for (int s = 0; s < num; ++s)
		{
			bufft[s * 3 + 0] = buff[6 * s + 3];
			bufft[s * 3 + 1] = buff[6 * s + 4];
			bufft[s * 3 + 2] = buff[6 * s + 5];
		}

		VSLStreamStatePtr astream;
		dUpdateFunc(&astream, NULL, buff);
		vsldNewAbstractStream(&astream, num * 3, bufft, 0.0, 1.0, dUpdateFunc);

		vdRngGaussianMV(VSL_METHOD_DGAUSSIANMV_ICDF, // Generation method
				astream, //Pointer to the stream state structure
				num, //Number of random values to be generated
				buffv, //
				3, //Dimension d ( d ≥ 1) of output random vectors
				VSL_MATRIX_STORAGE_FULL, //Matrix storage scheme for lower triangular matrix T.
				a1, // Mean vector a of dimension d
				C1 // Elements of the lower triangular matrix passed according to the matrix T storage scheme mstorage.
				);

		vslDeleteStream(&astream);

		for (size_t s = 0; s < num; ++s)
		{
			Point_s *p = (*pool)[s + sbegin];

			p->X[0] = buff[s * 6 + 0];
			p->X[1] = buff[s * 6 + 1];
			p->X[2] = buff[s * 6 + 2];
			p->V[0] = buffv[s * 3 + 0];
			p->V[1] = buffv[s * 3 + 1];
			p->V[2] = buffv[s * 3 + 2];
		}

	}

	vslDeleteStream(&stream);

	delete[] buff;
	delete[] bufft;
	delete[] buffv;

}

} // namespace pic

} // namespace simpla

#endif /* PIC_INIT_MKL_H_ */
