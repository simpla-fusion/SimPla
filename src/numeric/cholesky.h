/*
 * cholesky.h
 *
 *  Created on: 2013年10月22日
 *      Author: salmon
 */

#ifndef CHOLESKY_H_
#define CHOLESKY_H_
#include "fetl/ntuple.h"
namespace simpla
{

template<int N, typename T>
Matrix<N, T> && cholesky_decomposition(Matrix<N, T> &a)
{
	/**
	 *  Constructor.Given a positive-definite symmetric matrix
	 *  a[0..n - 1][0..n - 1], construct and store its Cholesky
	 *  decomposition, $A= L \cdot  L^T$.
	 */
	Matrix<N, T> el;

	for (int i = 0; i < N; ++i)
		for (int j = i; j < N; ++j)
		{
			T sum = el[i][j];
			for (int k = i - 1; k >= 0; --k)
			{
				sum -= el[i][k] * el[j][k];
			}
			if (i == j)
			{
				if (sum <= 0.0) // A, with rounding errors, is not positive-definite.
					throw("Cholesky failed");

				el[i][i] = sqrt(sum);
			}
			else
			{
				el[j][i] = sum / el[i][i];
			}
		}

	for (int i = 0; i < N; ++i)
		for (int j = 0; j < i; ++j)
		{
			el[j][i] = 0.;
		}

	return std::move(el);
}

}  // namespace simpla

#endif /* CHOLESKY_H_ */
