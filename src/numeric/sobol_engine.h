/*
 * sobol_engine.h
 *
 *  Created on: 2013年10月22日
 *      Author: salmon
 */

#ifndef SOBOL_ENGINE_H_
#define SOBOL_ENGINE_H_
#include "fetl/ntuple.h"
#include "utilities/log.h"
namespace simpla
{
/**
 * @brief Numerical Recipes 3rd pp.403, 7.8 Quasi- (that is,
 *  Sub-) Random Sequences
 *
 * @ref: http://web.maths.unsw.edu.au/~fkuo/sobol/index.html
 *  - S. Joe and F. Y. Kuo, Remark on Algorithm 659:
 *     Implementing Sobol's quasirandom sequence generator,
 *     ACM Trans. Math. Softw. 29, 49-57 (2003).
 *  - S. Joe and F. Y. Kuo, Constructing Sobol sequences with
 *      better two-dimensional projections, SIAM J. Sci. Comput.
 *      30, 2635-2654 (2008). Link to paper.
 *
 *
 *
 *
		d       s       a       m_i
		2       1       0       1
		3       2       1       1 3
		4       3       1       1 3 1
		5       3       2       1 1 1
		6       4       1       1 1 3 3
		7       4       4       1 3 5 13
		8       5       2       1 1 5 5 17
		9       5       4       1 1 5 5 5
		10      5       7       1 1 7 11 19
		11      5       11      1 1 5 1 1
		12      5       13      1 1 1 3 11
		13      5       14      1 3 5 5 31
		14      6       1       1 3 3 9 7 49
		15      6       13      1 1 1 15 21 21
		16      6       16      1 3 1 13 27 49
		17      6       19      1 1 1 15 7 5
		18      6       22      1 3 1 15 13 25
		19      6       25      1 1 5 5 19 61
		20      7       1       1 3 7 11 23 15 103
 * */
template<int N, typename T = unsigned int>
class sobol_engine
{
	typedef T result_type;

	const int MAXDIM = 6;
	const int MAXBIT = sizeof(result_type) * 8;

	const int mdeg[MAXDIM] =
	{ 1, 2, 3, 3, 4, 4 };

	result_type ix[MAXDIM];
	result_type* iu[MAXBIT];
	const result_type ip[MAXDIM] = //MAXDIM
			{ 0, 1, 1, 2, 1, 4 };
	result_type iv[MAXDIM * MAXBIT] = //MAXDIM*MAXBIT
			{ 1, 1, 1, 1, 1, 1, 3, 1, 3, 3, 1, 1, 5, 7, 7, 3, 3, 5, 15, 11, 5,
					15, 13, 9 };

	result_type in;

	size_t seed_;
	size_t count_;
public:
	sobol_engine() :
			count_(0), seed_(0)
	{

		result_type i, im, ipp;

//	sobseq.h
		for (int k = 0; k < N; ++k)
			ix[k] = 0;
		in = 0;
		if (iv[0] != 1)
			return;
		for (int j = 0, k = 0; j < MAXBIT; ++j, k += MAXDIM)
			iu[j] = &iv[k];
		for (int k = 0; k < MAXDIM; ++k)
		{
			for (int j = 0; j < mdeg[k]; ++j)
				iu[j][k] <<= (MAXBIT - 1 - j);
			for (int j = mdeg[k]; j < MAXBIT; ++j)
			{
				ipp = ip[k];
				i = iu[j - mdeg[k]][k];
				i ^= (i >> mdeg[k]);
				for (int l = mdeg[k] - 1; l >= 1; --l)
				{
					if (ipp & 1)
						i ^= iu[j - l][k];
					ipp >>= 1;
				}
				iu[j][k] = i;
			}
		}
	}

	~sobol_engine()
	{
	}
	result_type max() const
	{
		return 1 << MAXBIT;

	}
	result_type min() const
	{
		return 0;
	}

	inline void sed(size_t s)
	{
		seed_ = s % N;
	}
	inline void discard(size_t u)
	{
		in += u;
	}
	result_type operator()()
	{

// Calculate the next vector in the sequence.  Find the rightmost zero bit
		if (count_ % N == 0)
		{
			result_type im = in;

			++in;

			int j;
			for (j = 0; j < MAXBIT; ++j)
			{
				if (!(im & 1))
					break;
				im >>= 1;
			}

			if (j >= MAXBIT)
				throw("MAXBIT too small in sobseq");
			im = j * N;
			for (int k = 0; k < N; ++k)
			{
//	XOR the appropriate direction number into each component of the vector and convert to a floating
				ix[k] ^= iv[im + k];
			}
		}
		++count_;

		return ix[(count_ + seed_ - 1) % N];
	}
}
;
}
// namespace simpla

#endif /* SOBOL_ENGINE_H_ */
