/*
 * multi_normal_distribution.h
 *
 *  Created on: 2013年10月22日
 *      Author: salmon
 */

#ifndef MULTI_NORMAL_DISTRIBUTION_H_
#define MULTI_NORMAL_DISTRIBUTION_H_

#include <random>
#include "../utilities/ntuple.h"
#include "cholesky.h"

namespace simpla
{

/**
 * \brief A normal continuous distribution for random numbers.
 *
 * The formula for the normal probability density function is
 * @f[
 *     p(x|\mu,\sigma) = \frac{1}{\sigma \sqrt{2 \pi}}
 *            e^{- \frac{{x - \mu}^ {2}}{2 \sigma ^ {2}} }
 * @f]
 */
template<int N, typename RealType = double, typename TNormalGen = std::normal_distribution<double> >
class multi_normal_distribution
{
	nTuple<N, nTuple<N, RealType>> A_;
	nTuple<N, RealType> u_;
	TNormalGen normal_dist_;

	typedef multi_normal_distribution<N, RealType, TNormalGen> this_type;
public:
	static const int NDIMS = N;

	multi_normal_distribution(RealType pT = 1.0, //
			nTuple<N, RealType> const &pu =
			{ 0, 0, 0 }) :
			u_(pu), normal_dist_(0, 1.0)
	{
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
			{
				if (i == j)
				{
					A_[i][j] = std::sqrt(pT);
				}
				else
				{
					A_[i][j] = 0;
				}
			}
	}

	multi_normal_distribution(nTuple<N, RealType> const & pT, nTuple<N, RealType> const &pu) :
			u_(pu), normal_dist_(0, 1.0)
	{
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
			{
				if (i == j)
				{
					A_[i][j] = std::sqrt(pT[i]);
				}
				else
				{
					A_[i][j] = 0;
				}
			}
	}

	multi_normal_distribution(Matrix<N, RealType> const & pA, nTuple<N, RealType> const &pu) :
			A_(cholesky_decomposition(pA)), u_(pu), normal_dist_(0, 1.0)
	{
	}
	~multi_normal_distribution()
	{
	}

	template<typename Generator, typename T> inline
	void operator()(Generator & g, T * res)
	{
		nTuple<N, RealType> v;
		for (int i = 0; i < NDIMS; ++i)
		{
			v[i] = normal_dist_(g);
		}
		v = Dot(A_, v);
		for (int i = 0; i < N; ++i)
		{
			res[i] = v[i];
		}
	}

	template<typename Generator> inline nTuple<N, RealType> operator()(Generator & g)
	{
		nTuple<N, RealType> res;
		this_type::operator()(g, res);
		return std::move(res);
	}

};

}  // namespace simpla

#endif /* MULTI_NORMAL_DISTRIBUTION_H_ */
