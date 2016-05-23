/**
 *  @file multi_normal_distribution.h
 *
 *  created on: 2013-10-22
 *      Author: salmon
 */

#ifndef MULTI_NORMAL_DISTRIBUTION_H_
#define MULTI_NORMAL_DISTRIBUTION_H_

#include <random>
#include "../gtl/ntuple.h"
#include "cholesky.h"

namespace simpla
{

/**
 * @ingroup numeric
 *
 * \brief A normal continuous distribution for random numbers.
 *
 * The formula for the normal probability density function is
 * @f[
 *     p(x|\mu,\sigma) = \frac{1}{\sigma \sqrt{2 \pi}}
 *            e^{- \frac{{x - \mu}^ {2}}{2 \sigma ^ {2}} }
 * @f]
 */
template<size_t N, typename RealType = double,
		typename TNormalGen = std::normal_distribution<RealType> >
class multi_normal_distribution
{
	nTuple<RealType, N, N> A_;
	nTuple<RealType, N> u_;
	TNormalGen normal_dist_;
	typedef multi_normal_distribution<N, RealType, TNormalGen> this_type;

public:
	static constexpr size_t ndims = N;

	multi_normal_distribution(RealType pT = 1.0, //
			nTuple<RealType, N> const &pu = { 0, 0, 0 })
			: u_(pu), normal_dist_(0, 1.0)
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

	multi_normal_distribution(nTuple<RealType, N> const & pT,
			nTuple<RealType, N> const &pu)
			: u_(pu), normal_dist_(0, 1.0)
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

	multi_normal_distribution(nTuple<RealType, N, N> const & pA,
			nTuple<RealType, N> const &pu)
			: A_(cholesky_decomposition(pA)), u_(pu), normal_dist_(0, 1.0)
	{
	}
	~multi_normal_distribution()
	{
	}

	template<typename Generator, typename T> inline
	void operator()(Generator & g, T * res)
	{
		nTuple<RealType, N> v;
		for (int i = 0; i < ndims; ++i)
		{
			v[i] = normal_dist_(g);
		}
		v = A_[0] * v[0] + A_[1] * v[1] + A_[2] * v[2];
		for (int i = 0; i < N; ++i)
		{
			res[i] = v[i];
		}
	}

	template<typename Generator> inline nTuple<RealType, N> operator()(
			Generator & g)
	{
		nTuple<RealType, N> res;
		this_type::operator()(g, &res[0]);
		return std::move(res);
	}

};

}
// namespace simpla

#endif /* MULTI_NORMAL_DISTRIBUTION_H_ */
