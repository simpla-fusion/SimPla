/*
 * simplex_distribution.h
 *
 *  Created on: 2013年10月23日
 *      Author: salmon
 */

#ifndef SIMPLEX_DISTRIBUTION_H_
#define SIMPLEX_DISTRIBUTION_H_
#include "utilities/ntuple.h"
#include <vector>
#include <random>
#include <numeric>
namespace simpla
{

/**
 * @brief Sampling Uniformly from the Unit Simplex
 * @ref http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
 *
 * */
template<int NDIM>
class simplex_distribution
{
public:
	template<typename ... Args>
	simplex_distribution(Args ... args) :
			pixels_(args), xn_(NDIM + 2)
	{
	}
	~simplex_distribution()
	{

	}
	template<typename ... Args>
	inline void Reset(Args ... args)
	{
		std::swap(nTuple<NDIM + 1, nTuple<NDIM, double>>(args...), pixels_);
	}

	template<typename Generator> inline nTuple<NDIM, double> operator()(
			Generator &g) const
	{
		xn_[0] = 0;
		for (int i = 1; i < NDIM; ++i)
		{
			xn_[i] = static_cast<double>(g() - g.min())
					/ static_cast<double>(g.max() - g.min());

		}
		xn_[NDIM] = 1.0;
		std::sort(xn_.begin(), xn_.end());
		std::adjacent_difference(xn_.begin(), xn_.end(), xn_.begin());

		nTuple<NDIM, double> res;
		std::fill(&res[0], &res[NDIM], 0);

		for (int i = 1; i < NDIM + 1; ++i)
		{
			res += xn_[i] * pixels_[i];
		}
		return res;
	}
private:
	nTuple<NDIM + 1, nTuple<NDIM, double>> pixels_;
	std::vector<double> xn_;
};

}  // namespace simpla

#endif /* SIMPLEX_DISTRIBUTION_H_ */
