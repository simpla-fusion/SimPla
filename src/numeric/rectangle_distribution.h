/*
 * rectangle_distribution.h
 *
 *  Created on: 2013年10月24日
 *      Author: salmon
 */

#ifndef RECTANGLE_DISTRIBUTION_H_
#define RECTANGLE_DISTRIBUTION_H_

#include <fetl/ntuple.h>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

namespace simpla
{

template<int NDIM>
class rectangle_distribution
{

public:

	rectangle_distribution()
	{
	}
	rectangle_distribution(nTuple<NDIM, double> const &xmin,
			nTuple<NDIM, double> const & xmax) :
			xmin_(xmin), xmax_(xmax)
	{
	}

	template<typename TRANGE>
	rectangle_distribution(TRANGE const &xrange) :
			xmin_(xrange[0]), xmax_(xrange[1])
	{
	}
	~rectangle_distribution()
	{

	}

	template<typename TR>
	inline void Reset(TR const &xrange)
	{
		xmin_ = xrange[0];
		xmax_ = xrange[1];
	}

	inline void Reset(nTuple<NDIM, double> const &xmin,
			nTuple<NDIM, double> const & xmax)
	{
		xmin_ = xmin;
		xmax_ = xmax;
	}

	template<typename Generator>
	nTuple<NDIM, double> operator()(Generator &g)
	{
		nTuple<NDIM, double> res;

		for (int i = 0; i < NDIM; ++i)
		{
			res[i] = static_cast<double>(g() - g.min())
					/ static_cast<double>(g.max() - g.min())
					* (xmax_[i] - xmin_[i]) + xmin_[i];
		}
		return std::move(res);

	}

	template<typename Generator, typename T>
	void operator()(Generator &g, T& res)
	{
		for (int i = 0; i < NDIM; ++i)
		{
			res[i] = static_cast<double>(g() - g.min())
					/ static_cast<double>(g.max() - g.min())
					* (xmax_[i] - xmin_[i]) + xmin_[i];
		}
	}
private:
	nTuple<NDIM, double> xmin_;
	nTuple<NDIM, double> xmax_;
};

}  // namespace simpla

#endif /* RECTANGLE_DISTRIBUTION_H_ */
