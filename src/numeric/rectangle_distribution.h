/*
 * rectangle_distribution.h
 *
 *  created on: 2013-10-24
 *      Author: salmon
 */

#ifndef RECTANGLE_DISTRIBUTION_H_
#define RECTANGLE_DISTRIBUTION_H_

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>
#include "../utilities/ntuple.h"
namespace simpla
{

template<unsigned int NDIM>
class rectangle_distribution
{

public:

	rectangle_distribution()
	{
		nTuple<NDIM, double> xmin, xmax;

		for (int i = 0; i < NDIM; ++i)
		{
			xmin[i] = 0;
			xmax[i] = 1;
		}
		Reset(xmin, xmax);
	}
	rectangle_distribution(nTuple<NDIM, double> const &xmin, nTuple<NDIM, double> const & xmax)
	{
		Reset(xmin, xmax);
	}

	template<typename TRANGE>
	rectangle_distribution(TRANGE const &xrange)
	{
		Reset(xrange);
	}
	~rectangle_distribution()
	{
	}

	template<typename TR>
	inline void Reset(TR const &xrange)
	{
		Reset(std::get<0>(xrange), std::get<1>(xrange));
	}

	inline void Reset(nTuple<NDIM, double> const *xrange)
	{
		Reset(xrange[0], xrange[1]);
	}

	inline void Reset(nTuple<NDIM, double> const &xmin, nTuple<NDIM, double> const & xmax)
	{
		xmin_ = xmin;
		xmax_ = xmax;

		for (int i = 0; i < NDIM; ++i)
		{

			if (xmax_[i] > xmin_[i]) l_[i] = (xmax_[i] - xmin_[i]);
			else l_[i] = 0;
		}
	}

	template<typename Generator>
	nTuple<NDIM, double> operator()(Generator &g)
	{
		nTuple<NDIM, double> res;

		for (int i = 0; i < NDIM; ++i)
		{
			res[i] = static_cast<double>(g() - g.min()) / static_cast<double>(g.max() - g.min()) * l_[i] + xmin_[i];
		}

		return std::move(res);

	}

	template<typename Generator, typename T>
	void operator()(Generator &g, T* res)
	{

		for (int i = 0; i < NDIM; ++i)
		{
			res[i] = static_cast<double>(g() - g.min()) / static_cast<double>(g.max() - g.min()) * l_[i] + xmin_[i];
		}
	}
private:
	nTuple<NDIM, double> xmin_;
	nTuple<NDIM, double> xmax_;
	nTuple<NDIM, double> l_;

};

}  // namespace simpla

#endif /* RECTANGLE_DISTRIBUTION_H_ */
