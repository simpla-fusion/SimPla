/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * InverseCDF.h
 *
 *  Created on: 2010-10-11
 *      Author: salmon
 */

#ifndef INCLUDE_INVERSE_CDF_H_
#define INCLUDE_INVERSE_CDF_H_
#include <vector>
#include <algorithm>
#include <map>
#include <utility>
namespace simpla
{
struct ComparePair
{
	typedef std::pair<double, double> eleType;
	bool operator()(eleType const & lhs, eleType const & rhs)
	{
		return lhs.first < rhs.first;
	}
};
class InverseCDF1D
{
public:

	typedef std::map<double, double> FunType;

	FunType icdf;
	Real cdf;
	explicit InverseCDF1D(FunType const &pdf)
	{
		FunType tmp;
		// integral
		auto it = pdf.begin();
		cdf = it->second;
		Real x = it->first;
		Real f = it->second;
		tmp[f] = x;
		++it;
		while (it != pdf.end())
		{
			cdf += 0.5 * (it->second + f) * (it->first - x);
			f = it->second;
			x = it->first;
			tmp[x] = cdf;
			++it;
		}
		// normalize
		icdf.clear();
		for (auto it = tmp.begin(); it != tmp.end(); ++it)
		{
			icdf[it->second / cdf] = it->first;
		}
	}
	~InverseCDF1D()
	{
	}

	template<typename TIDX>
	void operator()(TIDX *f) const
	{
		auto it = icdf.lower_bound(f[0]);
		if (it == icdf.end())
		{
			ERROR("icdf out of boundary!!");
		}
		auto it2 = it;
		++it2;
		if (it2 != icdf.end())
		{
			f[0] = ((it2)->second - it->second) / ((it2)->first - it->first)
					* (f[0] - it->first) + it->second;
		}
		else
		{
			f[0] = icdf.rbegin()->second;
		}
	}
};
} // namespace simpla
#endif  // INCLUDE_INVERSE_CDF_H_
