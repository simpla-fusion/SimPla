/*
 * icdf_distribution.h
 *
 *  created on: 2013-10-23
 *      Author: salmon
 */

#ifndef ICDF_DISTRIBUTION_H_
#define ICDF_DISTRIBUTION_H_

#include "interpolation.h"
#include "inverse_function.h"
#include "integration.h"
namespace simpla
{

template<typename TX, typename TY, typename TIntepolation>
inline TIntepolation MakeICDF(std::map<double, double> const &fun)
{
	auto res = Inverse(Integrate(fun));
	auto f = res.rbegin()->second;
	for (auto & p : res)
	{
		p.second /= f;
	}
	return std::move(TIntepolation(std::move(res)));
}

}
// namespace simpla

#endif /* ICDF_DISTRIBUTION_H_ */
