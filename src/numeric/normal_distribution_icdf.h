/*
 * normal_distribution_icdf.h
 *
 *  Created on: 2013年10月23日
 *      Author: salmon
 */

#ifndef NORMAL_DISTRIBUTION_ICDF_H_
#define NORMAL_DISTRIBUTION_ICDF_H_

#include "icdf_distribution.h"
#include <cmath>
namespace simpla
{

template<typename T>
class normal_distribution_icdf: public icdf_distribution
{

public:
private:
	normal_distribution_icdf(double mean, double stddev) :
			icdf_distribution(-3.0 * stddev, 3.0 * stddev,
					[mean,stddev](double x)
					{
						return std::exp(-0.5*std::power((x-mean)/stddev,2.0)))
						/(stddev*std::sqrt(2.0*3.141592653589793));
					})
	{

	}

};

}  // namespace simpla

#endif /* NORMAL_DISTRIBUTION_ICDF_H_ */
