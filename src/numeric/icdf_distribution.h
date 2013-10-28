/*
 * icdf_distribution.h
 *
 *  Created on: 2013年10月23日
 *      Author: salmon
 */

#ifndef ICDF_DISTRIBUTION_H_
#define ICDF_DISTRIBUTION_H_
#include "utilities/log.h"
namespace simpla
{

class icdf_distribution
{

public:

	template<typename Fun>
	icdf_distribution(double min, double max, Fun const &pdf,
			size_t num_of_bins = 1025) :
			xmin_(min), xmax_(max)
	{
		double dx = (xmax_ - xmin_) / static_cast<double>(num_of_bins - 1);
		double sum = 0;
		double x = xmin_;

		std::map<double, double> cdf;
		for (size_t i = 0; i < num_of_bins; ++i)
		{
			sum += pdf(x);
			cdf[x] = sum;
			x += dx;
		}
		invers_cdf(cdf);
	}

	icdf_distribution(std::map<double, double> const &fun, bool is_cdf = true) :
			xmin_(fun.begin()->first), xmax_(fun.rbegin()->first)
	{

		if (is_cdf)
		{
			invers_cdf(fun);
		}
		else
		{
			std::map<double, double> cdf;
			// integral

			double fmin_ = fun.begin()->second;
			double fmax_ = fmin_;

			auto it = fun.begin();

			cdf[fmax_] = xmin_;

			++it;
			double f = fmax_, x = xmin_;

			for (; it != fun.end(); ++it)
			{

				fmax_ += 0.5 * (it->second + f) * (it->first - x);
				x = it->first;
				f = it->second;
				cdf[x] = fmax_;
			}

			invers_cdf(cdf);
		}
	}
	~icdf_distribution()
	{
	}

	inline double min() const
	{
		return xmin_;
	}
	inline double max() const
	{
		return xmax_;
	}

	template<typename Generator>
	inline double operator()(Generator & g) const
	{
		double f = static_cast<double>(g() - g.min())
				/ static_cast<double>(g.max() - g.min());

		double res;
		auto it = icdf_.lower_bound(f);
		if (it == icdf_.end())
		{
			ERROR << ("icdf out of boundary!!");
		}
		auto it2 = it;
		++it2;
		if (it2 != icdf_.end())
		{
			res = ((it2)->second - it->second) / ((it2)->first - it->first)
					* (f - it->first) + it->second;
		}
		else
		{
			res = icdf_.rbegin()->second;
		}
		return res * (xmax_ - xmin_) + xmin_;
	}
private:
	double xmin_, xmax_;
	std::map<double, double> icdf_;

	void invers_cdf(std::map<double, double> const &cdf)
	{
		icdf_.clear();

		double fmin_ = cdf.begin()->second;
		double fmax_ = cdf.rbegin()->second;

		xmin_ = cdf.begin()->first;
		xmax_ = cdf.rbegin()->first;

		for (auto it = cdf.begin(); it != cdf.end(); ++it)
		{
			icdf_[(it->second - fmin_) / (fmax_ - fmin_)] = (it->first - xmin_)
					/ (xmax_ - xmin_);
		}

	}

};
}  // namespace simpla

#endif /* ICDF_DISTRIBUTION_H_ */
