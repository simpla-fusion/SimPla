/*
 * pointinpolygen.h
 *
 *  Created on: 2013年12月4日
 *      Author: salmon
 */

#ifndef POINTINPOLYGEN_H_
#define POINTINPOLYGEN_H_

#include <cstddef>
#include <vector>

#include "../fetl/ntuple.h"
#include "../fetl/primitives.h"

namespace simpla
{

class PointInPolygen
{

	std::vector<nTuple<2, double> > const &polygen_;
	size_t num_of_vertex_;
	std::vector<double> constant_;
	std::vector<double> multiple_;
public:
	PointInPolygen(std::vector<nTuple<2, Real> > const &polygen)
			: polygen_(polygen), num_of_vertex_(polygen.size()), constant_(num_of_vertex_), multiple_(num_of_vertex_)
	{

		for (size_t i = 0, j = num_of_vertex_ - 1; i < num_of_vertex_; i++)
		{
			if (polygen_[j][1] == polygen_[i][1])
			{
				constant_[i] = polygen_[i][0];
				multiple_[i] = 0;
			}
			else
			{
				constant_[i] = polygen_[i][0] - (polygen_[i][1] * polygen_[j][0]) / (polygen_[j][1] - polygen_[i][1])
				        + (polygen_[i][1] * polygen_[i][0]) / (polygen_[j][1] - polygen_[i][1]);
				multiple_[i] = (polygen_[j][0] - polygen_[i][0]) / (polygen_[j][1] - polygen_[i][1]);
			}
			j = i;
		}
	}

	inline bool operator()(Real x, Real y) const
	{

		bool oddNodes = false;

		for (size_t i = 0, j = num_of_vertex_ - 1; i < num_of_vertex_; i++)
		{
			if (((polygen_[i][1] < y) && (polygen_[j][1] >= y)) || ((polygen_[j][1] < y) && (polygen_[i][1] >= y)))
			{
				oddNodes ^= (y * multiple_[i] + constant_[i] < x);
			}

			j = i;
		}

		return oddNodes;
	}
}
;
}
// namespace simpla

#endif /* POINTINPOLYGEN_H_ */
