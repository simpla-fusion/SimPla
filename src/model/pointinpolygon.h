/*
 * @file pointinpolygon.h
 *
 *  Created on: 2013年12月4日
 *  @author salmon
 */

#ifndef POINTINPOLYGEN_H_
#define POINTINPOLYGEN_H_


#include <vector>

#include "../utilities/ntuple.h"
#include "../utilities/primitives.h"

namespace simpla
{
/**
 *  \ingroup Algorithm  Geometry
 *  \brief check a point in 2D polygon
 */
class PointInPolygon
{

	std::vector<nTuple<2, double> > polygen_;
	size_t num_of_vertex_;
	std::vector<double> constant_;
	std::vector<double> multiple_;
	const int Z_;
public:
	template<int N>
	PointInPolygon(std::vector<nTuple<N, Real> > const &polygen, unsigned int Z = 2)
			: num_of_vertex_(0), Z_(Z)
	{

		for (auto const & v : polygen)
		{
			polygen_.emplace_back(nTuple<2, Real>( { v[(Z + 1) % 3], v[(Z + 2) % 3] }));
		}
		num_of_vertex_ = polygen_.size();
		constant_.resize(num_of_vertex_);
		multiple_.resize(num_of_vertex_);

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

	PointInPolygon(PointInPolygon const& rhs)
			: polygen_(rhs.polygen_), num_of_vertex_(rhs.num_of_vertex_), constant_(rhs.constant_), multiple_(
			        rhs.multiple_), Z_(rhs.Z_)
	{

	}
	PointInPolygon(PointInPolygon && rhs)
			: polygen_(rhs.polygen_), num_of_vertex_(rhs.num_of_vertex_), constant_(rhs.constant_), multiple_(
			        rhs.multiple_), Z_(rhs.Z_)
	{

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
	template<int N>
	inline bool operator()(nTuple<N, Real> x) const
	{
		return this->operator()(x[(Z_ + 1) % 3], x[(Z_ + 2) % 3]);
	}
}
;
}
// namespace simpla

#endif /* POINTINPOLYGEN_H_ */
