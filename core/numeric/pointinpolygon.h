/*
 * @file pointinpolygon.h
 *
 *  created on: 2013-12-4
 *  @author salmon
 */

#ifndef POINTINPOLYGEN_H_
#define POINTINPOLYGEN_H_

#include <vector>

#include "../utilities/ntuple.h"
#include "../utilities/primitives.h"
#include "../numeric/find_root.h"

namespace simpla
{
/**
 *  \ingroup Numeric  GeometryAlgorithm
 *  \brief check a point in 2D polygon
 */
class PointInPolygon
{

	std::vector<nTuple<double, 2> > polygen_;
	size_t num_of_vertex_;
	std::vector<double> constant_;
	std::vector<double> multiple_;
public:
	template<size_t N>
	PointInPolygon(std::vector<nTuple<double, N> > const &polygen, size_t Z = 2) :
			num_of_vertex_(0)
	{

		for (auto const & v : polygen)
		{
			polygen_.emplace_back(nTuple<double, 2>(
			{ v[(Z + 1) % 3], v[(Z + 2) % 3] }));
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
				constant_[i] = polygen_[i][0]
						- (polygen_[i][1] * polygen_[j][0])
								/ (polygen_[j][1] - polygen_[i][1])
						+ (polygen_[i][1] * polygen_[i][0])
								/ (polygen_[j][1] - polygen_[i][1]);
				multiple_[i] = (polygen_[j][0] - polygen_[i][0])
						/ (polygen_[j][1] - polygen_[i][1]);
			}
			j = i;
		}
	}

	PointInPolygon(PointInPolygon const& rhs) :
			polygen_(rhs.polygen_), num_of_vertex_(rhs.num_of_vertex_), constant_(
					rhs.constant_), multiple_(rhs.multiple_)
	{

	}
	PointInPolygon(PointInPolygon && rhs) :
			polygen_(rhs.polygen_), num_of_vertex_(rhs.num_of_vertex_), constant_(
					rhs.constant_), multiple_(rhs.multiple_)
	{

	}

	template<typename ...Args>
	inline bool operator()(Args &&... args) const
	{
		return IsInside(std::forward<Args>(args)...);
	}

	template<size_t N>
	inline bool IsInside(nTuple<double, N> x, size_t ZAxis = 2) const
	{
		return IsInside(x[(ZAxis + 1) % 3], x[(ZAxis + 2) % 3]);
	}

	inline bool IsInside(double x, double y) const
	{

		bool oddNodes = false;

		for (size_t i = 0, j = num_of_vertex_ - 1; i < num_of_vertex_; i++)
		{
			if (((polygen_[i][1] < y) && (polygen_[j][1] >= y))
					|| ((polygen_[j][1] < y) && (polygen_[i][1] >= y)))
			{
				oddNodes ^= (y * multiple_[i] + constant_[i] < x);
			}

			j = i;
		}

		return oddNodes;
	}

	template<size_t N>
	std::tuple<bool, nTuple<double, N>> Intersection(
			nTuple<double, N> const & x0, nTuple<double, N> const &x1,
			size_t ZAxis = 2, double error = 0.001) const
	{
		std::function<double(nTuple<double, N> const &)> fun =
				[this,ZAxis](nTuple<double, N> const & x)->bool
				{
					return this->IsInside(x,ZAxis)?1:0;
				};

		return std::move(find_root(x0, x1, fun, error));
	}
}
;
}
// namespace simpla

#endif /* POINTINPOLYGEN_H_ */
