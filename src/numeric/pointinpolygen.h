/*
 * pointinpolygen.h
 *
 *  Created on: 2013年12月4日
 *      Author: salmon
 */

#ifndef POINTINPOLYGEN_H_
#define POINTINPOLYGEN_H_

namespace simpla
{

template<typename TV>
class PointInPolygen
{

	std::vector<TV> const &polygen_;
	size_t num_of_vertex_;
	std::vector<double> constant_;
	std::vector<double> multiple_;
public:
	PointInPolygen(std::vector<TV> const &polygen) :
			polygen_(polygen), num_of_vertex_(polygen.size()), constant_(
					num_of_vertex_), multiple_(num_of_vertex_)
	{
	}
	inline void precalc_values()
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

	template<typename T>
	inline bool operator()(T const & x) const
	{
		return operator()(x[0], x[1]);
	}

	template<typename ... Args>
	inline bool operator()(Real x, Real y, Args const &... args) const
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
}
;
}
// namespace simpla

#endif /* POINTINPOLYGEN_H_ */
