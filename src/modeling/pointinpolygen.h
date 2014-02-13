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
	const int X, Y;
public:
	PointInPolygen(std::vector<TV> const &polygen, int Z) :
			polygen_(polygen), num_of_vertex_(polygen.size()), constant_(
					num_of_vertex_), multiple_(num_of_vertex_), X((Z + 1) % 3), Y(
					(Z + 2) % 3)

	{
		precalc_values();
	}
	inline void precalc_values()
	{

		for (size_t i = 0, j = num_of_vertex_ - 1; i < num_of_vertex_; i++)
		{
			if (polygen_[j][Y] == polygen_[i][Y])
			{
				constant_[i] = polygen_[i][X];
				multiple_[i] = 0;
			}
			else
			{
				constant_[i] = polygen_[i][X]
						- (polygen_[i][Y] * polygen_[j][X])
								/ (polygen_[j][Y] - polygen_[i][Y])
						+ (polygen_[i][Y] * polygen_[i][X])
								/ (polygen_[j][Y] - polygen_[i][Y]);
				multiple_[i] = (polygen_[j][X] - polygen_[i][X])
						/ (polygen_[j][Y] - polygen_[i][Y]);
			}
			j = i;
		}
	}

	template<typename T>
	inline bool operator()(nTuple<3, T> const & x) const
	{
		return operator()(x[X], x[Y]);
	}

	template<typename ... Args>
	inline bool operator()(Real x, Real y, Args const &... args) const
	{

		bool oddNodes = false;

		for (size_t i = 0, j = num_of_vertex_ - 1; i < num_of_vertex_; i++)
		{
			if (((polygen_[i][Y] < y) && (polygen_[j][Y] >= y))
					|| ((polygen_[j][Y] < y) && (polygen_[i][Y] >= y)))
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
