/**
 * @file demo_model.cpp
 *
 * @date 2015年3月16日
 * @author salmon
 */

#include <iterator>
#include <vector>
#include <algorithm>
#include <cmath>

#include "../../core/gtl/ntuple.h"
#include "../../core/gtl/primitives.h"
#include "../../core/utilities/utilities.h"
#include "../../core/io/io.h"
#include "../../core/application/application.h"
#include "../../core/application/use_case.h"
using namespace simpla;

typedef nTuple<Real, 3> coordinates_type;

void divid_points(std::vector<coordinates_type> const& points,
		std::vector<coordinates_type> * res)
{

	auto first = points.begin();

	while (first != points.end())
	{
		size_t ib = res->size();

		auto x0 = *first;

		auto second = ++first;

		if (second == points.end())
		{
			second = points.begin();
		}

		auto x1 = *second;

		for (int n = 0; n < 3; ++n)
		{
			nTuple<Real, 3> xp;

			xp = 0;

			Real dx = 1;

			Real k1 = (x1[(n + 1) % 3] - x0[(n + 1) % 3]) / (x1[n] - x0[n]);
			Real k2 = (x1[(n + 2) % 3] - x0[(n + 2) % 3]) / (x1[n] - x0[n]);

			if (x1[n] > x0[n])
			{
				dx = 1;
				xp[n] = std::floor(x0[n]) + 1;
			}
			else
			{
				dx = -1;
				xp[n] = std::floor(x0[n]);
			}
			for (; (xp[n] - x0[n]) * (x1[n] - xp[n]) >= 0; xp[n] += dx)
			{
				xp[(n + 1) % 3] = (xp[n] - x0[n]) * k1 + x0[(n + 1) % 3];
				xp[(n + 2) % 3] = (xp[n] - x0[n]) * k2 + x0[(n + 2) % 3];
				res->push_back(xp);
			}

		}

		size_t ie = res->size();

		std::sort(&(*res)[ib], &(*res)[ie],
				[&](coordinates_type const & xa,coordinates_type const & xb)->bool
				{
					return inner_product(xb-xa,x1-x0)>0;
				});

	}

}
//static constexpr Real PI = 3.1415926535;
//
//template<typename T0, typename T1, typename T2, typename T3>
//std::tuple<bool, Real, Real, Real> line_segment_intersect(T0 const & P0,
//		T1 const & P1, T2 const & Q0, T3 const & Q1)
//{
//	bool is_parallel = false;
//
//	auto u = P1 - P0;
//	auto v = Q1 - Q0;
//	auto w0 = P0 - Q0;
//
//	Real a = inner_product(u, u);
//	Real b = inner_product(v, u);
//	Real c = inner_product(v, v);
//	Real d = inner_product(u, w0);
//	Real e = inner_product(v, w0);
//
//	Real s = 0, t = 0;
//
//	if (std::abs(a * c - b * b) << EPSILON)
//	{
//		t = d / b;
//		w0 -= -t * v;
//		is_parallel = true;
//	}
//	else
//	{
//		s = (b * e - c * d) / (a * c - b * b);
//		t = (b * d - c * e) / (a * c - b * b);
//
//		w0 += s * u - t * v;
//
//	}
//
//	return std::make_tuple(is_parallel, s, t, inner_product(w0, w0));
//}
int boundary_tag(coordinates_type const & x)
{
	int res = 0;
	if (x[0] != std::floor(x[0]))
	{
		res = 0;
	}
	else if (x[1] != std::floor(x[1]))
	{
		res = 1;
	}
	else if (x[2] != std::floor(x[2]))
	{
		res = 2;
	}
	return res;
}

static constexpr Real PI = 3.14159265353893;

void divid_points2(nTuple<size_t, 3> const & dims,
		std::vector<coordinates_type> const& points,
		std::vector<coordinates_type> * res, int ZAXIS = 2)
{
	int flag[dims[0]][dims[1]][dims[2]];
	auto i0 = points.rbegin();
	auto i1 = points.begin();
	auto i2 = ++points.begin();

	coordinates_type x0, x1, x2;
	x0 = *i0;
	x1 = *i1;
	x2 = *i2;

	Real q1 = std::atan2(x1[1] - x0[1], x1[0] - x0[0]);

	Real q2 = std::atan2(x2[1] - x1[1], x2[0] - x1[0]);

	while (i1 != points.end())
	{
		coordinates_type xp, xm;

		xm[0] = std::floor(xp[0]);
		xm[1] = std::floor(xp[1]);
		xm[2] = std::floor(xp[2]);

		if (q1 == q2)
		{
			if (-PI / 4 < angle && angle < PI / 4)
			{

			}
		}
//		auto tag0 = boundary_tag(x0);
//		auto tag1 = boundary_tag(x1);
//
//		coordinates_type xp = x0;
//
//		if (tag0 == tag1)
//		{
//			switch (tag0)
//			{
//			case 0:
//				if (x1[1] > x0[1])
//				{
//					xp[0] = std::floor(xp[0]);
//					xp[1] += 1;
//					res->push_back(xp);
//				}
//				else if (x1[1] < x0[1])
//				{
//					xp[0] = std::floor(xp[0]) + 1;
//					xp[1] -= 1;
//					res->push_back(xp);
//				}
//				else if (x1[0] > x0[0])
//				{
//					xp[0] = std::floor(xp[0]);
//					++xp[1];
//					res->push_back(xp);
//					++xp[0];
//					res->push_back(xp);
//					--xp[1];
//					res->push_back(xp);
//				}
//				else
//				{
//					xp[0] = std::floor(xp[0]) + 1;
//					--xp[1];
//					res->push_back(xp);
//					--xp[0];
//					res->push_back(xp);
//					++xp[1];
//					res->push_back(xp);
//				}
//				break;
//			case 1:
//
//				if (x1[0] > x0[0])
//				{
//					xp[1] = std::floor(xp[1]) + 1;
//					++xp[0];
//					res->push_back(xp);
//				}
//				else if (x1[0] < x0[0])
//				{
//					xp[1] = std::floor(xp[0]);
//					--xp[1];
//					res->push_back(xp);
//				}
//
//				break;
//			case 2:
//				break;
//
//			}
//		}
//		else if (tag0 == 0 && tag1 == 1)
//		{
//			if (x1[1] > x0[1] && x1[0] > x0[0])
//			{
//				xp[0] = std::floor(xp[0]);
//				xp[1] += 1;
//				res->push_back(xp);
//				++xp[0];
//				res->push_back(xp);
//			}
//			else if (x1[1] < x0[1] && x1[0] > x0[0])
//			{
//				xp[0] = std::floor(xp[0]);
//				xp[1] -= 1;
//				res->push_back(xp);
//				++xp[0];
//				res->push_back(xp);
//			}
//
//			else if (x1[1] > x0[1] && x1[0] < x0[0])
//			{
//				xp[0] = std::floor(xp[0]) + 1;
//				xp[1] += 1;
//				res->push_back(xp);
//				--xp[0];
//				res->push_back(xp);
//			}
//			else if (x1[1] < x0[1] && x1[0] < x0[0])
//			{
//				xp[0] = std::floor(xp[0]) + 1;
//				xp[1] -= 1;
//				res->push_back(xp);
//				--xp[0];
//				res->push_back(xp);
//			}
//
//		}
//		else if (tag0 == 1 && tag1 == 0)
//		{
////			if (x1[0] > x0[0] && x1[1] > x0[1])
////			{
////				xp[0] = std::floor(xp[0]) + 1;
////				res->push_back(xp);
////				--xp[1];
////				res->push_back(xp);
////			}
////			else
////
//			if (x1[0] < x0[0] && x1[1] > x0[1])
//			{
//				xp[1] = std::floor(xp[1]);
//				--xp[0];
//				res->push_back(xp);
//				++xp[1];
//				res->push_back(xp);
//			}
//			else if (x1[0] < x0[0] && x1[1] < x0[1])
//			{
//				xp[1] = std::floor(xp[1]) - 1;
//				res->push_back(xp);
//				--xp[0];
//				res->push_back(xp);
//			}
////			else if (x1[1] > x0[1] && x1[0] < x0[0])
////			{
////				xp[0] = std::floor(xp[0]);
////				xp[1] += 1;
////				res->push_back(xp);
////				--xp[0];
////				res->push_back(xp);
////			}
//
//		}
////		if (tag0 == tag1)
////		{
////			if (tag0 == 0)
////			{
////
////				if (x1[1] > x0[1])
////				{
////					xp[0] = std::floor(xp[0]);
////					xp[1] += 1;
////				}
////				else
////				{
////					xp[0] = std::floor(xp[0]) + 1;
////					xp[1] -= 1;
////				}
////
////			}
////			else if (tag0 == 1)
////			{
////
////				xp[0] += (x1[0] > x0[0]) ? 1 : -1;
////				xp[1] = std::floor(xp[0]) + ((x1[0] > x0[0]) ? 1 : 0);
////
////			}
////
////			res->push_back(xp);
		i0 = i1;
		i1 = i2;
		++i2;
		if (i2 == points.end())
		{
			i2 = points.begin();
		}

		q1 = q2;
		q2 = std::atan2(x2[1] - x1[1], x2[0] - x1[0]);

	}
}

USE_CASE(model)
{

	std::vector<coordinates_type> p0, p1, p2;

	options["Points"].as(&p0);

	LOGGER << p0 << std::endl;

	divid_points(p0, &p1);

	divid_points2(p0, &p2);

	LOGGER << p1 << std::endl;

	LOGGER << p2 << std::endl;

	p0.push_back(p0.front());

	p1.push_back(p1.front());

	p2.push_back(p2.front());

	LOGGER << SAVE(p0) << std::endl;
	LOGGER << SAVE(p1) << std::endl;
	LOGGER << SAVE(p2) << std::endl;
}
