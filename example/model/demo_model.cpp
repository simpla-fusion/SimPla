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
#include <set>
#include "../../core/gtl/ntuple.h"
#include "../../core/gtl/primitives.h"
#include "../../core/numeric/geometric_algorithm.h"
#include "../../core/utilities/utilities.h"
#include "../../core/io/io.h"
#include "../../core/application/application.h"
#include "../../core/dataset/datatype.h"
#include "../../core/model/model.h"
#include "../../core/numeric/pointinpolygon.h"

namespace simpla
{

template<size_t NDIMS = 3>
struct MeshDummy_
{
	static constexpr size_t FULL_DIGITS = std::numeric_limits<size_t>::digits;

	static constexpr size_t INDEX_DIGITS = (FULL_DIGITS
			- CountBits<FULL_DIGITS>::n) / 3;

	static constexpr size_t FLOATING_POINT_POS = 4;

	static constexpr size_t FLOATING_POINT_FACTOR = 1 << FLOATING_POINT_POS;

	static constexpr Real COORDINATES_TO_INDEX_FACTOR = static_cast<Real>(1
			<< FLOATING_POINT_POS);
	static constexpr Real INDEX_TO_COORDINATES_FACTOR = 1.0
			/ COORDINATES_TO_INDEX_FACTOR;

	static constexpr size_t INDEX_MASK = (1UL << (INDEX_DIGITS)) - 1;

	static constexpr size_t D_INDEX = (1UL << (FLOATING_POINT_POS));

	static constexpr size_t _DZ = D_INDEX << (INDEX_DIGITS * 2 - 1);

	static constexpr size_t _DY = D_INDEX << (INDEX_DIGITS - 1);

	static constexpr size_t _DX = D_INDEX >> 1;

	static constexpr size_t CELL_ID_MASK_ = //
			(((1UL << (INDEX_DIGITS - FLOATING_POINT_POS - 1)) - 1)
					<< (FLOATING_POINT_POS - 1)) & INDEX_MASK;

	static constexpr size_t CELL_ID_MASK =

	(CELL_ID_MASK_ << (INDEX_DIGITS * 2))

	| (CELL_ID_MASK_ << (INDEX_DIGITS))

	| (CELL_ID_MASK_);

	static constexpr size_t SUB_CELL_ID_MASK_ = 1 << (FLOATING_POINT_POS - 1);

	static constexpr size_t SUB_CELL_ID_MASK =

	(SUB_CELL_ID_MASK_ << (INDEX_DIGITS * 2))

	| (SUB_CELL_ID_MASK_ << (INDEX_DIGITS))

	| (SUB_CELL_ID_MASK_);

	static constexpr size_t ZERO_INDEX = 1UL
			<< (INDEX_DIGITS - FLOATING_POINT_POS - 1);

	static constexpr Real ZERO_COORDINATE = static_cast<Real>(ZERO_INDEX);

	typedef size_t id_type;
	typedef nTuple<Real, 3> coordinates_type;

	static coordinates_type id_to_coordinates(id_type s)
	{
		coordinates_type x;
		x[0] = static_cast<Real>(s & INDEX_MASK) * INDEX_TO_COORDINATES_FACTOR
				- ZERO_COORDINATE;
		x[1] = static_cast<Real>((s >> INDEX_DIGITS) & INDEX_MASK)
				* INDEX_TO_COORDINATES_FACTOR - ZERO_COORDINATE;
		x[2] = static_cast<Real>((s >> (INDEX_DIGITS * 2)) & INDEX_MASK)
				* INDEX_TO_COORDINATES_FACTOR - ZERO_COORDINATE;

		return std::move(x);
	}

	static constexpr size_t m_d_[4][3] = { 0, 0, 0, _DX, _DY, _DZ, _DY | _DZ,
			_DZ | _DX, _DX | _DY, _DX | _DY | _DZ, _DX | _DY | _DZ, _DX | _DY
					| _DZ };

	template<size_t IFORM = 0>
	static constexpr id_type id(size_t i, size_t j, size_t k, size_t n = 0)
	{
		return ((i + ZERO_INDEX) << FLOATING_POINT_POS)
				| ((j + ZERO_INDEX) << (FLOATING_POINT_POS + INDEX_DIGITS))
				| ((k + ZERO_INDEX) << (FLOATING_POINT_POS + INDEX_DIGITS * 2))
				| m_d_[IFORM][n];
	}

	static constexpr id_type coordinates_to_id(coordinates_type const &x)
	{
		return (static_cast<size_t>((x[0] + ZERO_COORDINATE)
				* COORDINATES_TO_INDEX_FACTOR))

				| ((static_cast<size_t>((x[1] + ZERO_COORDINATE)
						* COORDINATES_TO_INDEX_FACTOR)) << (INDEX_DIGITS))

				| ((static_cast<size_t>((x[2] + ZERO_COORDINATE)
						* COORDINATES_TO_INDEX_FACTOR)) << (INDEX_DIGITS * 2));
	}

};
template<size_t NDIMS>
constexpr size_t MeshDummy_<NDIMS>::m_d_[4][3];

typedef MeshDummy_<3> MeshDummy;
static constexpr Real PI = 3.1415926535;
typedef typename MeshDummy::id_type id_type;
typedef typename MeshDummy::coordinates_type coordinates_type;

template<typename TPoints>
void get_intersctions(TPoints const & points, coordinates_type const & shift,
		std::vector<coordinates_type> *res)
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
				xp[n] = std::floor(x0[n] - shift[n]) + shift[n] + 1;
			}
			else
			{
				dx = -1;
				xp[n] = std::floor(x0[n] - shift[n]) + shift[n];
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

typedef coordinates_type Vec3;

template<typename TPoints>
void find_boundary2D(TPoints const & points, coordinates_type const & shift,
		std::vector<coordinates_type> *res, int ZAXIS = 2)
{

	auto i0 = points.begin();

	while (i0 != points.end())
	{
		coordinates_type p0 = *i0;
		auto i1 = ++i0;

		if (i1 == points.end())
		{
			i1 = points.begin();
		}
		coordinates_type p1 = *i1;

		coordinates_type p2;
		p2 = p1;

		p2[ZAXIS] += 1.0;

		coordinates_type x0 = //
				{ //
				std::floor(std::min(p0[0], p1[0]) - shift[0]) + shift[0], //
				std::floor(std::min(p0[1], p1[1]) - shift[1]) + shift[1], //
				std::floor(std::min(p0[2], p1[2]) - shift[2]) + shift[2]     //
				};
		x0 += 0.5;
//		id_type s = MeshDummy::coordinates_to_id(x0);

		res->push_back(x0);
		res->push_back(distance_point_to_face(x0, p0, p1, p2));

	}

}

SP_APP(model)
{

	typedef typename MeshDummy::coordinates_type coordinates_type;

	typedef typename MeshDummy::id_type id_type;

	std::vector<coordinates_type> p0, p1, p2, p3, p4;

	options["Points"].as(&p0);

	PointInPolygon p_in_p(p0);

	std::set<id_type> IDs;

	for (size_t i = 0; i < 10; ++i)
	{
		for (size_t j = 0; j < 10; ++j)
		{
			for (size_t k = 0; k < 10; ++k)
			{
				coordinates_type x;
				x[0] = static_cast<Real>(i) + 0.5;
				x[1] = static_cast<Real>(j) + 0.5;
				x[2] = static_cast<Real>(k) + 0.5;

				if (p_in_p(x))
					IDs.insert(MeshDummy::coordinates_to_id(x));
			}
		}
	}

	std::set<id_type> in_side;

	std::copy_if(IDs.begin(), IDs.end(),
			std::inserter(in_side, in_side.begin()), [&](id_type const &s )
			{
				return p_in_p(MeshDummy::id_to_coordinates(s));
			});

	std::transform(in_side.begin(), in_side.end(), std::back_inserter(p1),
			[&](id_type const &s )
			{
				return std::move(MeshDummy::id_to_coordinates(s));
			});

	Model model;

	model.set(in_side, Model::VACUUM);

	std::set<id_type> boundary;

	for (size_t i = 0; i < 10; ++i)
	{
		for (size_t j = 0; j < 10; ++j)
		{
			for (size_t k = 0; k < 10; ++k)
			{
				for (size_t n = 0; n < 3; ++n)
				{
					size_t s = MeshDummy::id<2>(i, j, k, n);

					if (model.check_boundary_face(s, Model::VACUUM))
					{
						boundary.insert(s);
					}

				}
			}
		}
	}
	std::transform(boundary.begin(), boundary.end(), std::back_inserter(p2),
			[&](id_type const &s )
			{
				return std::move(MeshDummy::id_to_coordinates(s));
			});

//	LOGGER << p0 << std::endl;
//
//	LOGGER << p1 << std::endl;
//
//	LOGGER << p2 << std::endl;
//
	p0.push_back(p0.front());
//
//	p1.push_back(p1.front());
//
//	p2.push_back(p2.front());
//
	LOGGER << SAVE(p0) << std::endl;
	LOGGER << SAVE(p1) << std::endl;
	LOGGER << SAVE(p2) << std::endl;

	coordinates_type shift = { 0, 0, 0 };

	get_intersctions(p0, shift, &p3);

	find_boundary2D(p3, shift, &p4);

	p3.push_back(p3.front());
	LOGGER << SAVE(p3) << std::endl;

	size_t dims[2] = { p4.size() / 2, 2 };
	LOGGER << save("p4", &p4[0], 2, dims) << std::endl;
}

} //namespace simpla

//
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
//int boundary_tag(coordinates_type const & x)
//{
//	int res = 0;
//	if (x[0] != std::floor(x[0]))
//	{
//		res = 0;
//	}
//	else if (x[1] != std::floor(x[1]))
//	{
//		res = 1;
//	}
//	else if (x[2] != std::floor(x[2]))
//	{
//		res = 2;
//	}
//	return res;
//}

//void divid_to_boundary(std::vector<coordinates_type> const& points,
//		std::vector<boundary_s> * res, int ZAXIS = 2)
//{
//
//	auto i0 = points.rbegin();
//	auto i1 = points.begin();
//	auto i2 = ++points.begin();
//
//	coordinates_type x0, x1, x2;
//	x0 = *i0;
//	x1 = *i1;
//	x2 = *i2;
//
//	Real q1 = std::atan2(x1[1] - x0[1], x1[0] - x0[0]);
//
//	Real q2 = std::atan2(x2[1] - x1[1], x2[0] - x1[0]);
//
//	while (i1 != points.end())
//	{
//		coordinates_type xp, xm;
//
//		xm[0] = std::floor(xp[0]);
//		xm[1] = std::floor(xp[1]);
//		xm[2] = std::floor(xp[2]);
//
//		if (q1 == q2)
//		{
//			if (-PI / 4 < angle && angle < PI / 4)
//			{
//
//			}
//		}
////		auto tag0 = boundary_tag(x0);
////		auto tag1 = boundary_tag(x1);
////
////		coordinates_type xp = x0;
////
////		if (tag0 == tag1)
////		{
////			switch (tag0)
////			{
////			case 0:
////				if (x1[1] > x0[1])
////				{
////					xp[0] = std::floor(xp[0]);
////					xp[1] += 1;
////					res->push_back(xp);
////				}
////				else if (x1[1] < x0[1])
////				{
////					xp[0] = std::floor(xp[0]) + 1;
////					xp[1] -= 1;
////					res->push_back(xp);
////				}
////				else if (x1[0] > x0[0])
////				{
////					xp[0] = std::floor(xp[0]);
////					++xp[1];
////					res->push_back(xp);
////					++xp[0];
////					res->push_back(xp);
////					--xp[1];
////					res->push_back(xp);
////				}
////				else
////				{
////					xp[0] = std::floor(xp[0]) + 1;
////					--xp[1];
////					res->push_back(xp);
////					--xp[0];
////					res->push_back(xp);
////					++xp[1];
////					res->push_back(xp);
////				}
////				break;
////			case 1:
////
////				if (x1[0] > x0[0])
////				{
////					xp[1] = std::floor(xp[1]) + 1;
////					++xp[0];
////					res->push_back(xp);
////				}
////				else if (x1[0] < x0[0])
////				{
////					xp[1] = std::floor(xp[0]);
////					--xp[1];
////					res->push_back(xp);
////				}
////
////				break;
////			case 2:
////				break;
////
////			}
////		}
////		else if (tag0 == 0 && tag1 == 1)
////		{
////			if (x1[1] > x0[1] && x1[0] > x0[0])
////			{
////				xp[0] = std::floor(xp[0]);
////				xp[1] += 1;
////				res->push_back(xp);
////				++xp[0];
////				res->push_back(xp);
////			}
////			else if (x1[1] < x0[1] && x1[0] > x0[0])
////			{
////				xp[0] = std::floor(xp[0]);
////				xp[1] -= 1;
////				res->push_back(xp);
////				++xp[0];
////				res->push_back(xp);
////			}
////
////			else if (x1[1] > x0[1] && x1[0] < x0[0])
////			{
////				xp[0] = std::floor(xp[0]) + 1;
////				xp[1] += 1;
////				res->push_back(xp);
////				--xp[0];
////				res->push_back(xp);
////			}
////			else if (x1[1] < x0[1] && x1[0] < x0[0])
////			{
////				xp[0] = std::floor(xp[0]) + 1;
////				xp[1] -= 1;
////				res->push_back(xp);
////				--xp[0];
////				res->push_back(xp);
////			}
////
////		}
////		else if (tag0 == 1 && tag1 == 0)
////		{
//////			if (x1[0] > x0[0] && x1[1] > x0[1])
//////			{
//////				xp[0] = std::floor(xp[0]) + 1;
//////				res->push_back(xp);
//////				--xp[1];
//////				res->push_back(xp);
//////			}
//////			else
//////
////			if (x1[0] < x0[0] && x1[1] > x0[1])
////			{
////				xp[1] = std::floor(xp[1]);
////				--xp[0];
////				res->push_back(xp);
////				++xp[1];
////				res->push_back(xp);
////			}
////			else if (x1[0] < x0[0] && x1[1] < x0[1])
////			{
////				xp[1] = std::floor(xp[1]) - 1;
////				res->push_back(xp);
////				--xp[0];
////				res->push_back(xp);
////			}
//////			else if (x1[1] > x0[1] && x1[0] < x0[0])
//////			{
//////				xp[0] = std::floor(xp[0]);
//////				xp[1] += 1;
//////				res->push_back(xp);
//////				--xp[0];
//////				res->push_back(xp);
//////			}
////
////		}
//////		if (tag0 == tag1)
//////		{
//////			if (tag0 == 0)
//////			{
//////
//////				if (x1[1] > x0[1])
//////				{
//////					xp[0] = std::floor(xp[0]);
//////					xp[1] += 1;
//////				}
//////				else
//////				{
//////					xp[0] = std::floor(xp[0]) + 1;
//////					xp[1] -= 1;
//////				}
//////
//////			}
//////			else if (tag0 == 1)
//////			{
//////
//////				xp[0] += (x1[0] > x0[0]) ? 1 : -1;
//////				xp[1] = std::floor(xp[0]) + ((x1[0] > x0[0]) ? 1 : 0);
//////
//////			}
//////
//////			res->push_back(xp);
//		i0 = i1;
//		i1 = i2;
//		++i2;
//		if (i2 == points.end())
//		{
//			i2 = points.begin();
//		}
//
//		q1 = q2;
//		q2 = std::atan2(x2[1] - x1[1], x2[0] - x1[0]);
//
//	}
//}

