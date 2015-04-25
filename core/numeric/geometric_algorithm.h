/*
 * geometry_algorithm.h
 *
 *  created on: 2014-4-21
 *      Author: salmon
 */

#ifndef GEOMETRY_ALGORITHM_H_
#define GEOMETRY_ALGORITHM_H_

#include <stddef.h>
#include <tuple>

#include "../gtl/ntuple.h"
#include "../gtl/primitives.h"

namespace simpla
{

template<typename T0, typename T1, typename T2, typename T3>
std::tuple<Real, Vec3> distance_point_to_plane(T0 const & x0, T1 const & p0,
		T2 const & p1, T3 const & p2)
{
	Vec3 n;

	n = cross(p1 - p0, p2 - p1);

	n /= inner_product(n, n);

	return std::forward_as_tuple(inner_product(p0 - x0, n), std::move(n));

}
template<typename T0, typename T1, typename T2>
Real distance_point_to_line_segment(T0 const & p0, T1 const & p1, T2 const & x)
{
	return inner_product(x - p0, p1 - p0) / inner_product(p1 - p0, p1 - p0);
}
/**
 *
 *
 *     x  o
 *       /|
 *      / | d
 *     o--o---------------o
 *  p0   s                      p1
 *
 *
 * @return <x,p0,p1>
 *
 *
 */
template<typename TI, typename TX>
std::tuple<Real, TI, TI> distance_point_to_polylines(TI const & ib,
		TI const & ie, TX const & x)
{
	typedef TX Vec3;

	Real min_dist2 = std::numeric_limits<Real>::max();

	Real res_s = 0;

	TI res_p0, res_p1;

	TI it = ib;

	TI p0, p1;

	Real dist;

	p1 = it;

	while (it != ie)
	{
		p0 = p1;

		++it;

		if (it == ie)
		{
			p1 = ib;
		}
		else
		{
			p1 = it;
		};

		auto s = distance_point_to_line_segment(*p0, *p1, x);

		if (s < 0)
		{
			s = 0;
		}
		else if (s > 1)
		{
			s = 1;

		}

		Vec3 d = x - ((1 - s) * (*p0) + s * (*p1));

		Real dist2 = inner_product(d, d);

		if (min_dist2 > dist2 || (min_dist2 == dist2 && s == 0))
		{
			res_p0 = p0;
			res_p1 = p1;
			res_s = s;
			min_dist2 = dist2;
		}
	}

	return std::forward_as_tuple(res_s, res_p0, res_p1);

}

template<typename T0, typename T1, typename T2, typename T3>
Vec3 distance_line_to_line(T0 const& P0, T1 const & P1, T2 const & Q0,
		T3 const & Q1)
{
	Real s = 0.0;
	Real t = 0.0;
	Real dist = 0.0;

	auto u = P1 - P0;
	auto v = Q1 - Q0;
	auto w0 = P0 - Q0;

	// @ref http://geomalgorithms.com/a07-_distance.html
	Real a = inner_product(u, u);
	Real b = inner_product(u, v);
	Real c = inner_product(v, v);
	Real d = inner_product(u, w0);
	Real e = inner_product(v, w0);

	if (std::abs(a * c - b * b) < EPSILON)
	{
		//two lines are parallel
		s = 0;
		t = 0;
		dist = d / b;
	}
	else
	{
		s = (b * e - c * d) / (a * c - b * b);

		t = (a * e - b * d) / (a * c - b * b);

		auto w = w0
				+ ((b * e - c * d) * u - (a * e - b * d) * v) / (a * c - b * b);

		dist = inner_product(w, w);

	}

	Vec3 res =
	{ s, t, dist };
	return std::move(res);
}

template<typename T0, typename T1, typename T2>
Real intersection_line_to_polygons(T0 const & p0, T1 const & p1,
		T2 const & polygen)
{

	auto it = polygen.begin();

	auto q0 = *it;
	auto q1 = *(++it);
	auto q2 = *(++it);

	Vec3 n;
	n = cross(q2 - q1, q1 - q0);
	n /= std::sqrt(inner_product(n, n));

	it = polygen.begin();

	while (it != polygen.end())
	{
		auto q0 = *it;
		auto q1 = *(++it);
		q0 -= inner_product(q0, n) * n;
		q1 -= inner_product(q1, n) * n;
	}
}

/**
 *
 *
 *     x' o
 *       /
 *      /
 *     o------------------o
 *  p0  \                      p1
 *       \
 *        o
 *        x
 *
 *
 */
template<typename TPlane>
inline void reflect(TPlane const & p, Vec3 *x, Vec3 * v)
{
	Vec3 u;

	u = cross(p[1] - p[0], p[2] - p[0]);

	Real a = dot(u, *x - p[0]);

	if (a < 0)
	{
		Real b = 1.0 / inner_product(u, u);
		*x -= 2 * a * u * b;
		*v -= 2 * inner_product(u, *v) * u * b;
	}

}
//template<typename TC>
//std::tuple<Real, Real> intersection_line_and_triangle(TC const& l0,
//		TC const & l1, TC const & p0, TC const & p1, TC const & p2)
//{
//	Real s, t;
//
//	auto u = p1 - p0;
//	auto v = p2 - p0;
//
//	// @ref http://geomalgorithms.com/a07-_distance.html
//	Real a = inner_product(x1 - x0, x1 - x0);
//	Real b = inner_product(x1 - x0, y1 - y0);
//	Real c = inner_product(y1 - y0, y1 - y0);
//	Real d = inner_product(y0 - x0, x1 - x0);
//	Real e = inner_product(y0 - x0, y1 - y0);
//
//	if (std::abs(a * c - b * b) < EPSILON)
//	{
//		//two lines are parallel
//		s = 0;
//		t = d / b;
//	}
//	else
//	{
//		s = (b * e - c * d) / (a * c - b * b);
//		t = (a * e - b * d) / (a * c - b * b);
//	}
//	return std::make_tuple(s, t);
//}
template<typename TS, size_t NDIMS>
bool intersection(nTuple<TS, NDIMS> const & l_b, nTuple<TS, NDIMS> const &l_e,
		nTuple<TS, NDIMS> *r_b, nTuple<TS, NDIMS> *r_e)
{
	bool has_overlap = false;

	nTuple<TS, NDIMS> & r_start = *r_b;
	nTuple<TS, NDIMS> r_count = *r_e - *r_b;

	nTuple<TS, NDIMS> l_count = l_e - l_b;

	for (int i = 0; i < NDIMS; ++i)
	{
		if (r_start[i] + r_count[i] <= l_b[i]
				|| r_start[i] >= l_b[i] + l_count[i])
			return false;

		TS start = std::max(l_b[i], r_start[i]);
		TS end = std::min(l_b[i] + l_count[i], r_start[i] + r_count[i]);

		if (end > start)
		{
			r_start[i] = start;
			r_count[i] = end - start;

			has_overlap = true;
		}
	}

	if (!true)
	{
		*r_b = r_start;
		*r_e = r_start;
	}
	else
	{
		*r_b = r_start;
		*r_e = r_start + r_count;
	}
	return has_overlap;
}
//
///**
// *  @ingroup numeric
// *  @addtogroup geometry_algorithm
// *  @{
// */
//template<size_t DIM, typename TR, typename TRange>
//bool PointInRectangle(nTuple<TR, DIM> const &x, TRange const & range)
//{
//	bool res = true;
//
//	auto min = std::get<0>(range);
//
//	auto max = std::get<1>(range);
//
//	for (size_t i = 0; i < DIM; ++i)
//	{
//		res = res && (x[i] >= min[i] && x[i] <= max[i]);
//	}
//	return res;
//}
//
//template<typename TI>
//bool clipping(int ndims, TI const * l_begin, TI const * l_end, TI * r_begin,
//		TI * r_end)
//{
//	bool has_overlap = false;
//
//	for (int i = 0; i < ndims; ++i)
//	{
//		if (r_end[i] <= l_begin[i] || r_begin[i] >= l_end[i])
//			return false;
//
//		auto begin = std::max(l_begin[i], r_begin[i]);
//		auto end = std::min(l_end[i], r_end[i]);
//
//		if (end > begin)
//		{
//			r_begin[i] = begin;
//			r_end[i] = end;
//
//			has_overlap = true;
//		}
//	}
//
//	return has_overlap;
//}
//template<typename TS, size_t NDIMS>
//bool clipping(nTuple<TS, NDIMS> l_begin, nTuple<TS, NDIMS> l_end,
//		nTuple<TS, NDIMS> *pr_begin, nTuple<TS, NDIMS> *pr_end)
//{
//	return clipping(NDIMS, &l_begin[0], &l_end[0], &(*pr_begin)[0],
//			&(*pr_end)[0]);
//}

//inline nTuple<Real,3> Distance(nTuple<2, nTuple<Real,3>> p, nTuple<Real,3> const &x)
//{
//	nTuple<Real,3> u, v;
//	v = p[1] - p[0];
//	u = Cross(Cross(x - p[0], v), v) / _DOT3(v, v);
//	return std::move(u);
//}
//template<typename TDict, typename TModel, typename TSurface>
//void createSurface(TDict const & dict, TModel const & model, TSurface * surf)
//{
//	if (dict["Width"].is_number())
//	{
//		createSurface(model, dict["Width"].template as<Real>(), surf);
//	}
//	else
//	{
//		WARNING << "illegal configuation!";
//	}
//}
//template<typename TModel, typename TSurface>
//void createSurface(TModel const & model, Real width, TSurface * surf)
//{
////
//////	typedef typename TSurface::plane_type plane_type;
////
//////	auto extent = mesh.get_extents();
//////	auto dims = mesh.get_dimensions();
//////	auto xmin = extent.first;
//////	auto xmax = extent.second;
//////	auto d = mesh.get_dx();
//////	nTuple<Real,3> x0 = { 0, 0, 0 };
//////	nTuple<Real,3> x1 = { d[0], 0, 0 };
//////	nTuple<Real,3> x2 = { 0, d[1], 0 };
//////	nTuple<Real,3> x3 = { 0, 0, d[2] };
//////
//////	for (auto s : mesh.select(VERTEX))
//////	{
//////		auto x = mesh.get_coordinates(s);
//////
//////		if (x[0] < xmin[0] + width)
//////		{
//////			surf->insert(s, plane_type( { x0, x1, x2 }));
//////			continue;
//////		}
//////		else if (x[0] > xmax[0] - width)
//////		{
//////			surf->insert(s, plane_type( { x0, x2, x1 }));
//////			continue;
//////		}
//////
//////		if (x[1] < xmin[1] + width)
//////		{
//////			surf->insert(s, plane_type( { x0, x1, x2 }));
//////			continue;
//////		}
//////		else if (x[1] > xmax[1] + width)
//////		{
//////			surf->insert(s, plane_type( { x0, x1, x2 }));
//////			continue;
//////		}
//////
//////		if (x[2] < xmin[2] + width)
//////		{
//////			surf->insert(s, plane_type( { x0, x1, x2 }));
//////			continue;
//////		}
//////		else if (x[2] > xmax[2] - width)
//////		{
//////			surf->insert(s, plane_type( { x0, x1, x2 }));
//////			continue;
//////		}
//////
//////	}
//}
//
///**
// * decompose a N-dimensional block range [b,e) into 'num_part' parts,
// * and return the 'proc_num'th part [ob,oe)
// * @param b minus index of block range
// * @param e maxim index of block range
// * @param num_part
// * @param part_num
// * @param decompose_method select decompose algorithm (UNIMPLEMENTED)
// * @return  the 'proc_num'th part [ob,oe)
// * if 'num_part==0' return [b,e)
// */
//template<typename TI, size_t N>
//std::tuple<nTuple<TI, N>, nTuple<TI, N>> block_decompose(
//		nTuple<TI, N> const & b, nTuple<TI, N> const & e, int num_part = 0,
//		int part_num = 0, size_t decompose_method = 0UL)
//{
//	if (num_part == 0)
//	{
//		return std::forward_as_tuple(b, e);
//	}
//
//	nTuple<TI, N> ob, oe;
//
//	TI length = 0;
//	int dim_num = 0;
//	for (int i = 0; i < N; ++i)
//	{
//		if (e[i] - b[i] > length)
//		{
//			length = e[i] - b[i];
//			dim_num = i;
//		}
//	}
//
//	ob = b;
//	oe = e;
//	ob[dim_num] = b[dim_num] + (length * part_num) / num_part;
//	oe[dim_num] = b[dim_num] + (length * (part_num + 1)) / num_part;
//
//	return std::forward_as_tuple(ob, oe);
//}

//! @}
}// namespace simpla

#endif /* GEOMETRY_ALGORITHM_H_ */
