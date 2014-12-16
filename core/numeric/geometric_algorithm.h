/*
 * geometry_algorithm.h
 *
 *  created on: 2014-4-21
 *      Author: salmon
 */

#ifndef GEOMETRY_ALGORITHM_H_
#define GEOMETRY_ALGORITHM_H_

#include <algorithm>
#include <cmath>
#include <complex>
#include <valarray>

#include "../utilities/ntuple.h"
#include "../utilities/primitives.h"
#include "../utilities/log.h"

namespace simpla
{
//template<typename TS,  size_t  NDIMS>
//bool Clipping(nTuple<TS,NDIMS> const & l_start, nTuple<TS,NDIMS> const &l_count, nTuple<TS,NDIMS> *pr_start,
//        nTuple<TS,NDIMS> *pr_count)
//{
//	bool has_overlap = false;
//
//	nTuple<TS,NDIMS> & r_start = *pr_start;
//	nTuple<TS,NDIMS> & r_count = *pr_count;
//
//	for (int i = 0; i < NDIMS; ++i)
//	{
//		if (r_start[i] + r_count[i] <= l_start[i] || r_start[i] >= l_start[i] + l_count[i])
//			return false;
//
//		TS start = std::max(l_start[i], r_start[i]);
//		TS end = std::min(l_start[i] + l_count[i], r_start[i] + r_count[i]);
//
//		if (end > start)
//		{
//			r_start[i] = start;
//			r_count[i] = end - start;
//
//			has_overlap = true;
//		}
//	}
//
//	return has_overlap;
//}

/**
 *  \addtogroup GeometryAlgorithm
 *  @{
 */
template<size_t DIM, typename TR, typename TRange>
bool PointInRectangle(nTuple<TR, DIM> const &x, TRange const & range)
{
	bool res = true;

	auto min = std::get<0>(range);

	auto max = std::get<1>(range);

	for (size_t i = 0; i < DIM; ++i)
	{
		res = res && (x[i] >= min[i] && x[i] <= max[i]);
	}
	return res;
}

template<typename TI>
bool clipping(int ndims, TI const * l_begin, TI const * l_end, TI * r_begin,
		TI * r_end)
{
	bool has_overlap = false;

	for (int i = 0; i < ndims; ++i)
	{
		if (r_end[i] <= l_begin[i] || r_begin[i] >= l_end[i])
			return false;

		auto begin = std::max(l_begin[i], r_begin[i]);
		auto end = std::min(l_end[i], r_end[i]);

		if (end > begin)
		{
			r_begin[i] = begin;
			r_end[i] = end;

			has_overlap = true;
		}
	}

	return has_overlap;
}
template<typename TS, size_t NDIMS>
bool clipping(nTuple<TS, NDIMS> l_begin, nTuple<TS, NDIMS> l_end,
		nTuple<TS, NDIMS> *pr_begin, nTuple<TS, NDIMS> *pr_end)
{
	return clipping(NDIMS, &l_begin[0], &l_end[0], &(*pr_begin)[0],
			&(*pr_end)[0]);
}

//inline nTuple<Real,3> Distance(nTuple<2, nTuple<Real,3>> p, nTuple<Real,3> const &x)
//{
//	nTuple<Real,3> u, v;
//	v = p[1] - p[0];
//	u = Cross(Cross(x - p[0], v), v) / _DOT3(v, v);
//	return std::move(u);
//}
inline Real Distance(nTuple<nTuple<Real, 3>, 3> const & p,
		nTuple<Real, 3u> const &x)
{
	nTuple<Real, 3u> v;
	v = cross(p[1] - p[0], p[2] - p[0]);
	return dot(v, x - p[0]) / std::sqrt(dot(v, v));
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
inline void Reflect(TPlane const & p, nTuple<Real, 3>*x, nTuple<Real, 3> * v)
{
	nTuple<Real, 3> u;

	u = cross(p[1] - p[0], p[2] - p[0]);

	Real a = dot(u, *x - p[0]);

	if (a < 0)
	{
		Real b = 1.0 / dot(u, u);
		*x -= 2 * a * u * b;
		*v -= 2 * dot(u, *v) * u * b;
	}

}
template<typename TDict, typename TModel, typename TSurface>
void createSurface(TDict const & dict, TModel const & model, TSurface * surf)
{
	if (dict["Width"].is_number())
	{
		createSurface(model, dict["Width"].template as<Real>(), surf);
	}
	else
	{
		WARNING << "illegal configuation!";
	}
}
template<typename TModel, typename TSurface>
void createSurface(TModel const & model, Real width, TSurface * surf)
{
//
////	typedef typename TSurface::plane_type plane_type;
//
////	auto extent = mesh.get_extents();
////	auto dims = mesh.get_dimensions();
////	auto xmin = extent.first;
////	auto xmax = extent.second;
////	auto d = mesh.get_dx();
////	nTuple<Real,3> x0 = { 0, 0, 0 };
////	nTuple<Real,3> x1 = { d[0], 0, 0 };
////	nTuple<Real,3> x2 = { 0, d[1], 0 };
////	nTuple<Real,3> x3 = { 0, 0, d[2] };
////
////	for (auto s : mesh.select(VERTEX))
////	{
////		auto x = mesh.get_coordinates(s);
////
////		if (x[0] < xmin[0] + width)
////		{
////			surf->insert(s, plane_type( { x0, x1, x2 }));
////			continue;
////		}
////		else if (x[0] > xmax[0] - width)
////		{
////			surf->insert(s, plane_type( { x0, x2, x1 }));
////			continue;
////		}
////
////		if (x[1] < xmin[1] + width)
////		{
////			surf->insert(s, plane_type( { x0, x1, x2 }));
////			continue;
////		}
////		else if (x[1] > xmax[1] + width)
////		{
////			surf->insert(s, plane_type( { x0, x1, x2 }));
////			continue;
////		}
////
////		if (x[2] < xmin[2] + width)
////		{
////			surf->insert(s, plane_type( { x0, x1, x2 }));
////			continue;
////		}
////		else if (x[2] > xmax[2] - width)
////		{
////			surf->insert(s, plane_type( { x0, x1, x2 }));
////			continue;
////		}
////
////	}
}

/**
 * decompose a N-dimensional block range [b,e) into 'num_part' parts,
 * and return the 'proc_num'th part [ob,oe)
 * @param b minus index of block range
 * @param e maxim index of block range
 * @param num_part
 * @param part_num
 * @param decompose_method select decompose algorithm (UNIMPLEMENTED)
 * @return  the 'proc_num'th part [ob,oe)
 * if 'num_part==0' return [b,e)
 */
template<typename TI, size_t N>
std::tuple<nTuple<TI, N>, nTuple<TI, N>> block_decompose(
		nTuple<TI, N> const & b, nTuple<TI, N> const & e, int num_part = 0,
		int part_num = 0, size_t decompose_method = 0UL)
{
	if (num_part == 0)
	{
		return std::forward_as_tuple(b, e);
	}

	nTuple<TI, N> ob, oe;

	TI length = 0;
	int dim_num = 0;
	for (int i = 0; i < N; ++i)
	{
		if (e[i] - b[i] > length)
		{
			length = e[i] - b[i];
			dim_num = i;
		}
	}

	ob = b;
	oe = e;
	ob[dim_num] = b[dim_num] + (length * part_num) / num_part;
	oe[dim_num] = b[dim_num] + (length * (part_num + 1)) / num_part;

	return std::forward_as_tuple(ob, oe);
}

//! @}
}// namespace simpla

#endif /* GEOMETRY_ALGORITHM_H_ */
