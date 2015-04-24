/**
 * @file select.h
 *
 *  Created on: 2014年12月2日
 *      Author: salmon
 */

#ifndef CORE_MODEL_SELECT_H_
#define CORE_MODEL_SELECT_H_
#include <algorithm>

#include "../numeric/point_in_polygon.h"
#include "../numeric/geometric_algorithm.h"
#include "../mesh/mesh_ids.h"
namespace simpla
{

template<typename, size_t> struct Domain;

//template<typename TPred, typename InOut>
//void filter(TPred const & pred, InOut *res)
//{
////	res->erase(std::remove_if(res->begin(), res->end(), pred), res->end());
//}
//template<typename TPred, typename IN, typename OUT>
//void filter(TPred const & pred, IN const & range, OUT *res)
//{
//	for (auto s : range)
//	{
//		if (pred(s))
//		{
//			res->insert(s);
//		}
//	}
////	std::copy_if(range.begin(), range.end(), std::inserter(*res, res->begin()),
////			pred);
//
//}

template<typename TCoord>
void select_ids_in_rectangle(TCoord const & v0, TCoord const & v1)
{

	return [&](TCoord const &x)
	{
		return (((v0[0] - x[0]) * (x[0] - v1[0])) >= 0)
		&& (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
		&& (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0);

	};

}

template<typename TCoord>
void select_ids_by_line_segment(TCoord const & x0, TCoord const & x1,
		TCoord const & dx)
{

	Real dl = inner_product(dx, dx);

	return

	[&]( TCoord const & x)
	{

		Real l2 = inner_product(x1 - x0, x1 - x0);

		Real t = inner_product(x - x0, x1 - x0) / l2;

		if (0 <= t && t <= 1)
		{
			nTuple<Real, 3> d;

			d = x - x0 - (t * (x1 - x0) + x0);
			return (inner_product(d, d) <= dl);
		}
		else
		{
			return false;
		}
	};

}
template<typename TCoord>
std::function<bool(TCoord const &)> select_ids_in_polylines(
		std::vector<TCoord>const & poly_lines, int ZAXIS, bool flag = true)
{
	PointInPolygon checkPointsInPolygen(poly_lines, ZAXIS);

	return [&](TCoord const &x)
	{	return checkPointsInPolygen(x) == flag;};
}

template<typename TCoord>
std::function<bool(TCoord const &)> select_ids_on_polylines(
		std::vector<TCoord> const& g_points, int ZAXIS, bool on_left = true)
{
	typedef TCoord coordinates_type;

	std::vector<coordinates_type> points;

	std::vector<coordinates_type> intersect_points;

	auto first = points.begin();

	while (first != points.end())
	{

		auto second = first;

		++second;

		if (second == points.end())
		{
			second = points.begin();
		}

		auto x0 = *first;

		auto x1 = *second;

		++first;

		auto ib = intersect_points.end();

		for (int n = 0; n < 3; ++n)
		{
			nTuple<Real, 3> xp;

			xp = 0;

			Real dx = std::copysign(1, x1[n] - x0[n]);

			Real k1 = (x1[(n + 1) % 3] - x0[(n + 1) % 3]) / (x1[n] - x0[n]);
			Real k2 = (x1[(n + 2) % 3] - x0[(n + 2) % 3]) / (x1[n] - x0[n]);

			for (xp[n] = std::floor(x0[n]) + 1;
					(xp[n] - x0[n]) * (x1[n] - xp[n]) >= 0; xp[n] += dx)
			{
				xp[(n + 1) % 3] = (xp[n] - x0[n]) * k1;
				xp[(n + 2) % 3] = (xp[n] - x0[n]) * k2;
				intersect_points.push_back(xp);
			}

		}
		++ib;

		std::sort(ib, intersect_points.end(),
				[&](coordinates_type const & xa, coordinates_type const & xb)
				{
					return dot(xb-xa,x1-x0)>0;
				});
		ib = intersect_points.end();

	}

}
template<typename TCoord, typename TDict>
std::function<bool(TCoord const &)> make_select_function_by_config(
		TDict const & dict)
{
	typedef std::function<bool(TCoord const &)> function_type;

	if (dict["Polylines"])
	{
		std::vector<TCoord> points;

		dict["Polylines"]["ZAXIS"].as(&points);

		int ZAXIS = dict["PointInPolylines"]["ZAXIS"].template as<int>(2);

		std::string place = dict["Polylines"]["Place"].template as<std::string>(
				"InSide");
		if (place == "InSide")
		{
			return select_ids_in_polylines(points, ZAXIS, true);
		}
		else if (place == "OutSide")
		{
			return select_ids_in_polylines(points, ZAXIS, false);
		}
		else if (place == "BoundaryLeft")
		{
			return select_ids_on_polylines(points, ZAXIS, true);
		}
		else if (place == "BoundaryRight")
		{
			return select_ids_on_polylines(points, ZAXIS, false);
		}
	}

}
/**
 *           o Q
 *          /
 *      D--o-----------C
 *      | / s1         |
 *      |/             |
 *    s0o      O       |
 *     /|              |
 *    / |              |
 * P o  A--------------B
 *
 *
 * O is the center of ABCD
 * |AB|=|CD|=1
 * min_radius = | OB |
 *
 * if( dist(O,PQ)> |OB| )
 *    no intersection
 * else
 *
 *
 *
 *
 * @param dict
 * @param domain
 */
template<typename TDict, typename TM, size_t IFORM>
void select_boundary(TDict const &dict, Domain<TM, IFORM> *domain)
{
	NEED_OPTIMIZATION;
	typedef TM mesh_type;
	typedef typename mesh_type::id_type id_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type const & mesh = domain->mesh();

	static constexpr size_t ndims = mesh_type::ndims;
	static constexpr size_t iform = IFORM;

	auto volume_domain = domain->template clone<VOLUME>();

	if (dict["Polylines"])
	{

		std::vector<coordinates_type> points;

		dict["Polylines"]["ZAXIS"].as(&points);

		for (auto & v : points)
		{
			v = mesh.coordinates_to_topology(v);
		}

		coordinates_type p0, p1;

		Real min_radius = std::sqrt(

		0.5 * 0.5

		+ ((ndims > 1) ? 0.5 * 0.5 : 0)

		+ ((ndims > 2) ? 0.5 * 0.5 : 0));

		for (auto s : volume_domain)
		{
			coordinates_type x = mesh_type::topology_type::id_to_coordinates(s);

			Real dist = 0, ss = 0;
			std::tie(dist, ss, p0, p1) = nearest_point_on_polylines(
					points.begin(), points.end(), x);

			if (dist > min_radius)
			{
				continue;
			}

		}

		int ZAXIS = dict["PointInPolylines"]["ZAXIS"].template as<int>(2);

		PointInPolygon checkPointsInPolygen(points, ZAXIS);

	}
}

}
// namespace simpla

#endif /* CORE_MODEL_SELECT_H_ */
