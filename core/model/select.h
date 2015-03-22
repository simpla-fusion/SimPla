/*
 * select.h
 *
 *  Created on: 2014年12月2日
 *      Author: salmon
 */

#ifndef CORE_MODEL_SELECT_H_
#define CORE_MODEL_SELECT_H_
#include "../numeric/pointinpolygon.h"
#include <algorithm>
namespace simpla
{
template<typename TFun, typename TRes>
void filter(TFun const & pred, TRes *res)
{
	res->erase(std::remove_if(res->begin(), res->end(), pred), res->end());
}
template<typename TFun, typename TRange, typename TRes>
void filter(TFun const & pred, TRange const & range, TRes *res)
{
	std::copy_if(range.begin(), range.end(), std::inserter(*res, res->begin()),
			pred);

}

template<typename TM, typename ... Args>
void select_points_in_polylines(TM const & mesh,
		std::vector<typename TM::coordinates_type>const & poly_lines, int ZAXIS,
		Args && ... args)
{
	PointInPolygon checkPointsInPolygen(poly_lines, ZAXIS);

	filter([&](typename TM::id_type const &s)
	{	return checkPointsInPolygen(mesh.coordinates(s));},
			std::forward<Args>(args)...);
}

template<typename TM, typename ...Args>
void select_points_in_rectangle(TM const& mesh,
		typename TM::coordinates_type const & v0,
		typename TM::coordinates_type const & v1, Args && ...args)
{

	filter([&](typename TM::coordinates_type const &x)
	{
		return (((v0[0] - x[0]) * (x[0] - v1[0])) >= 0)
		&& (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
		&& (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0);
	}, std::forward<Args>(args)...);

}

template<typename TM, typename ...Args>
void select_line_segment(TM const& mesh,
		typename TM::coordinates_type const & x0,
		typename TM::coordinates_type const & x1, Args && ...args)
{

	auto dx = mesh.dx();

	Real dl = inner_product(dx, dx);
	filter([=]( typename TM::coordinates_type const & x )->bool
	{	Real l2 = inner_product(x1 - x0, x1 - x0);

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
	}, std::forward<Args>(args) ...);

}

template<typename TM, typename ...Args>
void select_polylines(TM const & mesh,
		std::vector<typename TM::coordinates_type> const& g_points,
		bool is_inner, Args && ...args)
{
	typedef typename TM::coordinates_type coordinates_type;

	std::vector<coordinates_type> points;

	std::vector<coordinates_type> intersect_points;

	std::transform(g_points.begin(), g_points.end(), std::back_inserter(points),
			[&](coordinates_type const & x)
			{
				return mesh.coordinates_to_topology(x);
			});

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

		auto ib = intersect_points.rbegin();

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
				[&](coordinates_type const & xa,coordinates_type const & xb)
				{
					return dot(xb-xa,x1-x0)>0;
				});
		ib = intersect_points.end();

	}

}
template<typename TM, typename TDict, typename ...Args>
void select_by_config(TM const& mesh, TDict const & dict, Args && ...args)
{

	if (dict.is_function())
	{
		filter([=]( typename TM::id_type const &s)->bool
		{
			return (dict(mesh.id_to_coordinates(s)).template as<bool>());
		}, std::forward<Args>(args)...);
	}
	else if (dict["Rectangle"])
	{
		std::vector<typename TM::coordinates_type> points;

		dict.as(&points);

		select_points_in_rectangle(mesh, points[0], points[1],
				std::forward<Args>(args)...);

	}
	else if (dict["Polyline"])
	{
		auto obj = dict["Polyline"];

		int ZAXIS = 0;

		std::vector<typename TM::coordinates_type> points;

		obj["ZAXIS"].as(&ZAXIS);

		obj["Points"].as(&points);

		if (obj["OnlyEdge"])
		{
			select_polylines(mesh, points, ZAXIS, std::forward<Args>(args)...);
		}
		else
		{
			select_points_in_polylines(mesh, points, ZAXIS,
					std::forward<Args>(args)...);
		}
	}

}
}
// namespace simpla

#endif /* CORE_MODEL_SELECT_H_ */
