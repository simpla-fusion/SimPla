/*
 * select.h
 *
 *  Created on: 2014年12月2日
 *      Author: salmon
 */

#ifndef CORE_MODEL_SELECT_H_
#define CORE_MODEL_SELECT_H_
#include "../numeric/pointinpolygon.h"
namespace simpla
{
template<typename TM, typename TFun, typename TRes>
void select_by_function(TM const & mesh, TFun const & pred, TRes *res)
{
	auto it = res->begin();

	while (it != res->end())
	{
		if (!pred(mesh.coordinates(*it)))
		{
			it = res->erase(it);
		}
		else
		{
			++it;
		}
	}
}
template<typename TM, typename TFun, typename TRange, typename TRes>
void select_by_function(TM const & mesh, TFun const & pred,
		TRange const & range, TRes *res)
{
	for (auto const & s : range)
	{
		if (!pred(mesh.coordinates(s)))
		{
			res->insert(s);
		}
	}

}
template<typename TM, typename ... Args>
void select_points_in_polylines(TM const & mesh,
		std::vector<typename TM::coordinates_type>const & poly_lines, int ZAXIS,
		Args && ... args)
{
	PointInPolygon checkPointsInPolygen(poly_lines, ZAXIS);

	select_by_function(mesh, [&](typename TM::coordinates_type const &x)
	{	return checkPointsInPolygen(x);},

	std::forward<Args>(args)...);
}

template<typename TM, typename ...Args>
void select_points_in_rectangle(TM const& mesh,
		typename TM::coordinates_type const & v0,
		typename TM::coordinates_type const & v1, Args && ...args)
{

	select_by_function(mesh, [&](typename TM::coordinates_type const &x)
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
	select_by_function(mesh,
			[=]( typename TM::coordinates_type const & x )->bool
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
		std::vector<typename TM::coordinates_type> const& polylines,
		bool is_inner, Args && ...args)
{

	PointInPolygon checkPointsInPolygen(polylines);

	auto dx = mesh.dx();
	Real d2 = inner_product(dx, dx);

	select_by_function(mesh,
			[&]( typename TM::coordinates_type const & x )->bool
			{
				auto first = polylines.begin();

				while (first != polylines.end())
				{
					auto second = first;

					++second;

					if (second == polylines.end())
					{
						second = polylines.begin();
					}

					auto x0=*first;

					auto x1=*second;

					++first;

					Real l2 = inner_product(x1 - x0, x1 - x0);

					Real t = inner_product(x - x0, x1 - x0) / l2;

					if (0 <= t && t <= 1)
					{
						nTuple<Real, 3> d;

						d = x - x0 - (t * (x1 - x0) + x0);

						if(inner_product(d, d) <= d2)
						{
							return checkPointsInPolygen(x) == is_inner;
						}
					}

				}

				return false;
			}, //
			std::forward<Args>(args) ...);
}
template<typename TM, typename TDict, typename ...Args>
void select_by_config(TM const& mesh, TDict const & dict, Args && ...args)
{

	if (dict.is_function())
	{
		select_by_function(mesh,
				[=]( typename TM::coordinates_type const & x )->bool
				{
					return (dict(x).template as<bool>());
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
