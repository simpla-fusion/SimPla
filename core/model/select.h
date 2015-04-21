/*
 * select.h
 *
 *  Created on: 2014年12月2日
 *      Author: salmon
 */

#ifndef CORE_MODEL_SELECT_H_
#define CORE_MODEL_SELECT_H_
#include <algorithm>

#include "../numeric/point_in_polygon.h"
namespace simpla
{
template<typename TPred, typename InOut>
void filter(TPred const & pred, InOut *res)
{
//	res->erase(std::remove_if(res->begin(), res->end(), pred), res->end());
}
template<typename TPred, typename IN, typename OUT>
void filter(TPred const & pred, IN const & range, OUT *res)
{
	for (auto s : range)
	{
		if (pred(s))
		{
			res->insert(s);
		}
	}
//	std::copy_if(range.begin(), range.end(), std::inserter(*res, res->begin()),
//			pred);

}

template<typename TM, typename ...Args>
void select_ids_in_rectangle(TM const& mesh,
		typename TM::coordinates_type const & v0,
		typename TM::coordinates_type const & v1, Args && ...args)
{

	filter([&](typename TM::id_type const &s)
	{	CHECK(s);
		auto x=mesh.coordinates(s);
		CHECK(x);
		return (((v0[0] - x[0]) * (x[0] - v1[0])) >= 0)
		&& (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
		&& (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0);

	}, std::forward<Args>(args)...);

}

template<typename TM, typename ...Args>
void select_ids_by_line_segment(TM const& mesh,
		typename TM::coordinates_type const & x0,
		typename TM::coordinates_type const & x1, Args && ...args)
{

	auto dx = mesh.dx();

	Real dl = inner_product(dx, dx);
	filter([&](typename TM::id_type const &s)
	{
		auto x=mesh.coordinates(s);
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
	}, std::forward<Args>(args) ...);

}
template<typename TM, typename ... Args>
void select_ids_in_polylines(TM const & mesh,
		std::vector<typename TM::coordinates_type>const & poly_lines, int ZAXIS,
		Args && ... args)
{
	PointInPolygon checkPointsInPolygen(poly_lines, ZAXIS);

	filter([&](typename TM::id_type const &s)
	{	return checkPointsInPolygen(mesh.coordinates(s));},
			std::forward<Args>(args)...);
}

template<typename TM, typename ...Args>
void select_ids_on_polylines(TM const & mesh,
		std::vector<typename TM::coordinates_type> const& g_points,
		bool is_inner, Args && ...args)
{
	typedef typename TM::coordinates_type coordinates_type;

	std::vector<coordinates_type> points;

	std::vector<coordinates_type> intersect_points;

	std::transform(g_points.begin(), g_points.end(), std::back_inserter(points),
			[&](coordinates_type const & x)
			{
				return x;
				//mesh.coordinates_to_topology(x);
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
template<typename TDomain, typename TDict>
TDomain filter_by_config(TDomain const & domain, TDict const & dict)
{
	TDomain res(domain);

	typedef TDomain domain_type;
	typedef typename domain_type::mesh_type mesh_type;

	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::id_type id_type;
	typedef typename mesh_type::index_tuple index_tuple;
	static constexpr size_t iform = domain_type::iform;
	mesh_type const & mesh = domain.mesh();
	auto & id_set = res.id_set();

	if (dict.is_function())
	{
		auto pred = [&](id_type s)
		{
			return (dict(mesh.coordinates(s)).template as<bool>());
		};

		if (id_set.size() > 0)
		{
			filter(pred, &id_set);
		}
		else
		{
			filter(pred, domain, &id_set);
		}
	}
	else if (dict["Polylines"])
	{
		auto obj = dict["Polylines"];

		int ZAXIS = 0;

		std::vector<coordinates_type> points;

		obj["Polylines"]["ZAXIS"].as(&ZAXIS);

		obj["Polylines"]["Points"].as(&points);

		PointInPolygon checkPointsInPolygen(points, ZAXIS);

		auto pred = [&]( id_type const &s)
		{	return checkPointsInPolygen(mesh.coordinates(s));};

		if (id_set.size() > 0)
		{
			filter(pred, &id_set);
		}
		else
		{
			filter(pred, domain, &id_set);
		}
	}
	else if (dict["Box"])
	{
		std::vector<coordinates_type> points;

		dict["Rectangle"].as(&points);

//		res.select(points[0], points[1]);

	}
	else if (dict["Indices"])
	{
		std::vector<index_tuple> points;

		dict["Indices"].as(&points);

		for (auto const & i : points)
		{
			id_set.insert(mesh.template pack<iform>(i));
		}
	}

	return std::move(res);
}
}
// namespace simpla

#endif /* CORE_MODEL_SELECT_H_ */
