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

template<typename TM, typename TRange, typename TFun>
std::vector<typename TM::id_type> select_by_function(TM const & mesh,
		TRange const &range, TFun const & pred)
{
	std::vector<typename TM::id_type> res;

	for (auto s : range)
	{
		if (pred(mesh.coordinates(s)))
		{
			res.push_back(s);
		}
	}
	return std::move(res);

}
template<typename TM, typename TRange>
std::vector<typename TM::id_type> select_by_polylines(TM const & mesh,
		TRange const& range, PointInPolygon const&checkPointsInPolygen)
{
	std::vector<typename TM::id_type> res;

	for (auto s : range)
	{
		if (checkPointsInPolygen(mesh.coordinates(s)))
		{
			res.push_back(s);
		}
	}
	return std::move(res);
}
//template<typename TM, typename TRange>
//std::vector<typename TM::id_type> select_by_NGP(TM const & mesh,
//		TRange const & range, typename TM::coordinates_type const & x)
//{
//	std::vector<typename TM::id_type> res;
//
//	typename TM::id_type dest;
//
//	res.push_back(mesh.coordinates_to_id(x));
//
//	return std::move(res);
//}
template<typename TM, typename TRange>
std::vector<typename TM::id_type> select_by_rectangle(TM const& mesh,
		TRange const &range, typename TM::coordinates_type const & v0,
		typename TM::coordinates_type const & v1)
{
	std::vector<typename TM::id_type> res;

	for (auto s : range)
	{

		auto x = mesh.coordinates(s);

		if ((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0)
				&& (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
				&& (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0))
		{
			res.push_back(s);
		}

	}
	return std::move(res);

}

template<typename TM, typename TRange, typename TC>
std::vector<typename TM::id_type> select_by_vertics(TM const& mesh,
		TRange const &range, std::vector<TC>const & points)
{

//	if (points.size() == 1)
//	{
//		return std::move(select_by_NGP(mesh, range, points[0]));
//	}
//	else
	if (points.size() == 2)
	{
		return std::move(select_by_rectangle(mesh, range, points[0], points[1]));
	}
	else if (points.size() > 2)
	{
		PointInPolygon poly(points);
		return std::move(select_by_polylines(mesh, range, poly));
	}

	return std::move(std::vector<typename TM::id_type>());

}

template<typename TM, typename TRange, typename TDict>
std::vector<typename TM::id_type> select_by_config(TM const& mesh,
		TRange const &range, TDict const & dict)
{

	if (dict.is_function())
	{
		return std::move(
				select_by_function(mesh, range,
						[=]( typename TM::coordinates_type const & x )->bool
						{
							return (dict(x).template as<bool>());
						}));

	}
	else if (dict.is_table())
	{
		std::vector<typename TM::coordinates_type> points;

		dict.as(&points);

		return std::move(select_by_vertics(mesh, range, points));
	}

	return std::move(std::vector<typename TM::id_type>());
}

}
// namespace simpla

#endif /* CORE_MODEL_SELECT_H_ */
