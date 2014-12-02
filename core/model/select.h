/*
 * select.h
 *
 *  Created on: 2014年12月2日
 *      Author: salmon
 */

#ifndef CORE_MODEL_SELECT_H_
#define CORE_MODEL_SELECT_H_
#include "../manifold/domain.h"
#include "../numeric/pointinpolygon.h"
namespace simpla
{
template<typename TD, typename TDict>
SubDomain<
		Domain<typename domain_traits<TD>::manifold_type,
				domain_traits<TD>::iform> > select(TD const & domain,
		TDict const & dict)
{

	for (auto s : domain)
	{
	}
	return std::move(select_by_config(domain, dict));

}

template<typename TD, typename TFun>
SubDomain<Domain<typename domain_traits<TD>::manifold_type, //
		domain_traits<TD>::iform> > select_by_function(TD const &domain,
		TFun const & pred)
{
	SubDomain<
			Domain<typename domain_traits<TD>::manifold_type,
					domain_traits<TD>::iform> > res(domain.manifold());

	for (auto s : domain)
	{
		if (pred(s))
		{
			res.push_back(s);
		}
	}
	return std::move(res);

}
template<typename TD>
SubDomain<Domain<typename domain_traits<TD>::manifold_type, //
		domain_traits<TD>::iform> > select_by_polylines(TD const& domain,
		PointInPolygon const&checkPointsInPolygen)
{
	typedef typename domain_traits<TD>::index_type index_type;

	return std::move(select_by_function(domain, [&]( index_type s )->bool
	{	return (checkPointsInPolygen(domain.coordinates(s) ));}));
}

template<typename TD, typename TC>
SubDomain<Domain<typename domain_traits<TD>::manifold_type, //
		domain_traits<TD>::iform> > select_by_NGP(TD const& domain,
		TC const & x)
{

	typedef typename domain_traits<TD>::index_type index_type;

	index_type dest;

	std::tie(dest, std::ignore) =
			domain.manifold()->coordinates_global_to_local(x);

	if (domain.manifold()->in_local_range(dest))
	{
		return std::move(
				select_by_function(domain, [&]( index_type const & s )->bool
				{
					return domain.manifold()->get_cell_index(s)
					==domain.manifold()->get_cell_index(dest);
				}));

	}
	else
	{
		return std::move(
				SubDomain<Domain<typename domain_traits<TD>::manifold_type, //
						domain_traits<TD>::iform> >(domain.manifold()));
	}

}
template<typename TD, typename TC>
SubDomain<Domain<typename domain_traits<TD>::manifold_type, //
		domain_traits<TD>::iform> > select_by_rectangle(TD const& domain, TC v0,
		TC v1)
{
	typedef typename domain_traits<TD>::index_type index_type;

	return std::move(
			select_by_function(domain,
					[&]( index_type const & s )->bool
					{
						auto x = domain.coordinates(s);
						return ((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0)
								&& (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
								&& (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0));
					}));
}

template<typename TD, typename TC>
SubDomain<Domain<typename domain_traits<TD>::manifold_type, //
		domain_traits<TD>::iform> > select_by_points(TD const& domain,
		std::vector<TC>const & points)
{
	typedef SubDomain<Domain<typename domain_traits<TD>::manifold_type, //
			domain_traits<TD>::iform> > result_type;
	if (points.size() == 1)
	{
		return std::move(select_by_NGP(domain, points[0]));
	}
	else if (points.size() == 2)
	{
		return std::move(select_by_rectangle(domain, points[0], points[1]));
	}
	else if (points.size() > 2)
	{
		PointInPolygon poly(points);
		return std::move(select_by_polylines(domain, poly));
	}
	else
	{
		return std::move(result_type(domain.manifold()));
	}
}
template<typename TD, typename TDict>
SubDomain<Domain<typename domain_traits<TD>::manifold_type, //
		domain_traits<TD>::iform> > select_by_config(TD const &domain,
		TDict const & dict)
{
	typedef SubDomain<Domain<typename domain_traits<TD>::manifold_type, //
			domain_traits<TD>::iform> > result_type;

	typedef typename domain_traits<TD>::index_type index_type;
	typedef typename domain_traits<TD>::coordinates_type coordinates_type;

	if (dict.is_function())
	{
		return std::move(
				select_by_function(domain, [&]( index_type const & s )->bool
				{
					auto x=domain.coordinates(s);
					return (dict(x).template as<bool>());
				}));

	}
	else if (dict["Points"].is_table())
	{
		std::vector<coordinates_type> points;

		dict["Points"].as(&points);

		return std::move(select_by_points(domain, points));
	}
	else
	{
		PARSER_ERROR("Invalid 'Select' options");

		return std::move(result_type(domain.manifold()));
	}

}

}
// namespace simpla

#endif /* CORE_MODEL_SELECT_H_ */
