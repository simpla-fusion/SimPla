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
#include "../gtl/ntuple.h"
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

		if (place == "OutSide")
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
		else
		{
			return select_ids_in_polylines(points, ZAXIS, true);
		}

	}

}

/**
 *
 *
 *     x  o
 *       /|
 *      / | D
 *     o--a---------------o
 *  p0   s                      p1
 *
 *
 *  std::tie(ss,p0,p1,d)=std::tuple<Real, TI, TI, Vec3>
 *
 *  (ss ==0 ) ->  a==p0
 *   0<ss<1   ->  a \in (p0,p1)
 *  (ss ==1 ) ->  a==p1
 *
 *  d= cross(x-p0,p1-p0)/sqrt(|p1-p0|)
 *
 *  |d| = 0  -> x on line (p1,p0)
 *
 *  dot(d,normal_vector) >0  -> x on the right of (p1,p0)
 *
 *  VMAP= std::map<id_type,tuple<dist,ss,p0,p1> >
 */
template<typename TDomain, typename TI, typename VMAP>
void select_vetrices_near_to_polylines(TDomain const& domain, TI const & ib,
		TI const &ie, Real snap_radius, VMAP * res)
{
	typedef typename TDomain::mesh_type mesh_type;
	typedef typename mesh_type::id_type id_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::topology_type topology;

	mesh_type const & mesh = domain.mesh();

	domain.for_each([&](id_type const& s )
	{
		auto x= mesh.coordinates(s);

		TI p0,p1;

		Real dist,ss;

		auto v_tuple=distance_from_point_to_polylines( x,ib, ie);

		if(std::get<0>(v_tuple) <=snap_radius*2 )
		{
			(*res)[s] = v_tuple;
		}

	});
}
template<typename TDomain, typename PIP>
void select_cell_cross_polylines(PIP const & point_in_polygon, TDomain * domain,
		int flag = 0 /* 0 in side,1 out side,2 intersection */)
{
	typedef typename TDomain::mesh_type mesh_type;
	typedef typename mesh_type::id_type id_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::topology_type topology;

	mesh_type const & mesh = domain->mesh();

	domain->filter([&](id_type const& s )
	{
		id_type vertices[topology::MAX_NUM_OF_CELL];

		int num_of_vertices = topology::template get_adjoints<VERTEX>(s,
				vertices);

		int count_in = 0;
		int count_out = 0;
		for (int i = 0; i < num_of_vertices; ++i)
		{

			if (point_in_polygon(mesh.coordinates(vertices[i])))
			{
				++count_in;
			}
			else
			{
				++count_out;
			}

		}

		if(flag==0)
		{
			return count_out == 0;
		}
		else if(flag==1)
		{
			return count_in == 0;
		}
		else
		{
			return (count_in * count_out != 0);
		}

	});
}
template<typename TM, size_t IFORM, typename TI>
void select_boundary_by_polylines(Domain<TM, IFORM> *domain, TI const & ib,
		TI const &ie, int ZAxis = 2,
		int flag = 0 /* 0 in side,1 out side,2 intersection */)
{
	// FIXME This implement is O(N^2), NEED OPTIMIZATION;

	typedef TM mesh_type;
	typedef typename mesh_type::id_type id_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::topology_type topology;

	mesh_type const & mesh = domain->mesh();
	static constexpr size_t ndims = mesh_type::ndims;
	static constexpr size_t iform = IFORM;

	std::map<id_type, std::tuple<Real, Real, TI, TI>> vmap;

	Real snap_radius = std::sqrt(inner_product(mesh.dx(), mesh.dx()));

	select_vetrices_near_to_polylines(mesh.template domain<VERTEX>(), ib, ie,
			snap_radius, &vmap);

	PointInPolygon point_in_polygon(ib, ie, ZAxis);

//	if (iform == VERTEX)
//	{
//		for (auto const & item : vmap)
//		{
//			if (point_in_polygon(mesh.coordinates(item.first))
//					== (flag == 0 || flag == 2))
//			{
//				domain->id_set().insert(item.first);
//			}
//		}
//	}
//	else

	if (iform == VOLUME)
	{
		if (domain->is_simply())
		{
			for (auto const & item : vmap)
			{

				id_type cell[topology::MAX_NUM_OF_CELL];

				int num_of_cell = topology::template get_adjoints<VOLUME>(
						item.first, cell);

				for (int i = 0; i < num_of_cell; ++i)
				{
					domain->id_set().insert(cell[i]);
				}
			}
		}

		select_cell_cross_polylines(point_in_polygon, domain, flag);

	}
	else
	{

		auto v_domain = mesh.template domain<VOLUME>();

		if (v_domain.is_simply())
		{
			for (auto const & item : vmap)
			{

				id_type cell[topology::MAX_NUM_OF_CELL];

				int num_of_cell = topology::template get_adjoints<VOLUME>(
						item.first, cell);

				for (int i = 0; i < num_of_cell; ++i)
				{
					v_domain.id_set().insert(cell[i]);
				}
			}
		}

		select_cell_cross_polylines(point_in_polygon, &v_domain, 2);

		if (domain->is_simply())
		{
			v_domain.for_each([&](id_type s)
			{
				id_type cell[topology::MAX_NUM_OF_CELL];
				int num_of_cell = topology::template get_adjoints<iform>(
						s, cell);

				for (int i = 0; i < num_of_cell; ++i)
				{
					domain->id_set().insert(cell[i]);
				}
			});
		}
		select_cell_cross_polylines(point_in_polygon, domain, flag);

	}
	if (domain->id_set().size() == 0)
	{
		domain->clear();
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
	if (dict["Polylines"])
	{
		select_boundary_on_polylines(dict, domain);
	}
}

template<typename TDict, typename TDomain>
void filter_domain_by_config(TDict const & dict, TDomain * domain)
{
	typedef TDomain domain_type;

	typedef typename domain_type::mesh_type mesh_type;
	typedef typename mesh_type::id_type id_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	static constexpr int iform = domain_type::iform;

	static constexpr int ndims = domain_type::ndims;
	if (!dict)
	{
		return;
	}
	else if (dict["Box"])
	{
		std::vector<coordinates_type> p;

		dict["Box"].as(&p);

		domain->reset(p[0], p[1]);

	}
	else if (dict["IndexBox"])
	{

		std::vector<nTuple<index_type, ndims>> points;

		dict["IndexBox"].as(&points);

		domain->reset(points[0], points[1]);

	}

	if (dict["Indices"])
	{
		std::vector<
				nTuple<long,
						(iform == VERTEX || iform == VOLUME) ?
								ndims : (ndims + 1)>> points;

		dict["Indices"].as(&points);

		for (auto const & idx : points)
		{

			auto s = domain->pack_relative_index(idx);

			if (domain->in_box(s))
			{
				SHOW(domain->hash(s));
				SHOW(domain->max_hash());
				domain->id_set().insert(s);
			}

		}

		if (domain->id_set().size() == 0)
		{
			domain->clear();
		}

	}
	else if (dict["SelectCell"])
	{
//		select_cell(dict, this);
	}
	else if (dict["OnBoundary"])
	{

	}
	else
	{
		if (dict.is_function())
		{
			domain->filter_by_coordinates([&](coordinates_type const & x )
			{
				return (static_cast<bool>(dict(x)));
			});

		}
	}

}
}
// namespace simpla

#endif /* CORE_MODEL_SELECT_H_ */
