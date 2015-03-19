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
#include "../../core/mesh/mesh_ids.h"
#include "../../core/model/model.h"
#include "../../core/numeric/pointinpolygon.h"
#include "../../core/model/geqdsk.h"

namespace simpla
{

static constexpr Real PI = 3.1415926535;
typedef typename MeshIDs::id_type id_type;
typedef typename MeshIDs::coordinates_type coordinates_type;

//template<typename TPoints>
//void get_intersctions(TPoints const & points, coordinates_type const & shift,
//		std::vector<coordinates_type> *res)
//{
//
//	auto first = points.begin();
//
//	while (first != points.end())
//	{
//		size_t ib = res->size();
//
//		auto x0 = *first;
//
//		auto second = ++first;
//
//		if (second == points.end())
//		{
//			second = points.begin();
//		}
//
//		auto x1 = *second;
//
//		for (int n = 0; n < 3; ++n)
//		{
//			nTuple<Real, 3> xp;
//
//			xp = 0;
//
//			Real dx = 1;
//
//			Real k1 = (x1[(n + 1) % 3] - x0[(n + 1) % 3]) / (x1[n] - x0[n]);
//			Real k2 = (x1[(n + 2) % 3] - x0[(n + 2) % 3]) / (x1[n] - x0[n]);
//
//			if (x1[n] > x0[n])
//			{
//				dx = 1;
//				xp[n] = std::floor(x0[n] - shift[n]) + shift[n] + 1;
//			}
//			else
//			{
//				dx = -1;
//				xp[n] = std::floor(x0[n] - shift[n]) + shift[n];
//			}
//			for (; (xp[n] - x0[n]) * (x1[n] - xp[n]) >= 0; xp[n] += dx)
//			{
//				xp[(n + 1) % 3] = (xp[n] - x0[n]) * k1 + x0[(n + 1) % 3];
//				xp[(n + 2) % 3] = (xp[n] - x0[n]) * k2 + x0[(n + 2) % 3];
//				res->push_back(xp);
//			}
//
//		}
//
//		size_t ie = res->size();
//
//		std::sort(&(*res)[ib], &(*res)[ie],
//				[&](coordinates_type const & xa,coordinates_type const & xb)->bool
//				{
//					return inner_product(xb-xa,x1-x0)>0;
//				});
//
//	}
//}
//typedef coordinates_type Vec3;
//
//template<typename TPoints>
//void find_boundary2D(TPoints const & points, coordinates_type const & shift,
//		std::vector<coordinates_type> *res, int ZAXIS = 2)
//{
//
//	auto i0 = points.begin();
//
//	while (i0 != points.end())
//	{
//		coordinates_type p0 = *i0;
//		auto i1 = ++i0;
//
//		if (i1 == points.end())
//		{
//			i1 = points.begin();
//		}
//		coordinates_type p1 = *i1;
//
//		coordinates_type p2;
//		p2 = p1;
//
//		p2[ZAXIS] += 1.0;
//
//		coordinates_type x0 = //
//				{ //
//				std::floor(std::min(p0[0], p1[0]) - shift[0]) + shift[0], //
//				std::floor(std::min(p0[1], p1[1]) - shift[1]) + shift[1], //
//				std::floor(std::min(p0[2], p1[2]) - shift[2]) + shift[2]     //
//				};
//		x0 += 0.5;
////		id_type s = MeshDummy::coordinates_to_id(x0);
//
//		res->push_back(x0);
//		coordinates_type n;
//		std::tie(std::ignore, n) = distance_point_to_plane(x0, p0, p1, p2);
//		res->push_back(n);
//
//	}
//
//}
template<typename T0, typename T1, typename T2, typename T3>
std::tuple<Real, Real> line_intersection(T0 const& P0, T1 const & P1,
		T2 const & Q0, T3 const & Q1)
{
	Real s = 0.0;
	Real t = 0.0;

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
		s = std::numeric_limits<double>::max();
		t = std::numeric_limits<double>::max();

	}
	else
	{
		s = (b * e - c * d) / (a * c - b * b);

		t = (a * e - b * d) / (a * c - b * b);
	}

	return std::make_tuple(s, t);
}

/**
 *    Case 0 : vertex in cell
 *
 *
 *                 o p2
 *              s1/
 *      q3-------/-------q2
 *       |      /        |
 *       |     /         |
 *       |    /          |s0
 *       |p1 o-----------+---o p0
 *       |               |
 *      q0---------------q1
 *
 *
 */

/**
 *     Case 1 : line cut cell
 *
 *                 o p1
 *              s1/
 *      q3-------/-------q2
 *       |      /        |
 *       |     /         |
 *       |    /          |s0
 *       |   /           |
 *      q0--/------------q1
 *         /s0
 *        o
 *       p0
 *
 *              o p1
 *             /
 *            /s1
 *       q3--/-------q2
 *        | /        |
 *        |/         |
 *     s0 /          |
 *       /|          |
 *      / q0--------q1
 *     /
 *    o p1
 */
void line_segment_intersect_grid(coordinates_type x0, coordinates_type x1,
		coordinates_type const & shift, std::vector<id_type> *ids,
		std::vector<coordinates_type> *out_points, const int ZAXIS = 2)
{
	const int XAXIS = (ZAXIS + 1) % 3;
	const int YAXIS = (ZAXIS + 2) % 3;

	bool keep_looking = true;

	id_type cell_id = MeshIDs::coordinates_to_id(coordinates_type {

	std::floor(x0[0] + shift[0]),

	std::floor(x0[1] + shift[1]),

	std::floor(x0[2] + shift[2]) });

	static const id_type id_edge[4] = {

	MeshIDs::_DI,

	(MeshIDs::_DJ) | (MeshIDs::_DI << 1),

	(MeshIDs::_DJ << 1) | (MeshIDs::_DI),

	MeshIDs::_DJ

	};
	static const id_type id_cell_shift[4] = {

	-(MeshIDs::_DJ << 1),

	(MeshIDs::_DI << 1),

	(MeshIDs::_DJ << 1),

	-(MeshIDs::_DI << 1)

	};

	while (keep_looking)
	{

		coordinates_type q[4];

		q[0] = MeshIDs::id_to_coordinates(cell_id);
		q[1] = q[0];
		q[2] = q[0];
		q[3] = q[0];

		q[1][XAXIS] += 1.0;
		q[2][XAXIS] += 1.0;
		q[2][YAXIS] += 1.0;
		q[3][YAXIS] += 1.0;

		keep_looking = false;

		for (int i = 0; i < 4; ++i)
		{
			Real s, t;

			std::tie(s, t) = line_intersection(q[i], q[(i + 1) % 4], x0, x1);

//			VERBOSE << (i) << " " << s << "\t" << q[i] << q[(i + 1) % 4]
//					<< std::endl;

			if (s >= 0.0 && s <= 1.0 && t > EPSILON && t <= 1.0)
			{

				x0 = q[i] + s * (q[(i + 1) % 4] - q[i]);

				VERBOSE << s << "\t" << x0 << std::endl;

				keep_looking = true;

				ids->push_back(cell_id + id_edge[i]);

				out_points->push_back(x0);

				cell_id += id_cell_shift[i];

				break;
			}
		}

	}

}

void polyline_intersect_grid(std::vector<coordinates_type> const & polygons,
		coordinates_type const & shift, std::vector<id_type> *ids,
		std::vector<coordinates_type> *out_points, const int ZAXIS = 2)
{

	auto it = polygons.begin();

	while (it != polygons.end())
	{
		coordinates_type p0 = *it;
		coordinates_type p1;
		++it;

		if (it == polygons.end())
		{

			p1 = *polygons.begin();
		}
		else
		{
			p1 = *it;
		}

		line_segment_intersect_grid(p0, p1, shift, ids, out_points, ZAXIS);
	}
}

SP_APP(model)
{

	typedef typename MeshIDs::coordinates_type coordinates_type;

	typedef typename MeshIDs::id_type id_type;

	std::vector<coordinates_type> p0, p1, p2, p3, p4, p5, p6;

	nTuple<size_t, 3> dims = { 10, 10, 10 };

	if (options["GFile"])
	{
		GEqdsk geqdsk(options["GFile"].as<std::string>());

		auto const & glimter = geqdsk.limiter();

		dims = geqdsk.dimensins();

		coordinates_type xmin, xmax;

		std::tie(xmin, xmax) = geqdsk.extents();

		coordinates_type L;

		L = dims / (xmax - xmin);

		std::transform(glimter.begin(), glimter.end(), std::back_inserter(p0),
				[&](coordinates_type const & x)
				{
					return (x-xmin)*L;
				});

	}
	else
	{
		options["Points"].as(&p0);
		options["Dimensions"].as(&dims);
	}

	if (p0.empty())
	{
		return;
	}

	PointInPolygon p_in_p(p0);

	std::set<id_type> volume_in_side;

	for (size_t i = 0; i < dims[0]; ++i)
	{
		for (size_t j = 0; j < dims[1]; ++j)
		{
			for (size_t k = 0; k < dims[2]; ++k)
			{
				coordinates_type x;
				x[0] = static_cast<Real>(i) + 0.5;
				x[1] = static_cast<Real>(j) + 0.5;
				x[2] = static_cast<Real>(k) + 0.5;

				if (p_in_p(x))
					volume_in_side.insert(MeshIDs::coordinates_to_id(x));
			}
		}
	}

//	std::set<id_type> volume_in_side;
//
//	std::copy_if(volume_ids.begin(), volume_ids.end(),
//			std::inserter(volume_in_side, volume_in_side.begin()), [&](id_type const &s )
//			{
//				return p_in_p(MeshIDs::id_to_coordinates(s));
//			});

	std::transform(volume_in_side.begin(), volume_in_side.end(),
			std::back_inserter(p1), [&](id_type const &s )
			{
				return std::move(MeshIDs::id_to_coordinates(s));
			});

	Model model;

	model.set(volume_in_side, Model::VACUUM);

	std::set<id_type> boundary_face;

	for (size_t i = 0; i < dims[0]; ++i)
	{
		for (size_t j = 0; j < dims[1]; ++j)
		{
			for (size_t k = 0; k < dims[2]; ++k)
			{
				for (size_t n = 0; n < 3; ++n)
				{
					size_t s = MeshIDs::id<2>(i, j, k, n);

					if (model.check_boundary_surface(s, Model::VACUUM))
					{
						boundary_face.insert(s);
					}

				}
			}
		}
	}

	std::transform(boundary_face.begin(), boundary_face.end(),
			std::back_inserter(p2), [&](id_type const &s )
			{
				return std::move(MeshIDs::id_to_coordinates(s));
			});

	std::set<id_type> boundary_edge;

	for (auto s : boundary_face)
	{
		size_t d[3] = { (s) & MeshIDs::_DI, (s) & MeshIDs::_DJ, (s)
				& MeshIDs::_DK };

		for (int i = 0; i < 3; ++i)
		{
			if (d[i] != 0UL)
			{
				boundary_edge.insert(s - d[i]);
				boundary_edge.insert(s + d[i]);
			}
		}

	}

	std::transform(boundary_edge.begin(), boundary_edge.end(),
			std::back_inserter(p3), [&](id_type const &s )
			{
				return std::move(MeshIDs::id_to_coordinates(s));
			});

	p0.push_back(p0.front());

	LOGGER << SAVE(p0) << std::endl;
	LOGGER << SAVE(p1) << std::endl;
	LOGGER << SAVE(p2) << std::endl;
	LOGGER << SAVE(p3) << std::endl;
	coordinates_type shift0 = { 0, 0, 0 };

	std::vector<id_type> ids;

	polyline_intersect_grid(p0, shift0, &ids, &p4);

	std::transform(ids.begin(), ids.end(), std::back_inserter(p5),
			[&](id_type const &s )
			{
				return std::move(MeshIDs::id_to_coordinates(s));
			});

	LOGGER << SAVE(p4) << std::endl;
	LOGGER << SAVE(p5) << std::endl;
//
//	find_boundary2D(p3, shift0, &p4);
//
//	coordinates_type shift1 = { 0.5, 0.5, 0.5 };
//
//	get_intersctions(p0, shift1, &p5);
//
//	find_boundary2D(p5, shift1, &p6);
//
//	p3.push_back(p3.front());
//	LOGGER << SAVE(p3) << std::endl;
//
//	p5.push_back(p5.front());
//	LOGGER << SAVE(p5) << std::endl;
//
//	size_t tdims[2] = { p4.size() / 2, 2 };
//	LOGGER << save("p4", &p4[0], 2, tdims) << std::endl;
//
//	dims[0] = p6.size() / 2;
//	LOGGER << save("p6", &p6[0], 2, tdims) << std::endl;
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

