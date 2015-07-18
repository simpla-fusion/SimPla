/**
 * @file demo_model.cpp
 *
 * @date 2015年3月16日
 * @author salmon
 */

#include <memory>
#include <vector>

#include "../../core/application/application.h"
#include "../../core/application/use_case.h"
#include "../../core/gtl/ntuple.h"
#include "../../core/gtl/primitives.h"
#include "../../core/io/io.h"
#include "../../core/mesh/mesh.h"
#include "utilities.h"
namespace simpla
{
typedef StructuredMesh<geometry::coordinate_system::Cartesian<3>,
		InterpolatorLinear, FiniteDiffMethod> mesh_type;

USE_CASE(model,"Model")
{
	static constexpr Real PI = 3.1415926535;
	typedef typename mesh_type::id_type id_type;
	typedef typename mesh_type::point_type point_type;
	auto mesh = std::make_shared<mesh_type>();

	mesh->load(options["Mesh"]);

	mesh->deploy();

	typedef typename mesh_type::point_type point_type;

	typedef typename mesh_type::id_type id_type;

	std::vector<point_type> pp, p0, p1, p1b, p2, p2b, p3;

	std::set<id_type> res;
	auto extents = mesh->extents();
	point_type x0;
	x0 = (std::get<0>(extents) + std::get<1>(extents)) * 0.5;

	mesh->select(

	options["Object"],

	options["SelectTag"].as<int>(0),

	static_cast<ManifoldTypeID>(options["SelectIForm"].as<int>(0)),

	std::get<0>(extents), std::get<1>(extents),

	&res

	);

	for (id_type s : res)
	{
		p0.push_back(mesh->point(s));
	}

	LOGGER << SAVE(p0) << std::endl;
//	if (options["GFile"])
//	{
//		GEqdsk geqdsk(options["GFile"].as<std::string>());
//
//		auto const & glimter = geqdsk.limiter();
//
//		dims = geqdsk.dimensins();
//
//		point_type xmin, xmax;
//
//		std::tie(xmin, xmax) = geqdsk.extents();
//
//		point_type L;
//
//		L = dims / (xmax - xmin);
//
//		std::transform(glimter.begin(), glimter.end(), std::back_inserter(p0),
//				[&](point_type const & x)
//				{
//					return (x-xmin)*L;
//				});
//
//	}
//	else
//
//	int ZAXIS = options["Domain"]["ZAXIS"].template as<int>(2);
//
//
//	options["Domain"]["Polylines"].as(&pp);
//
//	auto domain0 = mesh->template domain<VERTEX>();
//	auto domain1 = mesh->template domain<EDGE>();
//	auto domain1b = mesh->template domain<EDGE>();
//	auto domain2b = mesh->template domain<FACE>();
//	auto domain2 = mesh->template domain<FACE>();
//	auto domain3 = mesh->template domain<VOLUME>();
//	select_boundary_by_polylines(&(domain0), pp.begin(), pp.end(), ZAXIS);
//
//	select_boundary_by_polylines(&(domain1), pp.begin(), pp.end(), ZAXIS, 0);
//
//	select_boundary_by_polylines(&(domain1b), pp.begin(), pp.end(), ZAXIS, 1);
//
//	select_boundary_by_polylines(&(domain2), pp.begin(), pp.end(), ZAXIS, 0);
//
//	select_boundary_by_polylines(&(domain2b), pp.begin(), pp.end(), ZAXIS, 1);
//
//	select_boundary_by_polylines(&(domain3), pp.begin(), pp.end(), ZAXIS, 2);
//	domain0.for_each_coordinates([&](point_type const &x)
//	{
//		p0.push_back(x);
//	});
//
//	domain0.for_each_coordinates([&](point_type const &x)
//	{
//		p0.push_back(x);
//	});
//
//	domain1.for_each_coordinates([&](point_type const &x)
//	{
//		p1.push_back(x);
//	});
//
//	domain1b.for_each_coordinates([&](point_type const &x)
//	{
//		p1b.push_back(x);
//	});
//
//	domain2.for_each_coordinates([&](point_type const &x)
//	{
//		p2.push_back(x);
//	});
//
//	domain2b.for_each_coordinates([&](point_type const &x)
//	{
//		p2b.push_back(x);
//	});
//
//	domain3.for_each_coordinates([&](point_type const &x)
//	{
//		p3.push_back(x);
//	});
////

//
//	find_boundary2D(p3, shift0, &p4);
//
//	point_type shift1 = { 0.5, 0.5, 0.5 };
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

}
//namespace simpla

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
//int boundary_tag(point_type const & x)
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

//void divid_to_boundary(std::vector<point_type> const& points,
//		std::vector<boundary_s> * res, int ZAXIS = 2)
//{
//
//	auto i0 = points.rbegin();
//	auto i1 = points.begin();
//	auto i2 = ++points.begin();
//
//	point_type x0, x1, x2;
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
//		point_type xp, xm;
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
////		point_type xp = x0;
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
//template<typename TPoints>
//void get_intersctions(TPoints const & points, point_type const & shift,
//		std::vector<point_type> *res)
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
//				[&](point_type const & xa,point_type const & xb)->bool
//				{
//					return inner_product(xb-xa,x1-x0)>0;
//				});
//
//	}
//}
//typedef point_type Vec3;
//
//template<typename TPoints>
//void find_boundary2D(TPoints const & points, point_type const & shift,
//		std::vector<point_type> *res, int ZAXIS = 2)
//{
//
//	auto i0 = points.begin();
//
//	while (i0 != points.end())
//	{
//		point_type p0 = *i0;
//		auto i1 = ++i0;
//
//		if (i1 == points.end())
//		{
//			i1 = points.begin();
//		}
//		point_type p1 = *i1;
//
//		point_type p2;
//		p2 = p1;
//
//		p2[ZAXIS] += 1.0;
//
//		point_type x0 = //
//				{ //
//				std::floor(std::min(p0[0], p1[0]) - shift[0]) + shift[0], //
//				std::floor(std::min(p0[1], p1[1]) - shift[1]) + shift[1], //
//				std::floor(std::min(p0[2], p1[2]) - shift[2]) + shift[2]     //
//				};
//		x0 += 0.5;
////		id_type s = MeshDummy::coordinates_to_id(x0);
//
//		res->push_back(x0);
//		point_type n;
//		std::tie(std::ignore, n) = distance_point_to_plane(x0, p0, p1, p2);
//		res->push_back(n);
//
//	}
//
//}
