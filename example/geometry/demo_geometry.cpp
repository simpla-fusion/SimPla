/**
 * @file demo_geometry.cpp
 *
 * @date 2015-3-16
 * @author salmon
 */

//#include <stddef.h>
//#include <algorithm>
//#include <iostream>
//#include <iterator>
//#include <set>
//#include <string>
//#include <tuple>
//#include <vector>
//
//#include "../../core/task_flow/task_flow.h"
//#include "../../core/task_flow/use_case.h"
//#include "../../core/utilities/utilities.h"
//#include "../../core/io/io.h"
//#include "../../core/physics/physical_constants.h"
//
//#include "../../core/geometry/geometry.h"
//
//#include "../../core/model/cut_cell.h"
//#include <memory>
//using namespace simpla;
//
//typedef CartesianRectMesh mesh_type;
#include <iostream>

#include "../../core/geometry/geometry.h"
#include "../../core/gtl/utilities/utilities.h"

using namespace simpla;
using namespace simpla::geometry;
int main(int argc, char **argv)
{
	typedef typename coordinate_system::Cartesian<3> cs;

	model::Polygon<cs> poly;

	append(poly, model::Point<cs> { 1.0, 2.0, 0.0 });
	append(poly, model::Point<cs> { 6.0, 4.0, 0.0 });
	append(poly, model::Point<cs> { 5.0, 1.0, 0.0 });
	append(poly, model::Point<cs> { 1.0, 2.0, 0.0 });

	model::Point<cs> x0 = { 1.5, 1.5, 0.0 };

	std::cout << "Area: " << area(poly) << std::endl;

	std::cout << "Contains " << (x0) << std::endl;
	std::cout << "Within " << std::boolalpha << within(x0, poly) << std::endl;

	std::cout << poly << std::endl;

	model::LineSegment<cs> line( { 1.0, 2.0, 6.0, 4.0, 6.0, 4.0 });
	std::cout << "Line " << line << std::endl;
	model::Box<cs> box( { 1.0, 2.0, 6.0, 4.0, 6.0, 4.0 });
	std::cout << "Box " << box << std::endl;

//	std::list<model::Polygon<cs>> res;
//	intersection(box, poly, res);

//	for (auto const & item : res)
//	{
//		std::cout << "Res Polygon" << item << std::endl;
//	}
}

//
//template<typename T0, typename T1, typename T2, typename T3>
//std::tuple<Real, Real> line_intersection2d(T0 const& P0, T1 const & P1,
//		T2 const & Q0, T3 const & Q1, int ZAXIS = 2)
//{
//	Real s = 0.0;
//	Real t = 0.0;
//
//	Vec3 u = P1 - P0;
//	Vec3 v = Q1 - Q0;
//	Vec3 w0 = P0 - Q0;
//
//	u[ZAXIS] = 0;
//	v[ZAXIS] = 0;
//	w0[ZAXIS] = 0;
//	// @ref http://geomalgorithms.com/a07-_distance.html
//	Real a = inner_product(u, u);
//	Real b = inner_product(u, v);
//	Real c = inner_product(v, v);
//	Real d = inner_product(u, w0);
//	Real e = inner_product(v, w0);
//
//	if (std::abs(a * c - b * b) < EPSILON)
//	{
//		//two lines are parallel
//		s = std::numeric_limits<double>::max();
//		t = std::numeric_limits<double>::max();
//
//	}
//	else
//	{
//		s = (b * e - c * d) / (a * c - b * b);
//
//		t = (a * e - b * d) / (a * c - b * b);
//	}
//
//	return std::make_tuple(s, t);
//}
//
//Vec3 normal_vector_of_surface(std::vector<coordinate_tuple> const & polygons)
//{
//	bool on_same_plane = false;
//	Vec3 n;
//	n = 0;
//	auto first = polygons.begin();
//
//	while (first != polygons.end())
//	{
//		auto second = first;
//		++second;
//		if (second == polygons.end())
//		{
//			second = polygons.begin();
//		}
//		auto third = second;
//		++third;
//		if (third == polygons.end())
//		{
//			third = polygons.begin();
//		}
//
//		Vec3 n1;
//		n1 = cross(*second - *first, *third - *second);
//
//		if (inner_product(n1, n) < 0)
//		{
//			return std::move(Vec3( { 0, 0, 0 }));
//		}
//
//		Real nn = inner_product(n1, n1);
//		if (nn > EPSILON)
//		{
//			n = n1 / nn;
//		}
//
//		++first;
//
//	}
//	return std::move(n);
//
//}
//
//template<typename TV>
//void polyline_intersect_grid(std::vector<coordinate_tuple> const & polygons,
//		id_type shift, TV *volume, std::vector<coordinate_tuple> *new_path,
//		const int ZAXIS = 2)
//{
//	const int XAXIS = (ZAXIS + 1) % 3;
//	const int YAXIS = (ZAXIS + 2) % 3;
//	const Real dx = 1.0; // length of edge
//	const Real dA = 1.0; // area of cell
//
//	const coordinate_tuple n = normal_vector_of_surface(polygons);
//
//	if (inner_product(n, n) < EPSILON)
//	{
//		RUNTIME_ERROR("illegal polygon!");
//	}
//
//	static const id_type id_cell_shift[9] = {
//
//	0,
//
//	MeshIDs::ID_ZERO + (MeshIDs::_DI << 1),
//
//	MeshIDs::ID_ZERO + (MeshIDs::_DI << 1) + (MeshIDs::_DJ << 1),
//
//	MeshIDs::ID_ZERO + (MeshIDs::_DJ << 1),
//
//	MeshIDs::ID_ZERO - (MeshIDs::_DI << 1) + (MeshIDs::_DJ << 1),
//
//	MeshIDs::ID_ZERO - (MeshIDs::_DI << 1),
//
//	MeshIDs::ID_ZERO - (MeshIDs::_DI << 1) - (MeshIDs::_DJ << 1),
//
//	MeshIDs::ID_ZERO - (MeshIDs::_DJ << 1),
//
//	MeshIDs::ID_ZERO + (MeshIDs::_DI << 1) - (MeshIDs::_DJ << 1)
//
//	};
//
//	const int num_of_cell_vertics = 4;
//	coordinate_tuple q[num_of_cell_vertics];
//
//	coordinate_tuple x0, x1;
//
//	x0 = polygons[0UL];
//
//	x1 = *(++polygons.begin());
//
//	auto it = polygons.begin();
//
//	id_type current_cell = MeshIDs::coordinates_to_id(x0);
//	id_type current_cell_volume = 0;
//
//	int in_edge = 0;
//	while (true)
//	{
//
//		new_path->push_back(x0);
//
//		coordinate_tuple q0 = { std::floor(x0[0]), std::floor(x0[1]),
//				std::floor(x0[2]) };
//
//		current_cell = MeshIDs::coordinates_to_id(q0);
//
//		static const coordinate_tuple d[6][2] = {
//
//		-1, 0, 0, -1, 1, 0,
//
//		0, 0, 0, 0, 1, 0,
//
//		1, 0, 0, 1, 1, 0,
//
//		0, -1, 0, 1, -1, 0,
//
//		0, 0, 0, 1, 0, 0,
//
//		0, 1, 0, 1, 1, 0
//
//		};
//
//		Real t = std::numeric_limits<Real>::max();
//
//		int line_num = 0;
//		for (int i = 0; i < 6; ++i)
//		{
//			Real t_;
//
//			std::tie(std::ignore, t_) = line_intersection2d(q0 + d[i][0],
//					q0 + d[i][1], x0, x1);
//
//			if (t_ > 0 && t_ < t)
//			{
//				line_num = i;
//				t = t_;
//			}
//		}
//		if (t > 1)
//		{
//
//			current_cell_volume += cross(x1, x0);
//
//			x0 = x1;
//
//			if (it == polygons.end())
//			{
//				break;
//			}
//
//			++it;
//			if (it != polygons.end())
//			{
//				x1 = *it;
//			}
//			else
//			{
//				x1 = *polygons.begin();
//			}
//		}
//		else
//		{
//
//			(*volume)[MeshIDs::coordinates_to_id((x0 + x1) * 0.5)] += t
//					* cross(x0, x1);
//			x0 += t * (x1 - x0);
//		}
//	}
//}
//
//USE_CASE(model,"Cut Cell")
//{
//
//	typedef typename mesh_type::coordinate_tuple coordinate_tuple;
//
//	typedef typename mesh_type::id_type id_type;
//
//	auto geometry = std::make_shared<mesh_type>();
//
//	geometry->load(options["Mesh"]);
//
//	geometry->deploy();
//
//	std::vector<coordinate_tuple> p0, p1, p2, p3, p4, p5, p6, p7;
//
//	size_t node_id = 0;
//	options["Polylines"].as(&p0);
//	options["NodeId"].as(&node_id);
//
//	if (p0.empty())
//	{
//		return;
//	}
//
//	std::multimap<id_type, Real> b_cell;
//
//	polygen_cut_cell(*geometry, p0.begin(), p0.end(), &b_cell, node_id);
//
//	for (auto const & item : b_cell)
//	{
//		coordinate_tuple x0 = geometry->coordinates(item.first);
//
//		p1.push_back(x0);
//
//		p2.push_back(geometry->pull_back(x0, item.second));
//
//		//		coordinate_tuple q0, q1;
//		//
//		//		auto it = item.second;
//		//
//		//		q0 = *(it);
//		//		++it;
//		//		if (it == p0.end())
//		//		{
//		//			it = p0.begin();
//		//		}
//		//		q1 = *(it);
//		//
//		//		Vec3 normal;
//		//
//		//		normal = (x0 - q0)
//		//				- (q1 - q0)
//		//						* (inner_product(x0 - q0, q1 - q0)
//		//								/ inner_product(q1 - q0, q1 - q0));
//
//	}
//
//	p0.push_back(p0.front());
//
//	LOGGER << SAVE(p0) << std::endl;
//	LOGGER << SAVE(p1) << std::endl;
//	LOGGER << SAVE(p2) << std::endl;
//
//
//	find_boundary2D(p3, shift0, &p4);
//
//	coordinate_tuple shift1 = { 0.5, 0.5, 0.5 };
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
//}
//
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
//int boundary_tag(coordinate_tuple const & x)
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
//void divid_to_boundary(std::vector<coordinate_tuple> const& points,
//		std::vector<boundary_s> * res, int ZAXIS = 2)
//{
//
//	auto i0 = points.rbegin();
//	auto i1 = points.begin();
//	auto i2 = ++points.begin();
//
//	coordinate_tuple x0, x1, x2;
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
//		coordinate_tuple xp, xm;
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
////		coordinate_tuple xp = x0;
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
//void get_intersctions(TPoints const & points, coordinate_tuple const & shift,
//		std::vector<coordinate_tuple> *res)
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
//				[&](coordinate_tuple const & xa,coordinate_tuple const & xb)->bool
//				{
//					return inner_product(xb-xa,x1-x0)>0;
//				});
//
//	}
//}
//typedef coordinate_tuple Vec3;
//
//template<typename TPoints>
//void find_boundary2D(TPoints const & points, coordinate_tuple const & shift,
//		std::vector<coordinate_tuple> *res, int ZAXIS = 2)
//{
//
//	auto i0 = points.begin();
//
//	while (i0 != points.end())
//	{
//		coordinate_tuple p0 = *i0;
//		auto i1 = ++i0;
//
//		if (i1 == points.end())
//		{
//			i1 = points.begin();
//		}
//		coordinate_tuple p1 = *i1;
//
//		coordinate_tuple p2;
//		p2 = p1;
//
//		p2[ZAXIS] += 1.0;
//
//		coordinate_tuple x0 = //
//				{ //
//				std::floor(std::min(p0[0], p1[0]) - shift[0]) + shift[0], //
//				std::floor(std::min(p0[1], p1[1]) - shift[1]) + shift[1], //
//				std::floor(std::min(p0[2], p1[2]) - shift[2]) + shift[2]     //
//				};
//		x0 += 0.5;
////		id_type s = MeshDummy::coordinates_to_id(x0);
//
//		res->push_back(x0);
//		coordinate_tuple n;
//		std::tie(std::ignore, n) = distance_point_to_plane(x0, p0, p1, p2);
//		res->push_back(n);
//
//	}
//
//}
