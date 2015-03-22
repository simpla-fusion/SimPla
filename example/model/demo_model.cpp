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

template<typename T0, typename T1, typename T2, typename T3>
std::tuple<Real, Real> line_intersection2d(T0 const& P0, T1 const & P1,
		T2 const & Q0, T3 const & Q1, int ZAXIS = 2)
{
	Real s = 0.0;
	Real t = 0.0;

	Vec3 u = P1 - P0;
	Vec3 v = Q1 - Q0;
	Vec3 w0 = P0 - Q0;

	u[ZAXIS] = 0;
	v[ZAXIS] = 0;
	w0[ZAXIS] = 0;
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

std::tuple<size_t, Real, Real> line_intersect_polygon(coordinates_type x0,
		coordinates_type x1, int num_of_vertex, coordinates_type const * q,
		size_t ZAXIS = 2)
{
	int edge_id = -1;

	Real s = -1;
	Real t = std::numeric_limits<Real>::max();

	for (int i = 0; i < num_of_vertex; ++i)
	{
		Real s_, t_;
		std::tie(s_, t_) = line_intersection2d(q[i], q[(i + 1) % num_of_vertex],
				x0, x1, ZAXIS);

		if (t_ > 0 && t_ < t && s_ >= 0 && s_ < 1)
		{
			t = t_;
			s = s_;
			edge_id = i;
		}
	}

	return std::make_tuple(edge_id, s, t);
}

Vec3 normal_vector_of_surface(std::vector<coordinates_type> const & polygons)
{
	bool on_same_plane = false;
	Vec3 n;
	n = 0;
	auto first = polygons.begin();

	while (first != polygons.end())
	{
		auto second = first;
		++second;
		if (second == polygons.end())
		{
			second = polygons.begin();
		}
		auto third = second;
		++third;
		if (third == polygons.end())
		{
			third = polygons.begin();
		}

		Vec3 n1;
		n1 = cross(*second - *first, *third - *second);

		if (inner_product(n1, n) < 0)
		{
			return std::move(Vec3( { 0, 0, 0 }));
		}

		Real nn = inner_product(n1, n1);
		if (nn > EPSILON)
		{
			n = n1 / nn;
		}

		++first;

	}
	return std::move(n);

}
template<typename TV>
void polyline_intersect_grid(std::vector<coordinates_type> const & polygons,
		id_type shift, TV *volume, std::vector<coordinates_type> *new_path,
		const int ZAXIS = 2)
{
	const int XAXIS = (ZAXIS + 1) % 3;
	const int YAXIS = (ZAXIS + 2) % 3;
	const Real dx = 1.0; // length of edge
	const Real dA = 1.0; // area of cell

	const coordinates_type n = normal_vector_of_surface(polygons);

	if (inner_product(n, n) < EPSILON)
	{
		RUNTIME_ERROR("illegal polygon!");
	}

	//	static const id_type id_face_shift[3] = {
	//
	//	(MeshIDs::_DJ | MeshIDs::_DK),
	//
	//	(MeshIDs::_DK | MeshIDs::_DI),
	//
	//	(MeshIDs::_DJ | MeshIDs::_DI)
	//
	//	};
	//
	//	static const id_type id_edge_shift[4] = {
	//
	//	MeshIDs::_DI,
	//
	//	(MeshIDs::_DJ) | (MeshIDs::_DI << 1),
	//
	//	(MeshIDs::_DI) | (MeshIDs::_DJ << 1),
	//
	//	MeshIDs::_DJ
	//
	//	};
	static const id_type id_face = (MeshIDs::_DI | MeshIDs::_DJ);

	static const id_type id_cell_shift[9] = {

	0,

	MeshIDs::ID_ZERO + (MeshIDs::_DI << 1),

	MeshIDs::ID_ZERO + (MeshIDs::_DI << 1) + (MeshIDs::_DJ << 1),

	MeshIDs::ID_ZERO + (MeshIDs::_DJ << 1),

	MeshIDs::ID_ZERO - (MeshIDs::_DI << 1) + (MeshIDs::_DJ << 1),

	MeshIDs::ID_ZERO - (MeshIDs::_DI << 1),

	MeshIDs::ID_ZERO - (MeshIDs::_DI << 1) - (MeshIDs::_DJ << 1),

	MeshIDs::ID_ZERO - (MeshIDs::_DJ << 1),

	MeshIDs::ID_ZERO + (MeshIDs::_DI << 1) - (MeshIDs::_DJ << 1)

	};

	/**
	 *
	 *             0
	 *             ^
	 *       q3----|-----q2
	 *        |    2     |
	 *        |          |
	 *      1<-3        1->3
	 *        |    0     |
	 *        q0---|----q1
	 *             v
	 *             2
	 */

	static const id_type convert_out_in_edge_out[4] = { 2, 3, 0, 1 };

	const int num_of_cell_vertics = 4;
	coordinates_type q[num_of_cell_vertics];

	coordinates_type x0, x1;

	x0 = polygons[0UL];

	x1 = *(++polygons.begin());

	auto it = polygons.begin();

	while (true)
	{
		new_path->push_back(x0);
//
//		id_type cell0 = (MeshIDs::coordinates_to_id(
//				x0 - MeshIDs::id_to_coordinates(shift | MeshIDs::ID_ZERO))
//				& (MeshIDs::CELL_ID_MASK)) | shift;
//
//

//			MeshIDs::get_vertics<FACE>(ZAXIS, cell_hint, q);

		coordinates_type q0 = { std::floor(x0[0]), std::floor(x0[1]),
				std::floor(x0[2]) };

		static const coordinates_type d[6][2] = {

		-1, 0, 0, -1, 1, 0,

		0, 0, 0, 0, 1, 0,

		1, 0, 0, 1, 1, 0,

		0, -1, 0, 1, -1, 0,

		0, 0, 0, 1, 0, 0,

		0, 1, 0, 1, 1, 0

		};

		Real t = std::numeric_limits<Real>::max();
		for (int i = 0; i < 6; ++i)
		{
			Real t_;

			std::tie(std::ignore, t_) = line_intersection2d(q0 + d[i][0],
					q0 + d[i][1], x0, x1);

			if (t_ > 0 && t_ < t)
			{
				t = t_;
			}
		}
		if (t > 1)
		{
			x0 = x1;

			if (it == polygons.end())
			{
				break;
			}

			++it;
			if (it != polygons.end())
			{
				x1 = *it;
			}
			else
			{
				x1 = *polygons.begin();
			}
		}
		else
		{
			x0 += t * (x1 - x0);
		}
	}
}
//		Real face_area = 0;
//
//		size_t i = (it_prev + 1 + num_of_vertics) % num_of_vertics;
//		size_t j = (it_prev + 2 + num_of_vertics) % num_of_vertics;
//
//		for (size_t i = it_prev; i != it_next;
//				i = (i + 1 + num_of_vertics) % num_of_vertics)
//		{
//			size_t j = i + 1
//			face_area += -inner_product(n,
//					cross(poly_nodes[i] - x_in, x_out - poly_nodes[j])) * 0.5;
//
//		}
//
//		Real t_out = -1;
//
//		std::tie(edge_out, s_out, t_out) = line_intersect_polygon(x[0], x[1],
//				num_of_cell_vertics, q, ZAXIS);
//
//		VERBOSE << (x[0]) << " " << (x[1]) << " " << (edge_out) << " "
//				<< (s_out) << " " << (t_out) << std::endl;
//		if (t_out >= 1 || s_out < 0 || s_out > 1)
//		{
//			if (it == polygons.end())
//			{
//				break;
//			}
//
//			// move to next vertex of polygon
//
//			x[-1] = x[0];
//			x[0] = x[1];
//
//			++it;
//
//			if (it != polygons.end())
//			{
//				x[1] = *it;
//			}
//			else
//			{
//				x[1] = polygons.front();
//			}
//
//			continue;
//		}
//
//		x_out = q[edge_out]
//				+ (q[(edge_out + 1) % num_of_cell_vertex] - q[edge_out])
//						* s_out;
//
//// choice next cell
//
//// snap to grid point
//		if (std::abs(s_out) < EPSILON || std::abs(s_out - 1) < EPSILON)
//		{
//			s_out = std::floor(s_out + 0.5);
//
//			x[0] = q[edge_out]
//					+ (q[(edge_out + 1) % num_of_cell_vertex] - q[edge_out])
//							* s_out;
//
//			this_cell = MeshIDs::coordinates_to_id(x[0])
//					+ ((x[1][0] > x[0][0]) ? MeshIDs::_DI : -MeshIDs::_DI)
//					+ ((x[1][1] > x[0][1]) ? MeshIDs::_DJ : -MeshIDs::_DJ);
//
//			s_in = -1;
//
//			edge_in = -1;
//
//			x_in = x[0];
//
//			continue;
//		}
//		else
//		{
//			// move to next cell
//			x[0] = x_out;
//
//			if (edge_out == edge_in)
//			{
//			}
//			else if (edge_out == (edge_in + 1) % num_of_cell_vertex)
//			{
//				face_area += (1 - (1 - s_in) * s_out * 0.5) * dA;
//			}
//			else if (edge_out == (edge_in + 2) % num_of_cell_vertex)
//			{
//				face_area += (s_in + 1 - s_out) * 0.5 * 1 * dA;
//			}
//			else if (edge_out == (edge_in + 3) % num_of_cell_vertex)
//			{
//				face_area += s_in * (1 - s_out) * 0.5 * dA;
//			}
//
//			face_area += -inner_product(n, cross(x[0] - x_in, x_out - x[0]))
//					* 0.5;
//
//			// save edge/face/volume volume
//			(*volume)[this_cell + id_edge_shift[edge_out]] = (1 - s_out) * dx;
//			(*volume)[this_cell + id_face_shift[ZAXIS]] = face_area;
//			(*volume)[this_cell + (MeshIDs::_DA)] = face_area * dx;
//
//			this_cell += id_cell_shift[edge_out];
////		CHECK_BIT(id_cell_shift[edge_out]);
//
//			edge_in = convert_out_in_edge_out[edge_out];
//
//			s_in = 1 - s_out;
//
//			x_in = x_out;
//		}

SP_APP(model)
{

	typedef typename MeshIDs::coordinates_type coordinates_type;

	typedef typename MeshIDs::id_type id_type;

	std::vector<coordinates_type> p0, p1, p2, p3, p4, p5, p6, p7;

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
		size_t d[3] = {

		(s) & (MeshIDs::_DI),

		(s) & (MeshIDs::_DJ),

		(s) & (MeshIDs::_DK) };

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

	nTuple<size_t, 3> v_dims;

	v_dims = dims * 2;

	auto volume0 = MeshIDs::make_volume_container(v_dims);

	auto volume1 = MeshIDs::make_volume_container(v_dims);

	volume0.fill(0);

	for (auto s : volume_in_side)
	{
		volume0[s] = 1;
	}

	polyline_intersect_grid(p0, MeshIDs::_DA, &volume0, &p4);

//	polyline_intersect_grid(p0, ((MeshIDs::_DA)), &volume1, &p5);

//	LOGGER << save("volume0", volume0.data().get(), 3, &v_dims[0]) << std::endl;
//	LOGGER << save("volume1", volume1.data().get(), 3, &v_dims[0]) << std::endl;
//
	p0.push_back(p0.front());

	LOGGER << SAVE(p0) << std::endl;
	LOGGER << SAVE(p1) << std::endl;
	LOGGER << SAVE(p2) << std::endl;
	LOGGER << SAVE(p3) << std::endl;
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
