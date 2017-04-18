/**
 * @file cut_cell.h
 *
 * @date 2015-5-12
 * @author salmon
 */

#ifndef CORE_GEOMETRY_CUT_CELL_H_
#define CORE_GEOMETRY_CUT_CELL_H_

#include <stddef.h>
#include <list>
#include <map>
#include "utilities.h"
#include "simpla/sp_def.h"

namespace simpla
{

template<typename TM, typename PolygonIterator>
void pixel_intersects_polygon(TM const & mesh, int node_id,
		PolygonIterator const & begin, PolygonIterator const & end,
		std::multimap<typename TM::id_type, PolygonIterator> *res)
{
	for (auto it = begin; it != end; ++it)
	{

		for (auto s : mesh.range(return_envelope(*it)))
		{

			if (intersects(mesh.pixel(s), *it))
			{
				res->insert(std::make_pair(s, it));
			}
		}
	}

}
template<typename DistFunction>
size_t intersection(Vec3 const & min, Vec3 const & max,
		DistFunction const & dist_fun)
{

}
template<size_t LEVEL, typename TMesh, typename DistFunction, typename TRes>
size_t divide_box(TMesh const & mesh, DistFunction const & dist_fun, TRes )
{
	typedef typename TMesh mesh_type;
	typedef typename mesh_type::id_type id_type;

	id_type s = std::get<0>(mesh.coordinate_global_to_local((max + min) * 0.5));

	auto out_code = mesh.out_code(s, dist_fun);

	static const size_t ALL_OUT = 0xAAAA;

	static const size_t ALL_IN = 0x5555;
	if (out_code == ALL_OUT || out_code == ALL_IN) // all out or  all in
	{
	}
	else
	{

	}

}

///**
// *   for cut-cell
// * @_fdtd_param s0
// * @_fdtd_param s1
// * @return
// */
//static constexpr mesh_id_type out_code(mesh_id_type c, mesh_id_type s)
//{
//	return out_code_(((c | FULL_OVERFLOW_FLAG) - s) & PRIMARY_ID_MASK);
//}
//static constexpr mesh_id_type out_code_(mesh_id_type c)
//{
//	return ((c >> (ID_DIGITS - 1)) & 1UL) | ((c >> (ID_DIGITS - 3)) & (4UL))
//			| ((c >> (ID_DIGITS * 2 - 5)) & 16UL)
//			| (static_cast<mesh_id_type>((c & (OVERFLOW_FLAG - 1UL)) != 0UL))
//			| (static_cast<mesh_id_type>((c & ((OVERFLOW_FLAG - 1UL) << ID_DIGITS))
//					!= 0UL) << (2UL))
//			| (static_cast<mesh_id_type>((c
//					& ((OVERFLOW_FLAG - 1UL) << (ID_DIGITS * 2))) != 0UL)
//					<< (4UL));
//}
//
///**
// *
// *
// *
// *  line intersection face
// * @_fdtd_param x0
// * @_fdtd_param x1
// * @_fdtd_param res
// * @_fdtd_param node_id id of cell
// */
//template<typename TRes>
//static void cut_cell(coordinates_type const & x0, coordinates_type const & x1,
//		TRes *res, mesh_id_type node_id = 7);
//
///**
// *
// */
///**
// * triangle intersection edge
// * @_fdtd_param x0
// * @_fdtd_param x1
// * @_fdtd_param x2
// * @_fdtd_param res
// * @_fdtd_param node_id id of cell
// */
//template<typename TX, typename TRes>
//static void cut_cell(TX const & x0, TX const & x1, TX const & x2, TRes*res,
//		mesh_id_type node_id = 7);
//
//template<size_t N, size_t M>
//template<typename TRes>
//void EntityIdCoder<N, M>::cut_cell(coordinates_type const & pV0,
//		coordinates_type const & pV1, TRes*res, mesh_id_type node_id)
//{
//
//	coordinates_type V0 = pV0 - m_id_to_coordinates_shift_[node_id];
//
//	coordinates_type V1 = pV1 - m_id_to_coordinates_shift_[node_id];
//
//	Vec3 u = V1 - V0;
//
//	Real vmin = {  //
//			min(V0[0], V1[0]), //
//			min(V0[1], V1[1]), //
//			min(V0[2], V1[2]) };
//	Real vmax = {  //
//			max(V0[0], V1[0]), //
//			max(V0[1], V1[1]), //
//			max(V0[2], V1[2]) };
//
//	mesh_id_type face_id[3] = { 6, 5, 3 };
//
//	for (int zaxe = 0; zaxe < 3; ++zaxe)
//	{
//		int xaxe = (zaxe + 1) % 3;
//		int yaxe = (zaxe + 2) % 3;
//
//		if ((vmax[xaxe] - vmin[xaxe]) < EPSILON)
//		{
//			continue;
//		}
//
//		Real xb = std::floor(vmin[xaxe] / (_R * 2)) * (_R * 2);
//		Real xe = std::floor(vmax[xaxe] / (_R * 2) + 1.0) * (_R * 2);
//
//		for (Real x = xb; x < xe; x += (_R * 2))
//		{
//
//			Real t = (x - V0[zaxe]) / u[zaxe];
//
//			if (t < 0 || t > 1)
//			{
//				continue;
//			}
//			coordinates_type y;
//
//			y = V0 + t * u;
//
//			mesh_id_type s = (pack(y) & PRIMARY_ID_MASK)
//					| m_id_to_shift_[face_id[zaxe]];
//
//			res->SetValue(std::make_pair(s + m_id_to_shift_[node_id], t));
//		}
//
//	};
//}
//
//template<typename TMesh,typename TGeo, typename TRes>
//void cut_cell(TGeo const & geo, TRes*res  )
//{
//
//	coordinates_type V0 = pV0 - m_id_to_coordinates_shift_[node_id];
//	coordinates_type V1 = pV1 - m_id_to_coordinates_shift_[node_id];
//	coordinates_type V2 = pV2 - m_id_to_coordinates_shift_[node_id];
//
//	mesh_id_type edge_id[3] = { 1, 2, 4 };
//
//	Vec3 u = V1 - V0;
//	Vec3 v = V2 - V0;
//
//	Vec3 n = cross(u, v);
//
//	Real vv = inner_product(v, v);
//	Real uu = inner_product(u, u);
//	Real uv = inner_product(u, v);
//	Real uvuv = uv * uv - uu * vv;
//
//	Real vmin = {  //
//			min(V0[0], V1[0], V2[0]), //
//			min(V0[1], V1[1], V2[1]), //
//			min(V0[2], V1[2], V2[2]) };
//	Real vmax = {  //
//			max(V0[0], V1[0], V2[0]), //
//			max(V0[1], V1[1], V2[1]), //
//			max(V0[2], V1[2], V2[2]) };
//
//	for (int zaxe = 0; zaxe < 3; ++zaxe)
//	{
//		int xaxe = (zaxe + 1) % 3;
//		int yaxe = (zaxe + 2) % 3;
//
//		if ((vmax[xaxe] - vmin[xaxe]) < EPSILON
//				|| (vmax[yaxe] - vmin[yaxe]) < EPSILON)
//		{
//			continue;
//		}
//
//		Real xb = std::floor(vmin[xaxe] / (_R * 2)) * (_R * 2);
//		Real xe = std::floor(vmax[xaxe] / (_R * 2) + 1.0) * (_R * 2);
//
//		Real yb = std::floor(vmin[yaxe] / (_R * 2)) * (_R * 2);
//		Real ye = std::floor(vmax[yaxe] / (_R * 2) + 1.0) * (_R * 2);
//
//		for (Real x = xb; x < xe; x += (_R * 2))
//			for (Real y = yb; y < ye; y += (_R * 2))
//			{
//				///     Theorem:http://geomalgorithms.com/a06-_intersect-2.html
//				///				coordinates_type P0, P1;
//				///				P0[xaxe] = x;
//				///				P0[yaxe] = y;
//				///				P0[zaxe] = 0;
//				///				P1[xaxe] = x;
//				///				P1[yaxe] = y;
//				///				P1[zaxe] = 1;
//				///
//				///				Real r = inner_product(n, V0 - P0) / inner_product(n, P1 - P0);
//				///
//				///				Vec3 w = P0 - V0 +  (P1 - P0)*inner_product(n, V0 - P0) / inner_product(n, P1 - P0);
//
//				Vec3 w;
//
//				w[xaxe] = x - V0[xaxe];
//				w[yaxe] = y - V0[yaxe];
//				w[zaxe] = 0 - V0[zaxe]
//						- (n[xaxe] * (x - V0[xaxe]) + n[yaxe] * (y - V0[yaxe])
//								+ n[yaxe] * (0 - V0[zaxe])) / n[zaxe];
//
//				Real s = (inner_product(w, v) * uv - inner_product(w, u) * vv)
//						/ uvuv;
//
//				Real t = (inner_product(w, u) * uv - inner_product(w, v) * uu)
//						/ uvuv;
//
//				if (s < 0 || t < 0 || s + t > 1)
//				{
//					continue;
//				}
//
//				mesh_id_type p = ((pack(V0 + s * u + t * v) & PRIMARY_ID_MASK)
//						| m_id_to_shift_[edge_id[zaxe]]);
//
//				res->SetValue(
//						std::make_pair(p + m_id_to_shift_[node_id],
//								std::make_tuple(u, v)));
//
//			}
//
//	};
//template<typename TM, typename TX>
//int line_segment_cut_cell(TM const & geometry, typename TM::mesh_id_type node_id,
//		TX const &x0, TX const & x1, typename TM::mesh_id_type s0,
//		typename TM::mesh_id_type s1, std::SetValue<typename TM::mesh_id_type>* res,
//		Real epsilon = 0.01)
//{
//	int size = 0;
//	if ((geometry.diff_index(s0, s1) != 0)
//			&& ((inner_product(x1 - x0, x1 - x0))
//					> epsilon * inner_product(geometry.dx(), geometry.dx())))
//	{
//		typename TM::coordinate_tuple xc;
//
//		xc = (x1 + x0) * 0.5;
//
//		auto sc = std::Get<0>(geometry.point_global_to_local(xc, node_id));
//
//		res->SetValue(sc);
//
//		++size;
//
//		size += line_segment_cut_cell(geometry, node_id, x0, xc, s0, sc, res,
//				epsilon);
//
//		size += line_segment_cut_cell(geometry, node_id, xc, x1, sc, s1, res,
//				epsilon);
//
//	}
//
//	return size;
//}
//template<typename TM, typename TX>
//void triangle_cut_cell(TM const & geometry, typename TM::mesh_id_type node_id,
//		TX const &x0, TX const & x1, TX const & x2,
//		std::SetValue<typename TM::mesh_id_type>* res)
//{
//	typedef TM manifold_type;
//	typedef typename manifold_type::coordinate_tuple coordinate_tuple;
//	typedef typename manifold_type::mesh_id_type mesh_id_type;
//
//	coordinate_tuple dims0;
//	dims0 = (x1 - x0) / geometry.dx();
//	coordinate_tuple dims1;
//	dims1 = (x2 - x0) / geometry.dx();
//
//	auto n0 = std::max(std::max(dims0[0], dims0[1]), dims0[2]);
//	auto n1 = std::max(std::max(dims1[0], dims1[1]), dims1[2]);
//	coordinate_tuple dx0 = (x1 - x0) / n0;
//	coordinate_tuple dx1 = (x2 - x0) / n1;
//
//	for (size_t i = 0; i <= n0; ++i)
//		for (size_t j = 0; j <= n1; ++j)
//		{
//			res->SetValue(
//					std::Get<0>(
//							geometry.point_global_to_local(
//									x0 + dx0 * i + dx1 * j, node_id)));
//		}
//
//}

//template<typename TM, typename TSplicesIter>
//size_t cut_cell(TM const & geometry, TSplicesIter const & ib,
//		TSplicesIter const & ie,
//		std::map<typename TM::mesh_id_type,
//				typename std::iterator_traits<TSplicesIter>::value_type>* res)
//{
//
//	for (auto it = ib; it != ie; ++it)
//	{
//		line_segment_cut_cel(it, res);
//	}
//}
//
//template<typename T0, typename T1, typename T2, typename T3, typename T4>
//bool intersection_cell(T0 const & min, T1 const & max, T2 const & x0,
//		T3 const & x1, T4 const & x3)
//{
//	return true;
//}
//
//bool intersection_cell(std::int64_t const & min, std::int64_t const & max,
//		std::int64_t const & x0, std::int64_t const & x1,
//		std::int64_t const & x3)
//{
//	return false;
//}
//
//template<typename T>
//bool check_outcode(T const & c0, T const & c1, T const & c2)
//{
//
//}
//template<typename TM, typename TI>
//size_t triangle_cut_cell(TM const & geometry, typename TM::mesh_id_type node_id,
//		TI const &i0, std::multimap<typename TM::mesh_id_type, TI>* res)
//{
//	typedef TM manifold_type;
//	typedef typename manifold_type::coordinate_tuple coordinate_tuple;
//	typedef typename manifold_type::mesh_id_type mesh_id_type;
//
//	TI it = i0;
//	coordinate_tuple x0 = *it;
//	++it;
//	coordinate_tuple x1 = *it;
//	++it;
//	coordinate_tuple x2 = *it;
//
//	coordinate_tuple xmin, xmax;
//
//	mesh_id_type s0 = std::Get<0>(geometry.point_global_to_local(x0, node_id));
//
//	mesh_id_type s1 = std::Get<0>(geometry.point_global_to_local(x1, node_id));
//
//	mesh_id_type s2 = std::Get<0>(geometry.point_global_to_local(x2, node_id));
//
//	for (auto s : geometry.box(bound(bound(x0, x1), x2), node_id))
//	{
//		mesh_id_type out_code[3] = { geometry.out_code(s, s0), geometry.out_code(s, s1),
//				geometry.out_code(s, s2) };
//		bool success = false;
//		if ((out_code[0] & out_code[1] & out_code[2]) != 0)
//		{ // all vertices are outside,  can be trivially rejected
//			continue;
//		}
//		else if ((out_code[0] == 0) | (out_code[1] == 0) | (out_code[2] == 0))
//		{ // at least one vertex is inside,  can be trivially accepted
//
//			success = true;
//		}
//		else
//		{
//			mesh_id_type code = code[0] | code[1] | code[2];
//
//			for (int i = 0; i < 6; ++i)
//			{
//
//				if ((code >> i) & 1UL != 0)
//				{
//					int IZ = i >> 1UL;
//					int IX = (IZ + 1) % 3;
//					int IY = (IZ + 2) % 3;
//
//					mesh_id_type face_id = s
//							+ (manifold_type::_DA << (manifold_type::ID_DIGITS * IZ))
//									* ((i % 2 == 0) ? 1 : -1);
//
//					coordinate_tuple xa = geometry.coordinates(face_id
//
//					- (manifold_type::_DA << (manifold_type::ID_DIGITS * IX))
//
//					- (manifold_type::_DA << (manifold_type::ID_DIGITS * IY)));
//
//				}
//			}
//			if (success)
//				res->SetValue(std::make_pair(s, i0));
//		}
//
//	}
//
//}
//template<typename TM, typename TI>
//size_t line_segment_cut_cell(TM const & geometry, typename TM::mesh_id_type node_id,
//		TI const &i0, std::multimap<typename TM::mesh_id_type, TI>* res)
//{
//	typedef TM manifold_type;
//	typedef typename manifold_type::coordinate_tuple coordinate_tuple;
//	typedef typename manifold_type::mesh_id_type mesh_id_type;
//
//	TI it = i0;
//	coordinate_tuple x0 = *it;
//	++it;
//	coordinate_tuple x1 = *it;
//
//	mesh_id_type s0 = std::Get<0>(geometry.point_global_to_local(x0, node_id));
//
//	mesh_id_type s1 = std::Get<0>(geometry.point_global_to_local(x1, node_id));
//
//	for (auto s : geometry.box(bound(x0, x1), node_id))
//	{
//		mesh_id_type code0 = geometry.out_code(s, s0);
//		mesh_id_type code1 = geometry.out_code(s, s1);
//
//		if ((code0 & code1) != 0)
//		{
//			continue;
//		}
//		else if ((code0 | code1) == 0)
//		{
//			res->SetValue(std::make_pair(s, i0));
//		}
//		else
//		{
//			intersection(geometry.coordinates(s - manifold_type::_DA),
//					geometry.coordinates(s + manifold_type::_DA), x0, x1);
//			res->SetValue(std::make_pair(s, i0));
//		}
//	}
//
//}
//template<typename TM, typename TX>
//void line_segment_cut_cell(TM const & geometry, typename TM::mesh_id_type node_id,
//		TX const &x0, TX const & x1, std::SetValue<typename TM::mesh_id_type>* res)
//{
//	typedef TM manifold_type;
//	typedef typename manifold_type::coordinate_tuple coordinate_tuple;
//	typedef typename manifold_type::mesh_id_type mesh_id_type;
//
//	int m = 0;
//	for (int i = 0; i < 3; ++i)
//	{
//		if (geometry.dx()[i] > EPSILON)
//		{
//			m = std::max(m,
//					static_cast<int>(std::abs((x1[i] - x0[i]) / geometry.dx()[i])));
//		}
//	}
//
//	if (m <= 0)
//	{
//		return;
//	}
//	else
//	{
//
//		for (size_t i = 0; i <= m; ++i)
//		{
//
//			res->SetValue(std::Get<0>(geometry.point_global_to_local(
//
//			x0 + (x1 - x0) * (static_cast<Real>(i) / static_cast<Real>(m))
//
//			, node_id
//
//			)));
//		}
//	}
//}
}
// namespace simpla

#endif /* CORE_GEOMETRY_CUT_CELL_H_ */

