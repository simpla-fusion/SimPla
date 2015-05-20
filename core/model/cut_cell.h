/**
 * @file cut_cell.h
 *
 * @date 2015年5月12日
 * @author salmon
 */

#ifndef CORE_MODEL_CUT_CELL_H_
#define CORE_MODEL_CUT_CELL_H_

#include <stddef.h>
#include <list>
#include <map>
#include "../utilities/utilities.h"
namespace simpla
{

/**
 *
 *  TI **ib== TM::coordinates
 *
 * @param polygon
 * @param res
 * @return
 */


template<typename TM, typename TX>
int line_segment_cut_cell(TM const & mesh, TX const &y0, TX const & y1,
		std::set<typename TM::id_type>* res, typename TM::id_type node_id)
{
	typedef TM mesh_type;
	typedef typename mesh_type::topology_type topology_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::id_type id_type;
	coordinates_type x0 = mesh.inv_map(y0);
	coordinates_type x1 = mesh.inv_map(y1);
	topology_type::cut_cell(x0, x1, res, node_id);
	return res->size();
}


template<typename TM, typename T0, typename T1>
int polygen_cut_cell(TM const & mesh, T0 const &b, T1 const & e,
		std::multimap<typename TM::id_type, T0>* res,
		typename TM::id_type node_id = 0)
{
	typedef TM mesh_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::id_type id_type;

	int count = 0;

	for (auto i0 = b; i0 != e; ++i0)
	{
		auto i1 = i0;
		++i1;
		if (i1 == e)
		{
			i1 = b;
		}
		std::set<id_type> ids;
		count += line_segment_cut_cell(mesh, *i0, *i1, &ids, node_id);
		for (auto id : ids)
		{
			res->insert(std::make_pair(id, i0));
		}

	}
	return count;
}

//template<typename TM, typename TX>
//int line_segment_cut_cell(TM const & mesh, typename TM::id_type node_id,
//		TX const &x0, TX const & x1, typename TM::id_type s0,
//		typename TM::id_type s1, std::set<typename TM::id_type>* res,
//		Real epsilon = 0.01)
//{
//	int count = 0;
//	if ((mesh.diff_index(s0, s1) != 0)
//			&& ((inner_product(x1 - x0, x1 - x0))
//					> epsilon * inner_product(mesh.dx(), mesh.dx())))
//	{
//		typename TM::coordinates_type xc;
//
//		xc = (x1 + x0) * 0.5;
//
//		auto sc = std::get<0>(mesh.coordinates_global_to_local(xc, node_id));
//
//		res->insert(sc);
//
//		++count;
//
//		count += line_segment_cut_cell(mesh, node_id, x0, xc, s0, sc, res,
//				epsilon);
//
//		count += line_segment_cut_cell(mesh, node_id, xc, x1, sc, s1, res,
//				epsilon);
//
//	}
//
//	return count;
//}
//template<typename TM, typename TX>
//void triangle_cut_cell(TM const & mesh, typename TM::id_type node_id,
//		TX const &x0, TX const & x1, TX const & x2,
//		std::set<typename TM::id_type>* res)
//{
//	typedef TM mesh_type;
//	typedef typename mesh_type::coordinates_type coordinates_type;
//	typedef typename mesh_type::id_type id_type;
//
//	coordinates_type dims0;
//	dims0 = (x1 - x0) / mesh.dx();
//	coordinates_type dims1;
//	dims1 = (x2 - x0) / mesh.dx();
//
//	auto n0 = std::max(std::max(dims0[0], dims0[1]), dims0[2]);
//	auto n1 = std::max(std::max(dims1[0], dims1[1]), dims1[2]);
//	coordinates_type dx0 = (x1 - x0) / n0;
//	coordinates_type dx1 = (x2 - x0) / n1;
//
//	for (size_t i = 0; i <= n0; ++i)
//		for (size_t j = 0; j <= n1; ++j)
//		{
//			res->insert(
//					std::get<0>(
//							mesh.coordinates_global_to_local(
//									x0 + dx0 * i + dx1 * j, node_id)));
//		}
//
//}

//template<typename TM, typename TSplicesIter>
//size_t cut_cell(TM const & mesh, TSplicesIter const & ib,
//		TSplicesIter const & ie,
//		std::map<typename TM::id_type,
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
//size_t triangle_cut_cell(TM const & mesh, typename TM::id_type node_id,
//		TI const &i0, std::multimap<typename TM::id_type, TI>* res)
//{
//	typedef TM mesh_type;
//	typedef typename mesh_type::coordinates_type coordinates_type;
//	typedef typename mesh_type::id_type id_type;
//
//	TI it = i0;
//	coordinates_type x0 = *it;
//	++it;
//	coordinates_type x1 = *it;
//	++it;
//	coordinates_type x2 = *it;
//
//	coordinates_type xmin, xmax;
//
//	id_type s0 = std::get<0>(mesh.coordinates_global_to_local(x0, node_id));
//
//	id_type s1 = std::get<0>(mesh.coordinates_global_to_local(x1, node_id));
//
//	id_type s2 = std::get<0>(mesh.coordinates_global_to_local(x2, node_id));
//
//	for (auto s : mesh.box(bound(bound(x0, x1), x2), node_id))
//	{
//		id_type out_code[3] = { mesh.out_code(s, s0), mesh.out_code(s, s1),
//				mesh.out_code(s, s2) };
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
//			id_type code = code[0] | code[1] | code[2];
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
//					id_type face_id = s
//							+ (mesh_type::_DA << (mesh_type::ID_DIGITS * IZ))
//									* ((i % 2 == 0) ? 1 : -1);
//
//					coordinates_type xa = mesh.coordinates(face_id
//
//					- (mesh_type::_DA << (mesh_type::ID_DIGITS * IX))
//
//					- (mesh_type::_DA << (mesh_type::ID_DIGITS * IY)));
//
//				}
//			}
//			if (success)
//				res->insert(std::make_pair(s, i0));
//		}
//
//	}
//
//}
//template<typename TM, typename TI>
//size_t line_segment_cut_cell(TM const & mesh, typename TM::id_type node_id,
//		TI const &i0, std::multimap<typename TM::id_type, TI>* res)
//{
//	typedef TM mesh_type;
//	typedef typename mesh_type::coordinates_type coordinates_type;
//	typedef typename mesh_type::id_type id_type;
//
//	TI it = i0;
//	coordinates_type x0 = *it;
//	++it;
//	coordinates_type x1 = *it;
//
//	id_type s0 = std::get<0>(mesh.coordinates_global_to_local(x0, node_id));
//
//	id_type s1 = std::get<0>(mesh.coordinates_global_to_local(x1, node_id));
//
//	for (auto s : mesh.box(bound(x0, x1), node_id))
//	{
//		id_type code0 = mesh.out_code(s, s0);
//		id_type code1 = mesh.out_code(s, s1);
//
//		if ((code0 & code1) != 0)
//		{
//			continue;
//		}
//		else if ((code0 | code1) == 0)
//		{
//			res->insert(std::make_pair(s, i0));
//		}
//		else
//		{
//			intersection(mesh.coordinates(s - mesh_type::_DA),
//					mesh.coordinates(s + mesh_type::_DA), x0, x1);
//			res->insert(std::make_pair(s, i0));
//		}
//	}
//
//}
//template<typename TM, typename TX>
//void line_segment_cut_cell(TM const & mesh, typename TM::id_type node_id,
//		TX const &x0, TX const & x1, std::set<typename TM::id_type>* res)
//{
//	typedef TM mesh_type;
//	typedef typename mesh_type::coordinates_type coordinates_type;
//	typedef typename mesh_type::id_type id_type;
//
//	int m = 0;
//	for (int i = 0; i < 3; ++i)
//	{
//		if (mesh.dx()[i] > EPSILON)
//		{
//			m = std::max(m,
//					static_cast<int>(std::abs((x1[i] - x0[i]) / mesh.dx()[i])));
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
//			res->insert(std::get<0>(mesh.coordinates_global_to_local(
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

#endif /* CORE_MODEL_CUT_CELL_H_ */

