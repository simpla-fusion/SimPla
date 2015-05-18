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
#include "../gtl/iterator/sp_iterator_cycle.h"
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

template<typename TM, typename TSplicesIter>
size_t cut_cell(TM const & mesh, TSplicesIter const & ib,
		TSplicesIter const & ie,
		std::map<typename TM::id_type,
				typename std::iterator_traits<TSplicesIter>::value_type>* res)
{

	for (auto it = ib; it != ie; ++it)
	{
		line_segment_cut_cel(it, res);
	}
}

template<typename T0, typename T1, typename T2, typename T3, typename T4>
bool intersection_cell(T0 const & min, T1 const & max, T2 const & x0,
		T3 const & x1, T4 const & x3)
{
	return true;
}

bool intersection_cell(std::int64_t const & min, std::int64_t const & max,
		std::int64_t const & x0, std::int64_t const & x1,
		std::int64_t const & x3)
{
	return false;
}
template<typename TM, typename TI>
size_t cut_cell(TM const & mesh, typename TM::id_type node_id, TI const &i0,
		std::multimap<typename TM::id_type, TI>* res, int ZAXIS = 0)
{
	typedef TM mesh_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::id_type id_type;

	TI it = i0;
	coordinates_type x0 = *it;
	++it;
	coordinates_type x1 = *it;

	coordinates_type x2;

	coordinates_type xmin, xmax;

	std::tie(xmin, xmax) = bound(bound(x0, x1), x2);

	id_type s0 = std::get<0>(mesh.coordinates_global_to_local(xmin))
			- (mesh_type::_DA << 1UL);

	id_type s1 = std::get<0>(mesh.coordinates_global_to_local(xmax))
			+ (mesh_type::_DA << 1UL);

	typename mesh_type::range_type r(s0, s1);

	if ()
	{

	}
	auto r =;
	for (auto s : r)
	{
		if (intersection(mesh.coordinates(s - mesh_type::_DA),
				mesh.coordinates(s + mesh_type::_DA), x0, x1, x2))
		{
			res->insert(std::make_pair(s, i0));
		}
	}

	//8
}

template<typename TM, typename TI>
size_t triangle_cut_cel(TM const & mesh, TI const & i0,
		std::multimap<typename TM::id_type, TI>* res)
{
	TI it = i0;
	coordinates_type x0 = *it;
	++it;
	coordinates_type x1 = *it;
	++it;
	coordinates_type x2 = *it;

	auto r = mesh.box(bound(bound(x0, x1), x2));
	for (auto s : r)
	{
		for (auto s : mesh.box(x0, x1))
		{
			if (intersection(mesh.coordinates(s - mesh_type::_DA),
					mesh.coordinates(s + mesh_type::_DA), x0, x1, x2))
			{
				res->insert(std::make_pair(s, i0));
			}
		}
	}
}
}
// namespace simpla

#endif /* CORE_MODEL_CUT_CELL_H_ */
