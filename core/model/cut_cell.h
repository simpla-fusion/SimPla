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

template<typename TPolygon, typename TM>
size_t cut_cell(TPolygon const & polygon,
		std::map<typename TM::id_type, std::list<typename TM::coordinates_type>>* res)
{
	typedef TM mesh_type;

	auto it = make_cycle_iterator(polygon.begin(), polygon.end());

	auto ie = polygon.end();

	typename mesh_type::id_type s0 = 0;

	typename mesh_type::coordinates_type x0;

	x0 = *(it--);

	for (; it != ie; ++it)
	{

		auto s1 = std::get<0>(mesh_type::coordinate_global_to_local(*it));

		if (s0 == s1)
		{
			(*res)[s1].push_back(*it);

			x0 = *it;

			continue;
		}



	}
}
}  // namespace simpla

#endif /* CORE_MODEL_CUT_CELL_H_ */
