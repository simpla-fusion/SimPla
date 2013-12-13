/*
 * mesh_algorithm.h
 *
 *  Created on: 2013年12月13日
 *      Author: salmon
 */

#ifndef MESH_ALGORITHM_H_
#define MESH_ALGORITHM_H_
#include "../numeric/pointinpolygen.h"
namespace simpla
{

/**
 *
 * @param fun if fun(x)==true , vertex(x) is set to meida (media_idx)
 * @param media_idx > 0
 */
template<typename TM>
void SelectPointsInPolygen(TM const & mesh, std::vector<typename TM::coordinates_type> const polyline,
        std::function<void(typename TM::index_type const &, typename TM::coordinates_type const &)> const &fun)
{
	typedef TM mesh_type;

	PointInPolygen<typename mesh_type::coordinates_type> Check(polyline);

	mesh.TraversalCoordinates(0,

	[&](typename mesh_type::index_type const&s ,
			typename mesh_type:: coordinates_type const &x)
	{
		if(Check(x))
		{
			fun(s,x);
		}
	},

	mesh.WITH_GHOSTS);
}

}  // namespace simpla

#endif /* MESH_ALGORITHM_H_ */
