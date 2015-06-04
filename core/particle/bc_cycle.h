/*
 * @file bc_cycle.h
 *
 *  created on: 2014-4-24
 *      Author: salmon
 */

#ifndef BC_CYCLIC_H_
#define BC_CYCLIC_H_

#include <map>

#include "../gtl/ntuple.h"
#include "../gtl/primitives.h"
#include "../model/surface.h"
#include "../numeric/geometric_algorithm.h"
#include "../utilities/primitives.h"

namespace simpla
{
template<typename TParticle, typename TM>
void bc_cycle(Surface<TM> const & surface,
		std::map<typename TM::id_type, typename TM::id_type> const & id_map,
		TParticle *particle)
{
	auto const & mesh = surface.mesh();
	/// FIXME need parallel optimize
	for (auto const &item : surface)
	{
		if (id_map.find(item.first) == id_map.end())
		{
			continue;
		}

		Real dist = std::get<0>(item->second);

		Vec3 const & normal = std::get<1>(item->second);

		typename TM::coordinate_tuple x0 = mesh.coordinates(item->first)
				+ dist * normal;

		Vec3 d;

		d = mesh.coordiantes(id_map.at(item.first))
				- mesh.coordinates(item->first);

		particle->modify([&](typename TParticle::value_type * p)
		{
			typename TM::ccoordinate_tuple x;
			Vec3 v;
			Real f;

			std::tie(x, v, f) = particle->pull_back(*p);

			if (inner_product(x - x0, normal) > 0)
			{
				x+=d;
				particle->push_forward(x, v, f, &p);
			}

		}, item.first);

	}
}
}  // namespace simpla

#endif /* BC_CYCLIC_H_ */
