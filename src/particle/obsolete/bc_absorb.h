/**
 * @file bc_absorb.h
 *
 *  created on: 2014-4-24
 *      Author: salmon
 */

#ifndef BC_ABSORB_H_
#define BC_ABSORB_H_

#include <memory>
#include <string>

#include "../toolbox/ntuple.h"
#include "../sp_def.h"
#include "../model/surface.h"

namespace simpla
{
template<typename TParticle, typename TM>
void bc_absorb(Surface<TM> const & surface, TParticle *particle)
{
	auto const & mesh = surface.mesh();
	/// FIXME need parallel optimize
	for (auto const &item : surface)
	{

		Real dist = std::get<0>(item.second);

		Vec3 const & normal = std::get<1>(item.second);

		typename TM::coordinate_tuple x0 = mesh.coordinates(item.first)
				+ dist * normal;

		particle->remove(
				[&](typename TParticle::value_type & p)
				{
					return (inner_product(std::get<0>(particle->pull_back( p))- x0, normal) > 0);
				}

				, item.first);

	}
}
}  // namespace simpla

#endif /* BC_ABSORB_H_ */
