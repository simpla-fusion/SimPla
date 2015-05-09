/*
 * bc_reflecting.h
 *
 *  created on: 2014-4-24
 *      Author: salmon
 */

#ifndef BC_REFLECTING_H_
#define BC_REFLECTING_H_

#include <memory>
#include <string>

#include "../gtl/ntuple.h"
#include "../utilities/primitives.h"
#include "../utilities/log.h"
#include "../numeric/geometric_algorithm.h"

namespace simpla
{

template<typename TParticle, typename TM>
void reflect(Surface<TM> const & surface, TParticle *particle)
{
	auto const & mesh = surface.mesh();
	/// FIXME need parallel optimize
	for (auto const &item : surface)
	{
		auto it = particle->find(item.first);

		if (it != particle->end())
		{
			Real dist = std::get<0>(item->second);

			Vec3 const & normal = std::get<1>(item->second);

			coordinates_type x0 = mesh.coordinates(it->first) + dist * normal;

			/// @NOTE should be vectorized
			for (auto & p : (*particle)[item.first])
			{
				coordinates_type x;
				Vec3 v;
				Real f;

				std::tie(x, v, f) = particle->pull_back(p);

				// if point is out of surface
				if (inner_product(x - x0, normal) > 0)
				{
					x = x0 + simpla::reflect(x - x0, normal);
					v = simpla::reflect(v, normal);
				}
				p = particle->push_forward(x, v, f);
			}
		}
	}
}
}
// namespace simpla

#endif /* BC_REFLECTING_H_ */
