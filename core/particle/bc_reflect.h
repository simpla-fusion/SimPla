/*
 * @file bc_reflect.h
 *
 *  created on: 2014-4-24
 *      Author: salmon
 */

#ifndef BC_REFLECT_H_
#define BC_REFLECT_H_

#include <memory>
#include <string>

#include "../gtl/ntuple.h"
#include "../utilities/primitives.h"
#include "../utilities/log.h"
#include "../numeric/geometric_algorithm.h"

namespace simpla
{

template<typename TParticle, typename TSurface>
void bc_reflect(TSurface const & surface, TParticle *particle)
{
	auto const & mesh = particle->mesh();
	/// FIXME need parallel optimize
	for (auto const &item : surface)
	{
		Real dist = std::get<0>(item->second);

		Vec3 const & normal = std::get<1>(item->second);

		coordinate_tuple x0;
		x0 = mesh.coordinates(item->first) + dist * normal;

		particle->modify([&](typename TParticle::value_type * p)
		{
			coordinate_tuple x;
			Vec3 v;
			Real f;

			std::tie(x, v, f) = particle->pull_back(*p);

			if (inner_product(x - x0, normal) > 0)
			{
				x = x0 + simpla::reflect(x - x0, normal);
				v = simpla::reflect(v, normal);
			}

			particle->push_forward(x, v, f,p);

		}, item.first);

	}
}
}
// namespace simpla

#endif /* BC_REFLECT_H_ */
