/*
 * particle_boundary.h
 *
 *  Created on: 2013年12月28日
 *      Author: salmon
 */

#ifndef PARTICLE_BOUNDARY_H_
#define PARTICLE_BOUNDARY_H_

#include "../fetl/ntuple.h"
#include "../fetl/primitives.h"
#include "../mesh/media_tag.h"
#include "particle.h"

namespace simpla
{
enum
{
	REFELECT, ABSORB
};
template<class Engine, typename TMediaTag>
void Boundary(int flag, TMediaTag const &tag, typename TMediaTag::tag_type in, typename TMediaTag::tag_type out,
		Particle<Engine> * self, Particle<Engine> * other = nullptr)
{
	DEFINE_FIELDS(typename Engine::mesh_type);

	if (other == nullptr)
		other = self;

	tag.SelectBoundaryCell(Int2Type<0>(),

	[self,other](index_type src)
	{

		auto & cell = (*self)[src];

		auto pt = cell.begin();

		while (pt != cell.end())
		{
			auto p = pt;
			++pt;

			index_type dest=src;
			if (flag == REFELECT)
			{
				coordinates_type x;

				nTuple<3,Real> v;

				Engine::InvertTrans(p,&x,&v);

				dest=self->mesh.Refelect(src,&x,&v);

				Engine::Trans(x,v,&p);
			}

			if (dest != src)
			{
				other->data_[dest].splice(other->data_[dest].begin(), cell, p);
			}
			else
			{
				cell.erase(p);
			}

		}

	}, in, out, TMediaTag::ON_BOUNDARY);

}

}  // namespace simpla

#endif /* PARTICLE_BOUNDARY_H_ */
