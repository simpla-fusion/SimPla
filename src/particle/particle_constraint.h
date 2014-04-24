/*
 * particle_constraint.h
 *
 *  Created on: 2014年4月21日
 *      Author: salmon
 */

#ifndef PARTICLE_CONSTRAINT_H_
#define PARTICLE_CONSTRAINT_H_
#include "../utilities/visitor.h"
#include "particle_boundary.h"

namespace simpla
{

template<typename TM, typename TDict>
std::shared_ptr<VisitorBase> CreateParticleConstraint(TM const & material, TDict const & dict)
{
	return std::dynamic_pointer_cast<VisitorBase>(
	        std::shared_ptr<ParticleBoundary<typename TM::mesh_type>>(
	                new ParticleBoundary<typename TM::mesh_type>(material.mesh, dict)));
}

} // namespace simpla

#endif /* PARTICLE_CONSTRAINT_H_ */
