/*
 * particle_constraint.h
 *
 *  Created on: 2014年4月21日
 *      Author: salmon
 */

#ifndef PARTICLE_CONSTRAINT_H_
#define PARTICLE_CONSTRAINT_H_
#include "material.h"
#include "surface.h"
namespace simpla
{

template<typename TM, typename TDict>
std::function<void(std::string const &, std::shared_ptr<ParticleBase<TM>>)> CreateParticleConstraint(
        Material<TM> const & material, TDict const & dict)
{
	std::function<void(std::string const &, std::shared_ptr<ParticleBase<TM>>)> res =
	        [](std::string const &, std::shared_ptr<ParticleBase<TM>>)
	        {
		        WARNING<<"Nothing to do!";
	        };

	typedef TM mesh_type;

	mesh_type const & mesh = material.mesh;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	return std::move(res);
}

}  // namespace simpla

#endif /* PARTICLE_CONSTRAINT_H_ */
