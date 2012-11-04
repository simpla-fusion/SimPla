/*
 * particle_object.h
 *
 *  Created on: 2012-10-31
 *      Author: salmon
 */

#ifndef PARTICLE_OBJECT_H_
#define PARTICLE_OBJECT_H_
#include "simpla_def.h"
#include "engine/object.h"
#include "engine/arrayobject.h"

namespace simpla
{

struct ParticleObject
{
	Real Z, m, T;

	std::map<std::string, TR1::shared_ptr<Object> > fields;

	TR1::shared_ptr<Object> f;
};

}  // namespace simpla

#endif /* PARTICLE_OBJECT_H_ */
