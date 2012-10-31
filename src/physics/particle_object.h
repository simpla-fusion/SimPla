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
	TR1::shared_ptr<Object> n;
	TR1::shared_ptr<Object> J;
	TR1::shared_ptr<Object> P;
	TR1::shared_ptr<Object> f;
};

}  // namespace simpla

#endif /* PARTICLE_OBJECT_H_ */
