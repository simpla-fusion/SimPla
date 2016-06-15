/**
 * spParticle.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPPARTICLE_H_
#define SPPARTICLE_H_
#include "spMesh.h"
#include "spBucket.h"
struct spParticleSpecies_s;

typedef struct spParticleSpecies_s sp_particle_type;

void spCreateParticle(spMesh *ctx, sp_particle_type **pg, size_type num_of_pic);

void spDestroyParticle(spMesh *ctx, sp_particle_type **pg);

#define POINT_HEAD  SP_BUCKET_ENTITY_HEAD  Real r[3];

struct point_head
{
	POINT_HEAD
	byte_type data[];
};

#endif /* SPPARTICLE_H_ */
