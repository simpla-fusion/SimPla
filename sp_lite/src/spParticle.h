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

struct spParticleSpecies_s
{
	Real mass;
	Real charge;
	size_type entity_size_in_byte;
	spPagePool * pool;
	bucket_type ** buckets;

};
typedef struct spParticleSpecies_s sp_particle_type;

#define POINT_HEAD  SP_BUCKET_ENTITY_HEAD  Real r[3];
struct point_head
{
	POINT_HEAD
	byte_type data[];
};

MC_HOST_DEVICE void spCreateParticle(const spMesh *ctx, sp_particle_type **pg, size_type entity_size_in_byte, Real mass,
		Real charge);

MC_HOST_DEVICE void spDestroyParticle(sp_particle_type **pg);

MC_HOST int spWriteParticle(spMesh const *ctx, sp_particle_type const*f, char const name[], int flag);

MC_HOST int spReadParticle(spMesh const *ctx, sp_particle_type **f, char const name[], int flag);

MC_HOST int spSyncParticle(spMesh const *ctx, sp_particle_type **f, int flag);

MC_HOST_DEVICE bucket_type *spParticleCreateBucket(sp_particle_type const *p, size_type num);

#endif /* SPPARTICLE_H_ */
