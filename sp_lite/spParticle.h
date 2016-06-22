/**
 * spParticle.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPPARTICLE_H_
#define SPPARTICLE_H_

#include "spObject.h"
#include "spPage.h"
#include "spMesh.h"


struct spParticleSpecies_s
{
	SP_OBJECT_HEAD
	Real mass;
	Real charge;
	size_type entity_size_in_byte;

	spPage *m_free_page;

	spPage *m_pages;
	spPage ** buckets;

	byte_type __align__(8) *m_data;
};
typedef struct spParticleSpecies_s sp_particle_type;

#define POINT_HEAD  SP_BUCKET_ENTITY_HEAD  Real r[3];
struct point_head
{
	byte_type __align__(8) data[];
};

MC_HOST void spCreateParticle(const spMesh *ctx, sp_particle_type **pg, size_type entity_size_in_byte, size_type PIC);

MC_HOST void spDestroyParticle(sp_particle_type **pg);

MC_HOST int spWriteParticle(spMesh const *ctx, sp_particle_type const*f, char const name[], int flag);

MC_HOST int spReadParticle(spMesh const *ctx, sp_particle_type **f, char const name[], int flag);

MC_HOST int spSyncParticle(spMesh const *ctx, sp_particle_type * f);

MC_HOST_DEVICE spPage *spParticleCreateBucket(sp_particle_type const *p, size_type num);

#endif /* SPPARTICLE_H_ */
