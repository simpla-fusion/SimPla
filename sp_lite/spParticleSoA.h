/*
 * spParticleSoA.h
 *
 *  Created on: 2016年6月19日
 *      Author: salmon
 */

#ifndef SPPARTICLESOA_H_
#define SPPARTICLESOA_H_

#include "spObject.h"
#include "spPage.h"
#include "spMesh.h"

struct sp_point_s
{
 	int tag;
	Real  r[3];
	Real  v[3];
	Real  f;
	Real  w;
};
struct spDefaultPage_aos_s
{
	SP_PAGE_HEAD
	struct sp_point_s data[SP_NUMBER_OF_ENTITIES_IN_PAGE];

};
struct spDefaultPage_soa_s
{
	SP_PAGE_HEAD
	int tag[SP_NUMBER_OF_ENTITIES_IN_PAGE];
	Real  r[3][SP_NUMBER_OF_ENTITIES_IN_PAGE];
	Real  v[3][SP_NUMBER_OF_ENTITIES_IN_PAGE];
	Real  f[SP_NUMBER_OF_ENTITIES_IN_PAGE];
	Real  w[SP_NUMBER_OF_ENTITIES_IN_PAGE];
};
struct spParticleSpecies_s
{
	SP_OBJECT_HEAD
	Real mass;
	Real charge;
	size_type entity_size_in_byte;
	spPage *m_free_page;
	spPage *m_pages;
	spPage ** buckets;

 };
typedef struct spParticleSpecies_s sp_particle_type;

#define POINT_HEAD  SP_BUCKET_ENTITY_HEAD  Real r[3];
struct point_head
{
	POINT_HEAD
	byte_type data[];
};

MC_HOST void spCreateParticle(const spMesh *ctx, sp_particle_type **pg, size_type entity_size_in_byte, size_type PIC);

MC_HOST void spDestroyParticle(sp_particle_type **pg);

MC_HOST int spWriteParticle(spMesh const *ctx, sp_particle_type const*f, char const name[], int flag);

MC_HOST int spReadParticle(spMesh const *ctx, sp_particle_type **f, char const name[], int flag);

MC_HOST int spSyncParticle(spMesh const *ctx, sp_particle_type * f);

MC_HOST_DEVICE spPage *spParticleCreateBucket(sp_particle_type const *p, size_type num);
#endif /* SPPARTICLESOA_H_ */
