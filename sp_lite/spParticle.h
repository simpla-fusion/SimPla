/**
 * spParticle.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPPARTICLE_H_
#define SPPARTICLE_H_

#include "sp_lite_def.h"

#define SP_MAX_NUMBER_OF_PARTICLE_ATTR 32

struct spPage_s;
struct spMesh_s;
struct spParticleAttrEntity_s
{
	int type_tag;
	size_type size_in_byte;
	size_type addr_offset;
	char name[255];
};

struct spParticle_s
{
	struct spMesh_s const * m;
	Real mass;
	Real charge;

	size_type max_number_of_particles;
	size_type max_number_of_pages;

	int number_of_attrs;
	struct spParticleAttrEntity_s attrs[SP_MAX_NUMBER_OF_PARTICLE_ATTR];

	void *data;
	struct spPage_s *m_free_page;
	struct spPage_s *m_pages_holder;
	struct spPage_s **buckets;

};

typedef struct spParticle_s spParticle;

void spParticleCreate(const struct spMesh_s *ctx, struct spParticle_s **pg);

void spParticleDestroy(struct spParticle_s **sp);

struct spParticleAttrEntity_s * spParticleAddAttribute(struct spParticle_s *pg, char const *name, int type_tag,
		int size_in_byte);

void spParticleDeploy(struct spParticle_s *sp, size_type PIC);

void spParticleWrite(struct spParticle_s const *f, char const url[], int flag);

void spParticleRead(struct spParticle_s *f, char const url[], int flag);

void spParticleSync(struct spParticle_s *f);

#endif /* SPPARTICLE_H_ */
