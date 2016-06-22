/**
 * spParticle.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPPARTICLE_H_
#define SPPARTICLE_H_

#include "sp_def.h"
#include "spPage.h"
#include "spMesh.h"

struct spParticleDesc_attr_entity_s
{
	size_type size_in_byte;
	size_type type_tag;
	char name[255];
	byte_type * data;
};
#define SP_MAX_NUMBER_OF_PARTICLE_ATTR 32

struct spParticleSpecies_s
{

	Real mass;
	Real charge;

	size_type number_of_pages_per_cell;
	size_type max_number_of_particles;
	size_type max_number_of_pages;
	size_type page_size_in_byte;

	int number_of_attrs;
	struct spParticleDesc_attr_entity_s attrs[SP_MAX_NUMBER_OF_PARTICLE_ATTR];

	spPage *m_free_page;
	spPage *m_pages_holder;
	spPage ** buckets;

};

typedef struct spParticleSpecies_s sp_particle_type;

void spCreateParticle(const spMesh *ctx, sp_particle_type **pg);

void spDestroyParticle(sp_particle_type **sp);

int spParticleAddAttribute(sp_particle_type *pg, char const *name, int type_tag, int size_in_byte);

void spParticleInitialize(const spMesh *mesh, sp_particle_type *sp, size_type PIC);

int spWriteParticle(spMesh const *ctx, sp_particle_type const*f, char const url[], int flag);

int spReadParticle(spMesh const *ctx, sp_particle_type **f, char const url[], int flag);

int spSyncParticle(spMesh const *ctx, sp_particle_type * f);

#endif /* SPPARTICLE_H_ */
