/*
 * spParticle.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include "sp_def.h"
#include "spMesh.h"
#include "spParticle.h"
#include "spBucketFunction.h"
MC_HOST void spCreateParticle(const spMesh *mesh, sp_particle_type **sp, size_type entity_size_in_byte, Real mass,
		Real charge)
{
	*sp = (sp_particle_type*) malloc(sizeof(sp_particle_type));
	(*sp)->entity_size_in_byte = entity_size_in_byte;
	(*sp)->mass = mass;
	(*sp)->charge = charge;
	(*sp)->buckets = (bucket_type**) malloc(spMeshGetNumberOfEntity(mesh, 3/*volume*/) * sizeof(bucket_type*));

}
MC_HOST_DEVICE
void spParticleInitialize(const spMesh *mesh, sp_particle_type **sp, size_type PIC)
{
	spPagePoolCreate(&((*sp)->pool), (*sp)->entity_size_in_byte, spMeshGetNumberOfEntity(mesh, 3/*volume*/) * PIC * 2);

}
MC_HOST_DEVICE
void spDestroyParticle(sp_particle_type **sp)
{

	spPagePoolDestroy(&((*sp)->pool));
	free((*sp)->buckets);
	free(*sp);

}

MC_HOST_DEVICE bucket_type *
spParticleCreateBucket(sp_particle_type const *p, size_type num)
{
	return spPageCreate(num, p->pool);
}

