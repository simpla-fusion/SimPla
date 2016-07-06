/**
 * spParticle.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include <cuda.h>
#include <string.h>
#include "sp_lite_def.h"
#include "spMesh.h"
#include "spPage.h"
#include "spParticle.h"

void spParticleCreate(const spMesh *mesh, spParticle **sp)
{
	*sp = (spParticle*) malloc(sizeof(spParticle));

	(*sp)->m = mesh;
	(*sp)->number_of_attrs = 0;
	(*sp)->data = NULL;
	(*sp)->m_free_page = NULL;
	(*sp)->m_pages_holder = NULL;
	(*sp)->buckets = NULL;

}

void spParticleDestroy(spParticle **sp)
{
//	CUDA_CHECK_RETURN(cudaFree((*sp)->data));
//	CUDA_CHECK_RETURN(cudaFree((*sp)->buckets));
//	CUDA_CHECK_RETURN(cudaFree((*sp)->m_pages_holder));
	free(*sp);
	*sp = NULL;
}
struct spParticleAttrEntity_s* spParticleAddAttribute(spParticle *pg, char const *name, int type_tag, int size_in_byte)
{
	struct spParticleAttrEntity_s* res = &(pg->attrs[pg->number_of_attrs]);
	++pg->number_of_attrs;
	strcpy(res->name, name);
	res->type_tag = type_tag;
	res->size_in_byte = size_in_byte;
	return res;
}

void spParticleWrite(spParticle const*f, char const name[], int flag)
{
}
void spParticleRead(spParticle *f, char const url[], int flag)
{

}
void spParticleSync(spParticle *f)
{
}
