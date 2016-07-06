/**
 * spParticle.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include <stdlib.h>
#include <string.h>
#include "sp_lite_def.h"
#include "spParallel.h"

#include "spMesh.h"
#include "spPage.h"
#include "spParticle.h"

void spParticleCreate(const spMesh *mesh, spParticle **sp)
{
	*sp = (spParticle *) malloc(sizeof(spParticle));

	(*sp)->m = mesh;
	(*sp)->number_of_attrs = 0;
	(*sp)->data = NULL;
	(*sp)->m_free_page = NULL;
	(*sp)->m_pages_holder = NULL;
	(*sp)->buckets = NULL;

}

void spParticleDestroy(spParticle **sp)
{
	spParallelDeviceFree((*sp)->data);
	spParallelDeviceFree((void *) ((*sp)->buckets));
	spParallelDeviceFree((void *) ((*sp)->m_pages_holder));

	free(*sp);
	*sp = NULL;
}

struct spParticleAttrEntity_s *spParticleAddAttribute(spParticle *pg, char const *name, int type_tag,
		size_type size_in_byte, size_type offsetof)
{
	struct spParticleAttrEntity_s *res = &(pg->attrs[pg->number_of_attrs]);
	strcpy(res->name, name);
	res->type_tag = type_tag;
	res->size_in_byte = size_in_byte;
	if (offsetof < 0)
	{
		if (pg->number_of_attrs == 0)
		{
			offsetof = 0;
		}
		else
		{
			offsetof = pg->attrs[pg->number_of_attrs - 1].offsetof + pg->attrs[pg->number_of_attrs - 1].size_in_byte;
		}
	}
	res->offsetof = offsetof;
	++pg->number_of_attrs;
	return res;
}

void spParticleDeploy(spParticle *sp, size_type PIC)
{
	size_type number_of_cell = spMeshGetNumberOfEntity(sp->m, 3/*volume*/);

	sp->number_of_pages_per_cell = (PIC * 3 / SP_NUMBER_OF_ENTITIES_IN_PAGE) / 2;

	sp->entity_size_in_byte = sp->attrs[sp->number_of_attrs - 1].size_in_byte
			+ sp->attrs[sp->number_of_attrs - 1].offsetof;

	spParallelDeviceMalloc((void **) (&(sp->buckets)), number_of_cell * sizeof(spPage *));

	spParallelDeviceMalloc((void **) (&(sp->m_pages_holder)), number_of_cell * sp->number_of_pages_per_cell);

	spParallelDeviceMalloc((&(sp->data)),
			number_of_cell * sp->number_of_pages_per_cell * sp->entity_size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE);

}
void spParticleWrite(spParticle const *f, char const name[], int flag)
{
}

void spParticleRead(spParticle *f, char const url[], int flag)
{

}

void spParticleSync(spParticle *f)
{
}
