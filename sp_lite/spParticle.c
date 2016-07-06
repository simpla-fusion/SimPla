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
                                                      size_type size_in_byte)
{
    struct spParticleAttrEntity_s *res = &(pg->attrs[pg->number_of_attrs]);
    ++pg->number_of_attrs;
    strcpy(res->name, name);
    res->type_tag = type_tag;
    res->size_in_byte = size_in_byte;
    return res;
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

void spParticleDeploy(spParticle *sp, size_type PIC)
{
    if (sp->number_of_attrs <= 0) { return; }

    size_type number_of_cell = spMeshGetNumberOfEntity(sp->m, 3/*volume*/);

    sp->number_of_pages_per_cell = (PIC * 3 / SP_NUMBER_OF_ENTITIES_IN_PAGE) / 2;

    sp->max_number_of_pages = number_of_cell * sp->number_of_pages_per_cell;

    sp->max_number_of_particles = sp->max_number_of_pages * SP_NUMBER_OF_ENTITIES_IN_PAGE;

    size_type page_size_in_byte = (sizeof(spPage) + sp->number_of_attrs * sizeof(void *));

    spParallelDeviceMalloc((void **) (&(sp->m_pages_holder)), sp->max_number_of_pages * page_size_in_byte);

    spParallelDeviceMalloc((void **) (&(sp->buckets)), number_of_cell * sizeof(spPage *));

    size_type total_size = 0;

    for (int i = 0; i < sp->number_of_attrs; ++i)
    {
        sp->attrs[i].addr_offset = total_size;
        total_size += sp->attrs[i].size_in_byte * sp->max_number_of_particles;
    }

    spParallelDeviceMalloc((&(sp->data)), total_size);

}
