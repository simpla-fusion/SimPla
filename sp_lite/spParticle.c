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


    ADD_PARTICLE_ATTRIBUTE((*sp), struct spParticlePoint_s, int, flag);
    ADD_PARTICLE_ATTRIBUTE((*sp), struct spParticlePoint_s, Real, rx);
    ADD_PARTICLE_ATTRIBUTE((*sp), struct spParticlePoint_s, Real, ry);
    ADD_PARTICLE_ATTRIBUTE((*sp), struct spParticlePoint_s, Real, rz);

}

void spParticleDeploy(spParticle *sp, int PIC)
{
    size_type number_of_cell = spMeshGetNumberOfEntity(sp->m, 3/*volume*/);

    sp->max_number_of_pages = number_of_cell * (size_type) (PIC * 3 / SP_NUMBER_OF_ENTITIES_IN_PAGE) / 2;

    sp->entity_size_in_byte = (size_type) (sp->attrs[sp->number_of_attrs - 1].size_in_byte
                                           + sp->attrs[sp->number_of_attrs - 1].offsetof);

    sp->page_size_in_byte = sizeof(struct spPage_s) + sp->number_of_attrs * sizeof(void *);


    spParallelDeviceMalloc((void **) (&(sp->m_pages_holder)), sp->max_number_of_pages);

    spParallelDeviceMalloc((void **) (&(sp->buckets)), sp->page_size_in_byte * number_of_cell);

    spParallelDeviceMalloc((&(sp->data)), sp->entity_size_in_byte *
                                          sp->max_number_of_pages * SP_NUMBER_OF_ENTITIES_IN_PAGE);

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
                                                      size_type size_in_byte, size_type offset)
{
    struct spParticleAttrEntity_s *res = &(pg->attrs[pg->number_of_attrs]);
    strcpy(res->name, name);
    res->type_tag = type_tag;
    res->size_in_byte = size_in_byte;
    if (offsetof == -1)
    {
        if (pg->number_of_attrs == 0) { offset = 0; }
        else
        {
            offset = pg->attrs[pg->number_of_attrs - 1].offsetof +
                     pg->attrs[pg->number_of_attrs - 1].size_in_byte;
        }
    }
    res->offsetof = offset;
    ++pg->number_of_attrs;
    return res;
}


void spParticleWrite(spParticle const *f, spIOStream *os, const char name[], int flag)
{
    char curr_path[2048];

    spIOStreamPWD(os, curr_path);
    spIOStreamOpen(os, name);

    spIOStreamOpen(os, curr_path);

}

void spParticleRead(spParticle *f, char const url[], int flag)
{

}

void spParticleSync(spParticle *f)
{
}


MC_DEVICE void spUpdateParticleSortThreadKernel(int THREAD_ID, spPage **dest, spPage const *src, spPage **pool,
                                                size_type ele_size_in_byte, int tag)
{

    MC_SHARED int g_s_tail;
    MC_SHARED int g_d_tail;

    if (THREAD_ID == 0)
    {
        if ((*dest) == NULL)
        {
            *dest = *pool;
            *pool = (*pool)->next;
            (*dest)->next = NULL;
            (*dest)->tail = 0;
        }
        g_s_tail = 0;
        g_d_tail = (*dest)->tail;
    }
    spParallelThreadSync();

    int s_tail = spAtomicInc(&g_s_tail, 1);// return current value and s_tail+=1 equiv. s_tail++
    int d_tail = spAtomicInc(&g_d_tail, 1);

    while (src != NULL)
    {
        if (d_tail < SP_NUMBER_OF_ENTITIES_IN_PAGE)
        {
            while ((s_tail < SP_NUMBER_OF_ENTITIES_IN_PAGE) && P_GET_FLAG(src->data, s_tail) != tag)
            {
                s_tail = spAtomicInc(&g_s_tail, 1);
            }

            if (s_tail < SP_NUMBER_OF_ENTITIES_IN_PAGE)
            {
                P_GET_FLAG((*dest)->data, d_tail) = 0;

                spParallelMemcpy((*dest)->data + d_tail * ele_size_in_byte,
                                 src->data + s_tail * ele_size_in_byte,
                                 ele_size_in_byte);

                d_tail = spAtomicInc(&g_d_tail, 1);

                continue;
            }
        }
        spParallelThreadSync();

        if (d_tail == SP_NUMBER_OF_ENTITIES_IN_PAGE)
        {
            if (THREAD_ID == 0)
            {
                (*dest)->tail = SP_NUMBER_OF_ENTITIES_IN_PAGE;
            }

            dest = &((*dest)->next);

            if (THREAD_ID == 0)
            {
                if ((*dest) == NULL)
                {
                    *dest = *pool;
                    *pool = (*pool)->next;
                    (*dest)->next = NULL;
                }
                g_d_tail = (*dest)->tail;
            }
        }
        if (s_tail == SP_NUMBER_OF_ENTITIES_IN_PAGE)
        {
            src = src->next;
            if (THREAD_ID == 0) { g_s_tail = 0; }
        }

        spParallelThreadSync();
    }
}
