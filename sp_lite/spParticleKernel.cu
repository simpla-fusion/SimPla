/*
 * spParticleKernel.cu
 *
 *  Created on: 2016年7月6日
 *      Author: salmon
 */
#include "spParticle.h"
#include "spPage.h"
#include "spMesh.h"
#include "spParallel.h"

__global__ void spInitializeParticle_Kernel(spPage **buckets, spPage *pages, void *data, int num_of_attrs,
                                            struct spParticleAttrEntity_s *particle_attrs, int number_of_pages_per_cell)
{
    if (data == NULL)
    {
        return;
    }

    // TODO need optimize
#define MESH_ID (blockIdx.x + (blockIdx.y + blockIdx.z * gridDim.y) * gridDim.x)

    size_type page_size_in_byte = (sizeof(spPage) + num_of_attrs * sizeof(void *));
    if (threadIdx.x == 0)
    {
        void *page_offset = ((byte_type *) pages) + page_size_in_byte * ((MESH_ID) * number_of_pages_per_cell);

        spPage **t = &(buckets[MESH_ID]);

        for (int i = 0; i < number_of_pages_per_cell; ++i)
        {
            (*t) = (spPage *) ((byte_type *) (page_offset) + page_size_in_byte * i);

            for (int j = 0; j < num_of_attrs; ++j)
            {
                (*t)->data[j] = (byte_type *) (data) + particle_attrs[j].addr_offset
                                + particle_attrs[j].size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE
                                  * (MESH_ID * number_of_pages_per_cell + i);
            }
            (*t)->flag = MESH_ID << 6;
            (*t)->next = NULL;
            t = &((*t)->next);
        }
    }

#undef MESH_ID

}

void spParticleInitialize(spParticle *sp)
{
    struct spParticleAttrEntity_s *particle_attrs = NULL;

    spParallelDeviceMalloc((void **) (&particle_attrs), sizeof(sp->attrs));

    spParallelMemcpy((void *) particle_attrs, (void *) (sp->attrs), sizeof(sp->attrs));

    dim3 dims;
    dims.x = sp->m->dims.x;
    dims.y = sp->m->dims.y;
    dims.z = sp->m->dims.z;
    //@formatter:off
    spInitializeParticle_Kernel <<< dims, NUMBER_OF_THREADS_PER_BLOCK >>> (sp->buckets, sp->m_pages_holder, sp->data,
            sp->number_of_attrs, particle_attrs, sp->number_of_pages_per_cell);
    //@formatter:on
    spParallelGlobalSync();        //wait for iteration to finish

    spParallelDeviceFree((void *) particle_attrs);

}
