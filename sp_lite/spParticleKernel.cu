/*
 * spParticleKernel.cu
 *
 *  Created on: 2016年7月6日
 *      Author: salmon
 */
#include <assert.h>
#include "spParticle.h"
#include "spPage.h"
#include "spMesh.h"
#include "spParallel.h"

MC_GLOBAL void spParticleInitializeKernel(spPage **buckets, spPage *pages, void *data, size_type entity_size_in_byte,
                                          size_type page_size_in_byte)
{
    assert(data != NULL);


    MC_FOREACH_BLOCK_ID(BLOCK_ID)
    {

        spPage **t = &(buckets[BLOCK_ID]);
        (*t) = (spPage *) ((void *) pages + page_size_in_byte * BLOCK_ID);
        (*t)->next = NULL;
        (*t)->flag.v = BLOCK_ID;
        (*t)->data = data + entity_size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE * BLOCK_ID;


    }

#undef BLOCK_ID

}

void spParticleInitialize(spParticle *sp)
{
    //@formatter:off
	spParticleInitializeKernel<<<sp->m->dims, NUMBER_OF_THREADS_PER_BLOCK>>>(sp->buckets, sp->m_pages_holder, sp->data,
			sp->entity_size_in_byte, sp->page_size_in_byte);
	//@formatter:on
    spParallelThreadSync();        //wait for iteration to finish

}
