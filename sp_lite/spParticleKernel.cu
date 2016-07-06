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

__global__ void spParticleInitializeKernel(spPage **buckets, spPage *pages, void *data, size_type entity_size_in_byte,
		int number_of_pages_per_cell)
{
	assert(data != NULL);

#define MESH_ID (blockIdx.x + (blockIdx.y + blockIdx.z * gridDim.y) * gridDim.x)

	size_type page_size_in_byte = entity_size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE;
	if (threadIdx.x == 0)
	{
		void *page_offset = data + page_size_in_byte * ((MESH_ID) * number_of_pages_per_cell);

		spPage **t = &(buckets[MESH_ID]);

		for (int i = 0; i < number_of_pages_per_cell; ++i)
		{
			(*t) = pages + ((MESH_ID) * number_of_pages_per_cell + i);
			(*t)->data = data + page_size_in_byte * ((MESH_ID) * number_of_pages_per_cell + i);
			(*t)->flag = MESH_ID << 6;
			(*t)->next = NULL;
			t = &((*t)->next);
		}
	}

#undef MESH_ID

}

void spParticleInitialize(spParticle *sp)
{
	//@formatter:off
	spParticleInitializeKernel<<<sp->m->dims, NUMBER_OF_THREADS_PER_BLOCK>>>(sp->buckets, sp->m_pages_holder, sp->data,
			sp->entity_size_in_byte, sp->number_of_pages_per_cell);
	//@formatter:on
	spParallelGlobalSync();        //wait for iteration to finish

}
