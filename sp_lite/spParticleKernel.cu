/*
 * spParticleKernel.cu
 *
 *  Created on: 2016年7月6日
 *      Author: salmon
 */
#include "spParticle.h"
#include "spPage.h"
#include "spMesh.h"

__global__ void spInitializeParticle_Kernel(spPage** buckets, spPage * pages, void * data, int num_of_attrs,
		struct spParticleAttrEntity_s * particle_attrs, int number_of_pages_per_cell)
{
	if (data == NULL)
	{
		return;
	}

	// TODO need optimize
#define MESH_ID (blockIdx.x + (blockIdx.y + blockIdx.z * gridDim.y) * gridDim.x)

	size_type page_size_in_byte = (sizeof(spPage) + num_of_attrs * sizeof(void*));
	if (threadIdx.x == 0)
	{
		void * page_offset = ((byte_type*) pages) + page_size_in_byte * ((MESH_ID) * number_of_pages_per_cell);

		spPage** t = &(buckets[MESH_ID]);

		for (int i = 0; i < number_of_pages_per_cell; ++i)
		{
			(*t) = (spPage *) ((byte_type*) (page_offset) + page_size_in_byte * i);

			for (int j = 0; j < num_of_attrs; ++j)
			{
				(*t)->data[j] = (byte_type*) (data) + particle_attrs[j].addr_offset
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

void spParticleDeploy(spParticle *sp, size_type PIC)
{
	if (sp->number_of_attrs <= 0)
	{
		return;
	}

	size_type number_of_cell = spMeshGetNumberOfEntity(sp->m, 3/*volume*/);

	size_type number_of_pages_per_cell = (PIC * 3 / SP_NUMBER_OF_ENTITIES_IN_PAGE) / 2;

	sp->max_number_of_pages = number_of_cell * number_of_pages_per_cell;

	sp->max_number_of_particles = sp->max_number_of_pages * SP_NUMBER_OF_ENTITIES_IN_PAGE;

	size_type page_size_in_byte = (sizeof(spPage) + sp->number_of_attrs * sizeof(void*));

	CUDA_CHECK_RETURN(cudaMalloc((void** )(&(sp->m_pages_holder)), sp->max_number_of_pages * page_size_in_byte));

	CUDA_CHECK_RETURN(cudaMalloc((void** )(&(sp->buckets)), number_of_cell * sizeof(spPage*)));

	size_type total_size = 0;

	for (int i = 0; i < sp->number_of_attrs; ++i)
	{
		sp->attrs[i].addr_offset = total_size;
		total_size += sp->attrs[i].size_in_byte * sp->max_number_of_particles;
	}

	CUDA_CHECK_RETURN(cudaMalloc((void** )(&(sp->data)), total_size));

	struct spParticleAttrEntity_s * particle_attrs = NULL;

	CUDA_CHECK_RETURN(cudaMalloc((void** )(&particle_attrs), sizeof(sp->attrs)));

	CUDA_CHECK_RETURN(cudaMemcpy((void* )particle_attrs, (sp->attrs), sizeof(sp->attrs), cudaMemcpyDefault));

	dim3 dims;
	dims.x = sp->m->dims[0];
	dims.y = sp->m->dims[1];
	dims.z = sp->m->dims[2];

	spInitializeParticle_Kernel<<<dims, NUMBER_OF_THREADS_PER_BLOCK>>>(sp->buckets, sp->m_pages_holder, sp->data,
			sp->number_of_attrs, particle_attrs, number_of_pages_per_cell);

	cudaDeviceSynchronize();        //wait for iteration to finish

	CUDA_CHECK_RETURN(cudaFree((void* )particle_attrs));

}
