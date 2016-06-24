/*
 * spParticle.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include "sp_def.h"
#include "spMesh.h"
#include "spParticle.h"
#include "spPage.h"

void spCreateParticle(const spMesh *mesh, sp_particle_type **sp)
{
	*sp = (sp_particle_type*) malloc(sizeof(sp_particle_type));

	(*sp)->number_of_attrs = 0;
	(*sp)->data = 0x0;
	(*sp)->m_free_page = 0x0;
	(*sp)->m_pages_holder = 0x0;
	(*sp)->buckets = 0x0;

}

void spDestroyParticle(sp_particle_type **sp)
{

	CUDA_CHECK_RETURN(cudaFree((*sp)->data));
	CUDA_CHECK_RETURN(cudaFree((*sp)->buckets));
	CUDA_CHECK_RETURN(cudaFree((*sp)->m_pages_holder));
	free(*sp);
	*sp = 0x0;
}
int spParticleAddAttribute(sp_particle_type *pg, char const *name, int type_tag, int size_in_byte)
{
	strcpy(pg->attrs[pg->number_of_attrs].name, name);
	pg->attrs[pg->number_of_attrs].type_tag = type_tag;
	pg->attrs[pg->number_of_attrs].size_in_byte = size_in_byte;
	++pg->number_of_attrs;
	return pg->number_of_attrs;
}

__global__ void spInitializeParticle_Kernel(spPage** buckets, spPage * pages, void * data, int num_of_attrs,
		struct spParticleAttrEntity_s * particle_attrs, int number_of_pages_per_cell)
{
	if (data == 0x0)
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
			(*t) = (spPage *) (page_offset + page_size_in_byte * i);

			for (int j = 0; j < num_of_attrs; ++j)
			{
				(*t)->data[j] = (byte_type*) (data) + particle_attrs[j].addr_offset
						+ particle_attrs[j].size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE
								* (MESH_ID * number_of_pages_per_cell + i);
			}
			(*t)->flag = MESH_ID << 6;
			(*t)->next = 0x0;
			t = &((*t)->next);
		}
	}

#undef MESH_ID

}

void spParticleInitialize(const spMesh *mesh, sp_particle_type *sp, size_type PIC)
{
	if (sp->number_of_attrs <= 0)
	{
		return;
	}

	size_type number_of_cell = spMeshGetNumberOfEntity(mesh, 3/*volume*/);

	size_type number_of_pages_per_cell = (PIC * 3 / SP_NUMBER_OF_ENTITIES_IN_PAGE) / 2;

	sp->max_number_of_pages = number_of_cell * number_of_pages_per_cell;

	sp->max_number_of_particles = sp->max_number_of_pages * SP_NUMBER_OF_ENTITIES_IN_PAGE;

	size_type page_size_in_byte = (sizeof(spPage) + sp->number_of_attrs * sizeof(void*));

	CUDA_CHECK_RETURN(cudaMalloc(&(sp->m_pages_holder), sp->max_number_of_pages * page_size_in_byte));

	CUDA_CHECK_RETURN(cudaMalloc(&(sp->buckets), number_of_cell * sizeof(spPage*)));

	size_type total_size = 0;

	for (int i = 0; i < sp->number_of_attrs; ++i)
	{
		sp->attrs[i].addr_offset = total_size;
		total_size += sp->attrs[i].size_in_byte * sp->max_number_of_particles;
	}

	CUDA_CHECK_RETURN(cudaMalloc((void** )(&(sp->data)), total_size));

	struct spParticleAttrEntity_s * particle_attrs = 0x0;

	CUDA_CHECK_RETURN(cudaMalloc((void** )(&particle_attrs), sizeof(sp->attrs)));

	CUDA_CHECK_RETURN(cudaMemcpy((void* )particle_attrs, (sp->attrs), sizeof(sp->attrs), cudaMemcpyDefault));

	spInitializeParticle_Kernel<<<mesh->dims, NUMBER_OF_THREADS_PER_BLOCK>>>(sp->buckets, sp->m_pages_holder, sp->data,
			sp->number_of_attrs, particle_attrs, number_of_pages_per_cell);

	cudaDeviceSynchronize();        //wait for iteration to finish

	CUDA_CHECK_RETURN(cudaFree((void* )particle_attrs));

}
int spWriteParticle(spMesh const *ctx, sp_particle_type const*f, char const name[], int flag)
{
	return 0;
}
int spSyncParticle(spMesh const *ctx, sp_particle_type *f)
{
	return 0;
}
