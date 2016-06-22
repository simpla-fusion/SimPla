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
	for (int i = 0; i < SP_MAX_NUMBER_OF_PARTICLE_ATTR; ++i)
	{
		(*sp)->attrs[i].data = 0x0;
	}
	(*sp)->m_free_page = 0x0;
	(*sp)->m_pages_holder = 0x0;
	(*sp)->buckets = 0x0;

}

void spDestroyParticle(sp_particle_type **sp)
{

	for (int i = 0; i < SP_MAX_NUMBER_OF_PARTICLE_ATTR; ++i)
	{
		if ((*sp)->attrs[i].data != 0x0)
		{
			CUDA_CHECK_RETURN(cudaFree((*sp)->attrs[i].data));
		}
	}
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
__constant__ size_type page_size_in_byte;
__constant__ size_type number_of_pages_per_cell;
__constant__ size_type attrs_size_in_byte[SP_MAX_NUMBER_OF_PARTICLE_ATTR];
__global__ void spInitializeParticle_Kernel(spPage** buckets, spPage * pages, int num_of_attrs, void ** data)
{
#define MESH_ID blockDim.x 	+ (blockDim.y + blockDim.z * gridDim.y) * gridDim.x
	spPage** t = &(buckets[MESH_ID]);

	for (int i = 0; i < number_of_pages_per_cell; ++i)
	{
		*t = (spPage *) (((byte_type*) pages) + page_size_in_byte * (MESH_ID * number_of_pages_per_cell + i));
		for (int j = 0; j < num_of_attrs; ++j)
		{
			(*t)->data[j] = (byte_type*) (data[j])
					+ attrs_size_in_byte[j] * SP_NUMBER_OF_ENTITIES_IN_PAGE * (MESH_ID * number_of_pages_per_cell + i);
		}
		(*t)->flag = 0x0;
		t = &((*t)->next);
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

	sp->page_size_in_byte = sizeof(spPage) + sizeof(void*) * sp->number_of_attrs;

	sp->number_of_pages_per_cell = (PIC * 3 / SP_NUMBER_OF_ENTITIES_IN_PAGE) / 2;

	sp->max_number_of_pages = number_of_cell * sp->number_of_pages_per_cell;

	sp->max_number_of_particles = sp->max_number_of_pages * SP_NUMBER_OF_ENTITIES_IN_PAGE;

	CUDA_CHECK_RETURN(cudaMalloc(&(sp->m_pages_holder), sp->max_number_of_pages * sp->page_size_in_byte));

	CUDA_CHECK_RETURN(cudaMalloc(&(sp->buckets), number_of_cell * sizeof(spPage*)));

	for (int i = 0; i < sp->number_of_attrs; ++i)
	{
		CUDA_CHECK_RETURN(
				cudaMalloc((byte_type** )(&(sp->attrs[i].data)),
						sp->attrs[i].size_in_byte * sp->max_number_of_particles));
	}

	void *l_data[SP_MAX_NUMBER_OF_PARTICLE_ATTR];
	size_type l_size_in_byte[SP_MAX_NUMBER_OF_PARTICLE_ATTR];
	for (int i = 0; i < SP_MAX_NUMBER_OF_PARTICLE_ATTR; ++i)
	{
		l_size_in_byte[i] = sp->attrs[i].size_in_byte;
		l_data[i] = sp->attrs[i].data;
	}
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(page_size_in_byte, &(sp->page_size_in_byte), sizeof(size_type)));

	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(attrs_size_in_byte, &l_size_in_byte, sp->number_of_attrs * sizeof(size_type)));

	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(number_of_pages_per_cell, &(sp->number_of_pages_per_cell), sizeof(size_type)));

	void **data = 0x0;

	CUDA_CHECK_RETURN(cudaMalloc(&data, sp->number_of_attrs * sizeof(void*)));

	CUDA_CHECK_RETURN(cudaMemcpy(data, l_data, sp->number_of_attrs * sizeof(void*), cudaMemcpyDefault));

	spInitializeParticle_Kernel<<<mesh->dims, NUMBER_OF_THREADS_PER_BLOCK>>>(sp->buckets, sp->m_pages_holder,
			sp->number_of_attrs, data);

	CUDA_CHECK_RETURN(cudaFree(data));

}
int spWriteParticle(spMesh const *ctx, sp_particle_type const*f, char const name[], int flag)
{
	return 0;
}
int spSyncParticle(spMesh const *ctx, sp_particle_type *f)
{
	return 0;
}
