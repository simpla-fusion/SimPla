/*
 * spParticle.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include "sp_def.h"
#include "spObject.h"
#include "spMesh.h"
#include "spParticle.h"
#include "spPage.h"
MC_HOST void spCreateParticle(const spMesh *mesh, sp_particle_type **sp,
		size_type entity_size_in_byte, size_type PIC)
{
	spCreateObject((spObject **) sp, sizeof(sp_particle_type));
	//	*sp = (sp_particle_type*) malloc(sizeof(sp_particle_type));

	size_type max_number_of_cell = spMeshGetNumberOfEntity(mesh, 3/*volume*/);

	size_type max_number_of_pages = max_number_of_cell
			* (PIC / SP_NUMBER_OF_ENTITIES_IN_PAGE + 1) * 2;

	size_type max_number_of_particle = max_number_of_pages
			* SP_NUMBER_OF_ENTITIES_IN_PAGE;

	(*sp)->entity_size_in_byte = entity_size_in_byte;

	CUDA_CHECK_RETURN(
			cudaMalloc(&((*sp)->m_data),
					max_number_of_particle * entity_size_in_byte));

	{
		spPage *t_pages = (spPage *) malloc(
				sizeof(spPage) * max_number_of_pages);

		for (size_type s = 0; s < max_number_of_pages; ++s)
		{
			t_pages[s].flag = 0;
			t_pages[s].tag = 0;
			t_pages[s].data = (byte_type*) ((*sp)->m_data)
					+ s * entity_size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE;
			t_pages[s].next = (*sp)->m_pages + (s + 1);

		}
		t_pages[max_number_of_pages - 1].next = 0x0;

		CUDA_CHECK_RETURN(
				cudaMalloc(&((*sp)->m_pages),
						max_number_of_pages * sizeof(spPage)));
		CUDA_CHECK_RETURN(
				cudaMemcpy((*sp)->m_pages, t_pages,
						max_number_of_pages * sizeof(spPage*),
						cudaMemcpyDefault));
		(*sp)->m_free_page = (*sp)->m_pages;
		free(t_pages);
	}
	CUDA_CHECK_RETURN(
			cudaMalloc(&((*sp)->buckets), max_number_of_cell * sizeof(spPage*)));
	CUDA_CHECK_RETURN(cudaMemset((*sp)->buckets, 0x0, cudaMemcpyDefault));

}
MC_HOST
void spDestroyParticle(sp_particle_type **sp)
{
	CUDA_CHECK_RETURN(cudaFree((*sp)->buckets));
	CUDA_CHECK_RETURN(cudaFree((*sp)->m_pages));
	CUDA_CHECK_RETURN(cudaFree((*sp)->m_data));

	free(*sp);
	*sp = 0x0;

}

MC_HOST_DEVICE spPage *
spParticleCreateBucket(sp_particle_type *p, size_type num)
{
	return spPagePopFrontN(&(p->m_free_page), num);
}

__global__
void spInitializeParticle_Kernel(const spMesh *mesh, sp_particle_type *pg,
		size_type NUM_OF_PIC)
{
}

MC_HOST
void spParticleInitialize(const spMesh *mesh, sp_particle_type *sp,
		size_type PIC)
{
	spObjectHostToDevice((spObject*) sp);

//	spInitializeParticle_Kernel<<<mesh->numBlocks, mesh->threadsPerBlock>>>(
//			(const spMesh *) mesh->self, (sp_particle_type *) sp->self, PIC);
}
MC_HOST int spWriteParticle(spMesh const *ctx, sp_particle_type const*f,
		char const name[], int flag)
{
	return 0;

}
