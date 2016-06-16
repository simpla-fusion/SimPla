/*
 * spParticle.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include "sp_cuda_common.h"
#include "sp_def.h"
#include "spMesh.h"
#include "spParticle.h"
#include "spBucketFunction.h"
MC_HOST void spCreateParticle(const spMesh *mesh, sp_particle_type **sp, size_type entity_size_in_byte, Real mass,
		Real charge)
{

	sp_particle_type t_sp;
	t_sp.entity_size_in_byte = entity_size_in_byte;
	t_sp.mass = mass;
	t_sp.charge = charge;

	CUDA_CHECK_RETURN(cudaMalloc(&t_sp.buckets, spMeshGetNumberOfEntity(mesh, 3/*volume*/) * sizeof(Real)));

	CUDA_CHECK_RETURN(cudaMalloc(&t_sp.pool, sizeof(spPagePool)));

	CUDA_CHECK_RETURN(cudaMalloc(sp, sizeof(sp_particle_type)));

	CUDA_CHECK_RETURN(cudaMemcpy(*sp, &t_sp, sizeof(sp_particle_type), cudaMemcpyHostToDevice));

}

__global__ void spParticleInitialize(const spMesh *ctx, sp_particle_type **sp, size_type entity_size_in_byte)
{

}
void spDestroyParticle(const spMesh *ctx, sp_particle_type **sp)
{
	CUDA_CHECK(2);

	spPagePoolDestroy(&((*sp)->pool));
	CUDA_CHECK(2);
	cudaFree((*sp)->buckets);
	CUDA_CHECK(3);
	cudaFree(*sp);

}

MC_HOST MC_DEVICE bucket_type * spParticleCreateBucket(sp_particle_type const *p, size_type num)
{
	return spPageCreate(num, p->pool);
}

