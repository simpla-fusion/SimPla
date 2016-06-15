/*
 * spParticle.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#include "sp_def_cuda.h"
#include "spMesh.h"
#include "spParticle.h"
#include "spBucketFunction.h"

void spCreateParticle(spMesh *ctx, sp_particle_type **sp, size_type num_of_pic)
{
	CUDA_CHECK_RETURN(cudaMalloc(sp, sizeof(sp_particle_type)));

	spPagePoolCreate(&((*sp)->pool), (*sp)->entity_size_in_byte);

	CUDA_CHECK_RETURN(
			cudaMalloc((*sp)->buckets,
					ctx->number_of_cell * sizeof(bucket_type)));

	for (size_type s = 0, se = ctx->number_of_cell; s < se; ++s)
	{
		(*sp)->buckets[s] = 0x0;
	}
}

void spDestroyParticle(spMesh *ctx, sp_particle_type **sp)
{
	spPagePoolDestroy(&((*sp)->pool));

	for (size_type s = 0, se = ctx->number_of_cell; s < se; ++s)
	{
		spPageDestroy(&((*sp)->buckets[s]), (*sp)->pool);
	}
	CUDA_CHECK_RETURN(cudaFree(((*sp)->buckets)));
	CUDA_CHECK_RETURN(cudaFree(((*sp))));
	*sp = 0x0;
}
