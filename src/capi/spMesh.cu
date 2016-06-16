/*
 * spMesh.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#include "../sp_config.h"
#include "sp_def.h"
#include "sp_cuda_common.h"
#include "spMesh.h"

struct spMesh_pimpl_s
{
	dim3 numBlocks;
	dim3 threadsPerBlock;
};

void spCreateMesh(spMesh **ctx)
{
	CUDA_CHECK_RETURN(cudaMalloc(ctx, sizeof(spMesh)));
}
void spDestroyMesh(spMesh **ctx)
{
	CUDA_CHECK_RETURN(cudaFree(*ctx));
	*ctx = 0x0;
}
void spInitializeMesh(spMesh *self)
{
//	sp_malloc((void**) (&(self->pimpl_)), sizeof(struct spMesh_pimpl_s), true);

}
MC_HOST_DEVICE size_type spMeshGetNumberOfEntity(spMesh const *self, int iform)
{
	return self->dims[0] * self->dims[1] * self->dims[2] * ((iform == 0 || iform == 3) ? 1 : 3);
}

