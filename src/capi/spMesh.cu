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
void
spCreateMesh (spMesh **ctx)
{
  CUDA_CHECK_RETURN(cudaMalloc (ctx, sizeof(spMesh)));
}
void
spDestroyMesh (spMesh **ctx)
{
  CUDA_CHECK_RETURN(cudaFree (*ctx));
  *ctx = 0x0;
}
