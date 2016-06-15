/*
 * spField.cu
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include "sp_def.h"
#include "sp_cuda_common.h"
#include "spMesh.h"
#include "spField.h"

void
spCreateField(const spMesh *ctx, sp_field_type **f, int iform)
{
  CUDA_CHECK_RETURN(cudaMalloc (f, sizeof(sp_field_type)));
  CUDA_CHECK_RETURN(
	  cudaMalloc (
		  &((*f)->data),
		  ctx->number_of_cell * ((iform == 1 || iform == 2) ? 3 : 1)
			  * sizeof(Real)));
}

void
spDestroyField(const spMesh *ctx, sp_field_type **f)
{
  CUDA_CHECK_RETURN(cudaFree ((*f)->data));
  CUDA_CHECK_RETURN(cudaFree ((*f)));
  *f = 0x0;
}
