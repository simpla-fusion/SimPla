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

void spCreateField(const spMesh *mesh, sp_field_type **f, int iform)
{
#if !defined(__CUDA_ARCH__)

	CUDA_CHECK_RETURN(cudaMalloc(f, sizeof(sp_field_type)));
	CUDA_CHECK(spMeshGetNumberOfEntity(mesh, iform));
	Real * data=0x0;
	CUDA_CHECK_RETURN(cudaMalloc(&data, spMeshGetNumberOfEntity(mesh, iform) * sizeof(Real)));
	cudaMemcpy(&((**f).data),&data,sizeof(Real*),cudaMemcpyHostToDevice);
#else
	*f = (sp_field_type*) malloc(sizeof(sp_field_type));
	(*f)->data = (Real*) malloc(sizeof(Real) * spMeshGetNumberOfEntity(mesh, iform));
#endif
}

void spDestroyField(const spMesh *ctx, sp_field_type **f)
{
#if !defined(__CUDA_ARCH__)
	Real * data=0x0;
	cudaMemcpy(&data,&((**f).data),sizeof(Real*),cudaMemcpyDeviceToHost);
	CUDA_CHECK_RETURN(cudaFree(data));
	CUDA_CHECK_RETURN(cudaFree((*f)));
	*f = 0x0;
#else
	free((*f)->data);
	free(*f);
#endif
}
