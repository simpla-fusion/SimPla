/*
 * spMesh.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPMESH_H_
#define SPMESH_H_
#include "sp_def.h"
#include "spObject.h"

struct spMesh_s
{
	SP_OBJECT_HEAD
	float3 dx;
	float3 inv_dx;

	int ndims;

	dim3 x_lower;
	dim3 x_upper;
	int4 dims;

	int4 offset;
	int4 count;

	size_type number_of_idx;
	size_type *cell_idx;

	size_type number_of_shared_blocks;
	dim3 *shared_blocks;
	dim3 private_block;
	dim3 threadsPerBlock;
};

typedef struct spMesh_s spMesh;

MC_HOST void spCreateMesh(spMesh **ctx);

MC_HOST void spDestroyMesh(spMesh **ctx);

MC_HOST void spInitializeMesh(spMesh *self);

MC_HOST int spWriteMesh(const spMesh *ctx, const char *name, int flag);

MC_HOST int spReadMesh(spMesh *ctx, char const name[], int flag);

MC_HOST_DEVICE size_type spMeshGetNumberOfEntity(spMesh const *, int iform);

#endif /* SPMESH_H_ */
