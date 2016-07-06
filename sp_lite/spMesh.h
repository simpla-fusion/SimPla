/*
 * spMesh.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPMESH_H_
#define SPMESH_H_
#include "sp_lite_def.h"
#include "spObject.h"

struct spMesh_s
{
	float3 dx;

	int ndims;

	dim3 x_lower;
	dim3 x_upper;
	dim3 dims;

	int4 offset;
	int4 count;

	size_type number_of_idx;
	size_type *cell_idx;

	size_type number_of_shared_blocks;
	dim3 *shared_blocks;
	dim3 private_block;
	dim3 threadsPerBlock;

	spDistributedObject * dist_obj;

};

typedef struct spMesh_s spMesh;

void spMeshCreate(spMesh **ctx);

void spMeshDestroy(spMesh **ctx);

void spMeshDeploy(spMesh *self);

void spMeshWrite(const spMesh *ctx, const char *name, int flag);

void spMeshRead(spMesh *ctx, char const name[], int flag);

size_type spMeshGetNumberOfEntity(spMesh const *, int iform);

__constant__ int3 SP_NEIGHBOUR_OFFSET[27];
__constant__ unsigned int SP_NEIGHBOUR_OFFSET_flag[27];

#endif /* SPMESH_H_ */
