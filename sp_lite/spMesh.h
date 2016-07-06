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
#include "spMPI.h"

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

	spMPI * mpi_comm;
};

typedef struct spMesh_s spMesh;

void spMeshCreate(spMesh **ctx);

void spMeshDestroy(spMesh **ctx);

void spMeshDeploy(spMesh *self);

int spMeshWrite(const spMesh *ctx, const char *name, int flag);

int spMeshRead(spMesh *ctx, char const name[], int flag);

size_type spMeshGetNumberOfEntity(spMesh const *, int iform);

__constant__ int3 SP_NEIGHBOUR_OFFSET[27];
__constant__ unsigned int SP_NEIGHBOUR_OFFSET_flag[27];

#endif /* SPMESH_H_ */
