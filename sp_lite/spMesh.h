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
};

typedef struct spMesh_s spMesh;

void spCreateMesh(spMesh **ctx);

void spDestroyMesh(spMesh **ctx);

void spInitializeMesh(spMesh *self);

int spWriteMesh(const spMesh *ctx, const char *name, int flag);

int spReadMesh(spMesh *ctx, char const name[], int flag);

size_type spMeshGetNumberOfEntity(spMesh const *, int iform);



#endif /* SPMESH_H_ */
