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
	float dx[3];

	int ndims;

	size_t dims[3];
	size_t x_lower[3];
	size_t x_upper[3];

	int offset[4];
	int count[4];

	size_type number_of_idx;
	size_type *cell_idx;

	size_type number_of_shared_blocks;
	size_t *shared_blocks;
	size_t private_block[3];
	size_t threadsPerBlock[3];

	spDistributedObject * dist_obj;

};

typedef struct spMesh_s spMesh;

void spMeshCreate(spMesh **ctx);

void spMeshDestroy(spMesh **ctx);

void spMeshDeploy(spMesh *self);

void spMeshWrite(const spMesh *ctx, const char *name, int flag);

void spMeshRead(spMesh *ctx, char const name[], int flag);

size_type spMeshGetNumberOfEntity(spMesh const *, int iform);

int SP_NEIGHBOUR_OFFSET[27][3];
unsigned int SP_NEIGHBOUR_OFFSET_flag[27];

#ifdef __CUDACC__
__constant__ int3 SP_NEIGHBOUR_OFFSET_DEVICE[27];
__constant__ unsigned int SP_NEIGHBOUR_OFFSET_flag_DEVICE[27];
#endif
#endif /* SPMESH_H_ */
