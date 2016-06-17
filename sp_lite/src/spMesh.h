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
	Real dx[3];
	Real inv_dx[3];

	size_type x_lower[3];
	size_type x_upper[3];
	size_type dims[3];

	size_type number_of_idx;
	size_type *cell_idx;

	dim3 numBlocks;
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
