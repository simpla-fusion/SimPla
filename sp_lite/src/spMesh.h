/*
 * spMesh.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPMESH_H_
#define SPMESH_H_

#include "sp_def.h"

struct spMesh_pimpl_s;
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

	spMesh_s *d_self;
};

typedef struct spMesh_s spMesh;

MC_HOST_DEVICE void spCreateMesh(spMesh **ctx);

MC_HOST_DEVICE void spDestroyMesh(spMesh **ctx);

MC_HOST_DEVICE void spInitializeMesh(spMesh *self);

MC_HOST_DEVICE size_type spMeshGetNumberOfEntity(spMesh const *, int iform);

MC_HOST_DEVICE int spWriteMesh(const spMesh *ctx, const char *name, int flag);

MC_HOST_DEVICE int spReadMesh(spMesh *ctx, char const name[], int flag);

#endif /* SPMESH_H_ */
