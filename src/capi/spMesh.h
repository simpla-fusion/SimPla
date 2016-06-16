/*
 * spMesh.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPMESH_H_
#define SPMESH_H_

#include "sp_def.h"
#include "../../src/sp_config.h"

struct spMesh_pimpl_s;
struct spMesh_s
{

	Real dx[3];
	Real inv_dx[3];

	size_type x_lower[3];
	size_type x_upper[3];
	size_type dims[3];

	size_type number_of_idx;
	size_type *cell_idx;

	spMesh_pimpl_s * pimpl_;
};

typedef struct spMesh_s spMesh;

void spCreateMesh(spMesh **ctx);

void spDestroyMesh(spMesh **ctx);

void spInitializeMesh(spMesh *self);

size_type spMeshGetNumberOfEntity(spMesh const *, int iform);

int spWriteMesh(const spMesh *ctx, const char *name, int flag);

int spReadMesh(spMesh *ctx, char const name[], int flag);

#endif /* SPMESH_H_ */
