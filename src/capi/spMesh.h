/*
 * spMesh.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPMESH_H_
#define SPMESH_H_

#include "../../src/sp_config.h"

struct spMesh_s;

typedef struct spMesh_s spMesh;

void spCreateMesh(spMesh **ctx);

void spDestroyMesh(spMesh **ctx);

#endif /* SPMESH_H_ */
