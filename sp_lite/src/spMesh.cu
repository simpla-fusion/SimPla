/*
 * spMesh.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#include "sp_def.h"
#include "spMesh.h"

MC_HOST void spCreateMesh(spMesh **ctx)
{
	spCreateObject((spObject**) ctx, sizeof(spMesh));
//	*ctx = (spMesh *) malloc(sizeof(spMesh));

}
MC_HOST void spDestroyMesh(spMesh **ctx)
{
	free(*ctx);
}
MC_HOST void spInitializeMesh(spMesh *self)
{
}

MC_HOST_DEVICE size_type spMeshGetNumberOfEntity(spMesh const *self, int iform)
{
	return self->dims[0] * self->dims[1] * self->dims[2]
			* ((iform == 0 || iform == 3) ? 1 : 3);
}

