/*
 * spMesh.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#include "sp_def.h"
#include "sp_cuda_common.h"
#include "spMesh.h"

MC_HOST_DEVICE void spCreateMesh(spMesh **ctx)
{
	spCreateObject((spObject **) ctx, sizeof(spMesh));
}
MC_HOST_DEVICE void spDestroyMesh(spMesh **ctx)
{
	spDestroyObject((spObject **) ctx);
}
MC_HOST_DEVICE void spInitializeMesh(spMesh *self)
{
//	sp_malloc((void**) (&(self->pimpl_)), sizeof(struct spMesh_pimpl_s), true);

}
MC_HOST_DEVICE size_type spMeshGetNumberOfEntity(spMesh const *self, int iform)
{
	return self->dims[0] * self->dims[1] * self->dims[2] * ((iform == 0 || iform == 3) ? 1 : 3);
}

