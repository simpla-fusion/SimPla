/*
 * spMesh.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#include "sp_def.h"
#include "spMesh.h"
#include "spSimPlaWrap.h"

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
	self->ndims = 3;
	self->offset.x = 0;
	self->offset.y = 0;
	self->offset.z = 0;
	self->offset.w = 0;
	self->count.x = self->dims.x;
	self->count.y = self->dims.y;
	self->count.z = self->dims.z;
	self->x_lower = 0;
	self->x_upper.x = self->dims.x;
	self->x_upper.y = self->dims.y;
	self->x_upper.z = self->dims.z;

	self->dims.w = 3;
	self->offset.w = 0;
	self->count.w = 3;

	self->threadsPerBlock.x = 4;
	self->threadsPerBlock.y = 4;
	self->threadsPerBlock.z = 4;

	self->number_of_shared_blocks = 0;
	self->private_block.x = self->dims.x;
	self->private_block.y = self->dims.y;
	self->private_block.z = self->dims.z;
	spObjectHostToDevice((spObject*) self);
}

MC_HOST_DEVICE size_type spMeshGetNumberOfEntity(spMesh const *self, int iform)
{
	return self->dims.x * self->dims.y * self->dims.z * ((iform == 0 || iform == 3) ? 1 : 3);
}

