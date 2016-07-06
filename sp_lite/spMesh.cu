/*
 * spMesh.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include <assert.h>
#include "sp_def.h"
#include "spMesh.h"

void spMeshCreate(spMesh **ctx)
{
	*ctx = (spMesh *) malloc(sizeof(spMesh));

}
void spMeshDestroy(spMesh **ctx)
{
	free(*ctx);
}
void spMeshDeploy(spMesh *self)
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

//	self->dims.w = 3;
//	self->offset.w = 0;
//	self->count.w = 3;

	self->threadsPerBlock.x = 4;
	self->threadsPerBlock.y = 4;
	self->threadsPerBlock.z = 4;

	self->number_of_shared_blocks = 0;
	self->private_block.x = self->dims.x;
	self->private_block.y = self->dims.y;
	self->private_block.z = self->dims.z;

	/**          -1
	 *
	 *    -1     0    1
	 *
	 *           1
	 */
	/**
	 *\verbatim
	 *                ^y
	 *               /
	 *        z     /
	 *        ^    /
	 *    PIXEL0 110-------------111 VOXEL
	 *        |  /|              /|
	 *        | / |             / |
	 *        |/  |    PIXEL1  /  |
	 * EDGE2 100--|----------101  |
	 *        | m |           |   |
	 *        |  010----------|--011 PIXEL2
	 *        |  / EDGE1      |  /
	 *        | /             | /
	 *        |/              |/
	 *       000-------------001---> x
	 *                       EDGE0
	 *
	 *\endverbatim
	 */
	int3 neighbour_offset[27];
	int neighbour_flag[27];
	int count = 0;
	for (int i = -1; i <= 1; ++i)
		for (int j = -1; j <= 1; ++j)
			for (int k = -1; k <= 1; ++k)
			{
				neighbour_offset[count].x = i;
				neighbour_offset[count].y = j;
				neighbour_offset[count].z = k;
				neighbour_flag[count] = (i + 1) | ((j + 1) << 2) | ((k + 1) << 4);
				++count;
			}
	assert(count == 27);
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(SP_NEIGHBOUR_OFFSET, neighbour_offset, sizeof(neighbour_offset)));
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(SP_NEIGHBOUR_OFFSET_flag, neighbour_flag, sizeof(neighbour_flag)));

}

size_type spMeshGetNumberOfEntity(spMesh const *self, int iform)
{
	return self->dims.x * self->dims.y * self->dims.z * ((iform == 0 || iform == 3) ? 1 : 3);
}

