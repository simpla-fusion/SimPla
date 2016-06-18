/*
 * spField.cu
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include "sp_def.h"
#include "spMesh.h"
#include "spField.h"
#include "spObject.h"
#include "spSimPlaWrap.h"

MC_HOST void spCreateField(const spMesh *mesh, sp_field_type **f, int iform)
{
	spCreateObject((spObject **) f, sizeof(sp_field_type));
//	*f = (sp_field_type *) malloc(sizeof(sp_field_type));
	(*f)->iform = iform;
	size_type num_of_entity = spMeshGetNumberOfEntity(mesh, iform);

	CUDA_CHECK_RETURN(cudaMalloc((void ** ) &((*f)->device_data), num_of_entity * sizeof(Real)));

}

MC_HOST void spDestroyField(sp_field_type **f)
{
	if (f != 0x0 && *f != 0x0)
	{
		spFree((void **) &((**f).device_data));
		spDestroyObject((spObject **) f);
	}
	*f = 0x0;
}
MC_HOST void spClearField(spMesh const *mesh, sp_field_type *f)
{
	size_type num_of_entity = spMeshGetNumberOfEntity(mesh, f->iform);

	if (f->device_data == 0x0)
	{
		CUDA_CHECK_RETURN(cudaMalloc((void ** ) &(f->device_data), num_of_entity * sizeof(Real)));
	}
	CUDA_CHECK_RETURN(cudaMemset(f->device_data, 0, num_of_entity * sizeof(float)));
}

MC_HOST int spWriteField(spMesh const *mesh, sp_field_type const *f, char const url[], int flag)
{
	size_type num_of_entity = spMeshGetNumberOfEntity(mesh, f->iform);
	Real * tmp = (Real*) malloc(num_of_entity * sizeof(Real));
	CUDA_CHECK_RETURN(cudaMemcpy(tmp, (void* )(f->device_data), num_of_entity * sizeof(Real), cudaMemcpyDeviceToHost));
	int ndims = (f->iform == 1 || f->iform == 2) ? 4 : 3;
	hdf5_write_field(url, (void*) tmp, ndims, mesh->dims, mesh->offset, mesh->count, flag);
	free(tmp);
	return 0;
}
MC_HOST int spSyncField(spMesh const *mesh, sp_field_type *f)
{

}
