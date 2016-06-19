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
	(*f)->host_data = 0x0;
	(*f)->device_data = 0x0;

	size_type num_of_entities = spMeshGetNumberOfEntity(mesh, iform);

	CUDA_CHECK_RETURN(cudaMalloc((void ** ) &((*f)->device_data), num_of_entities * sizeof(Real)));

	(*f)->host_data = (Real*) malloc(num_of_entities * sizeof(Real));

}

MC_HOST void spDestroyField(sp_field_type **f)
{
	if (f != 0x0 && *f != 0x0)
	{
		if ((**f).device_data != 0x0)
		{
			CUDA_CHECK_RETURN(cudaFree((void** )((**f).device_data)))
		};

		if ((**f).host_data != 0x0)
		{
			free((void**) ((**f).host_data));
		}
		*f = 0x0;
	}
}
MC_HOST void spClearField(spMesh const *mesh, sp_field_type *f)
{
	size_type num_of_entities = spMeshGetNumberOfEntity(mesh, f->iform);

	if (f->device_data != 0x0)
	{
		CUDA_CHECK_RETURN(cudaMemset(f->device_data, 0, num_of_entities * sizeof(Real)));
	}
}

MC_HOST int spWriteField(spMesh const *mesh, sp_field_type *f, char const url[], int flag)
{
	size_type num_of_entities = spMeshGetNumberOfEntity(mesh, f->iform);

//	CUDA_CHECK_RETURN(
	cudaMemcpy((void*) (f->host_data), (void*) (f->device_data), num_of_entities * sizeof(Real), cudaMemcpyDefault)
//	)
			;
	int ndims = (f->iform == 1 || f->iform == 2) ? 4 : 3;
	hdf5_write_field(url, (void*) f->host_data, ndims, mesh->dims, mesh->offset, mesh->count, flag);
	return 0;
}
MC_HOST int spSyncField(spMesh const *mesh, sp_field_type *f)
{

}
