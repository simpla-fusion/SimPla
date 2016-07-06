/*
 * spField.cu
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include <assert.h>
#include "sp_lite_def.h"
#include "spMesh.h"
#include "spField.h"
#include "spObject.h"

void spFieldCreate(const spMesh *mesh, spField **f, int iform)
{
	spObjectCreate((spObject **) f, sizeof(spField));
//	*f = (sp_field_type *) malloc(sizeof(sp_field_type));
	(*f)->m = mesh;
	(*f)->iform = iform;
	(*f)->host_data = NULL;
	(*f)->device_data = NULL;

	size_type num_of_entities = spMeshGetNumberOfEntity(mesh, iform);

	CUDA_CHECK_RETURN(cudaMalloc((void ** ) &((*f)->device_data), num_of_entities * sizeof(Real)));

	(*f)->host_data = (Real*) malloc(num_of_entities * sizeof(Real));

}

void spFieldDestroy(spField **f)
{
	if (f != NULL && *f != NULL)
	{
		if ((**f).device_data != NULL)
		{
			CUDA_CHECK_RETURN(cudaFree((void** )((**f).device_data)));
		};

		if ((**f).host_data != NULL)
		{
			free((void**) ((**f).host_data));
		}
		*f = NULL;
	}
}
void spFieldClear(spField *f)
{
	size_type num_of_entities = spMeshGetNumberOfEntity(f->m, f->iform);

	if (f->device_data != NULL)
	{
		CUDA_CHECK_RETURN(cudaMemset(f->device_data, 0, num_of_entities * sizeof(Real)));
	}
}

void spFieldWrite(spField *f, spIOStream * file, char const url[], int flag)
{
//	size_type num_of_entities = spMeshGetNumberOfEntity(mesh, f->iform);
//	assert(f->host_data != 0);
//	CUDA_CHECK_RETURN(cudaMemcpy((void* ) (f->host_data), (void* ) (f->device_data), num_of_entities * sizeof(Real),
//					cudaMemcpyDeviceToHost));
//	int ndims = (f->iform == 1 || f->iform == 2) ? 4 : 3;
//	hdf5_write_field(url, (void*) f->host_data, ndims, mesh->dims, mesh->offset, mesh->count, flag);

}
void spFieldRead(spField * f, spIOStream * os, char const name[], int flag)
{

}

void spFieldSync(spField *f)
{

}
