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
#include "spIO.h"

MC_HOST void spCreateField(const spMesh *mesh, sp_field_type **f, int iform)
{
	spCreateObject((spObject **) f, sizeof(sp_field_type));
//	*f = (sp_field_type *) malloc(sizeof(sp_field_type));
	(*f)->iform = iform;
	(*f)->number_of_entities = spMeshGetNumberOfEntity(mesh, iform);

	CUDA_CHECK_RETURN(cudaMalloc((void ** ) &((*f)->data), (*f)->number_of_entities * sizeof(Real)));

}

MC_HOST void spDestroyField(sp_field_type **f)
{
	if (f != 0x0 && *f != 0x0)
	{
		spFree((void **) &((**f).data));
		spDestroyObject((spObject **) f);
	}
	*f = 0x0;
}
MC_HOST int spWriteField(spMesh const *mesh, sp_field_type const *f, char const url[], int flag)
{
	size_type num_of_entity = spMeshGetNumberOfEntity(mesh, f->iform);

	Real * tmp = (Real*) malloc(num_of_entity * sizeof(Real));

	CUDA_CHECK_RETURN(cudaMemcpy(tmp, (void* )(f->data), num_of_entity * sizeof(Real), cudaMemcpyDeviceToHost));

	int ndims = (f->iform == 1 || f->iform == 2) ? 3 : 4;
	size_type dims[4];
	size_type offset[4];
	size_type count[4];

	for (int i = 0; i < 3; ++i)
	{
		dims[i] = mesh->dims[i];
		offset[i] = mesh->offset[i];
		count[i] = mesh->count[i];
	}
	dims[3] = 3;
	offset[3] = 0;
	count[3] = 3;

	hdf5_write(url, (void*) tmp, sizeof(Real), SP_DOUBLE, ndims, dims, offset, count, flag);
	free(tmp);
	return 0;
}
