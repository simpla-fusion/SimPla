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

MC_HOST void spCreateField(const spMesh *mesh, sp_field_type **f, int iform)
{
	spCreateObject((spObject **) f, sizeof(sp_field_type));

	(*f)->number_of_entities = spMeshGetNumberOfEntity(mesh, iform);

	CUDA_CHECK_RETURN(cudaMalloc(&((*f)->data), (*f)->number_of_entities * sizeof(Real)));

	spObjectHostToDevice((spObject*) *f);

}

MC_HOST void spDestroyField(sp_field_type **f)
{
	if (*f != 0x0)
	{
		spFree((void **) &((**f).data));
		spDestroyObject((spObject**) f);
	}
	*f = 0x0;
}
