/*
 * spField.cu
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include "sp_def.h"
#include "spMesh.h"
#include "spField.h"

void spCreateField(const spMesh *mesh, sp_field_type **f, int iform)
{
	spCreateObject((spObject **) f, sizeof(sp_field_type));
	(*f)->number_of_entities = spMeshGetNumberOfEntity(mesh, iform);
	(*f)->data = (Real*) malloc(sizeof(Real) * spMeshGetNumberOfEntity(mesh, iform));

}

void spDestroyField(sp_field_type **f)
{

	if (*f != 0x0)
	{
		spDestroyObject((spObject**) &f);

		free((**f).data);
		free((*f));
		*f = 0x0;
	}
	spDestroyObject((spObject**) &f);
}
