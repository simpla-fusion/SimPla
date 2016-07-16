/*
 * spField.cu
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include <stdlib.h>
#include "spParallel.h"
#include "sp_lite_def.h"
#include "spMesh.h"
#include "spField.h"

void spFieldCreate(const spMesh *mesh, spField **f, int iform)
{
    *f = (spField *) malloc(sizeof(spField));
    (*f)->m = mesh;
    (*f)->iform = iform;
    (*f)->host_data = NULL;
    (*f)->device_data = NULL;
}

void spFieldDestroy(spField **f)
{
    if (f != NULL && *f != NULL)
    {

        spParallelDeviceFree((void **) (&((**f).device_data)));

        spParallelHostFree((void **) (&((**f).host_data)));

        free(*f);
        *f = NULL;
    }
}

void spFieldDeploy(spField *f)
{
    if (f->device_data == NULL)
    {
        int num_of_entities = spMeshGetNumberOfEntity(f->m, f->iform);
        spParallelDeviceMalloc((void **) &(f->device_data), num_of_entities * sizeof(Real));
    }
}

void spFieldClear(spField *f)
{
    spFieldDeploy(f);
    int num_of_entities = spMeshGetNumberOfEntity(f->m, f->iform);
    spParallelMemset(f->device_data, 0, num_of_entities * sizeof(Real));
}

void spFieldWrite(spField *f, spIOStream *os, char const name[], int flag)
{
    int size_in_byte = spMeshGetNumberOfEntity(f->m, f->iform) * sizeof(Real);

    void *f_host;
    spParallelHostMalloc(&f_host, size_in_byte);
    spParallelMemcpy((f_host), (void *) (f->device_data), size_in_byte);
    int ndims = (f->iform == 1 || f->iform == 2) ? 4 : 3;
    size_type shape[4];
    size_type start[4];
    size_type count[4];

    dim3 d_shape, lower, upper;

    d_shape = spMeshGetShape(f->m);


    shape[0] = d_shape.x;
    shape[1] = d_shape.y;
    shape[2] = d_shape.z;
    shape[3] = 3;

    spMeshGetDomain(f->m, 0, &lower, &upper, NULL);

    start[0] = lower.x;
    start[1] = lower.y;
    start[2] = lower.z;
    start[3] = 0;

    count[0] = upper.x - lower.x;
    count[1] = upper.y - lower.y;
    count[2] = upper.z - lower.z;
    count[3] = 3;
    spIOStreamWriteSimple(os, name, SP_TYPE_Real, f_host, ndims, shape, start, NULL, count, NULL, flag);
//	free(f_host);
    spParallelHostFree(&f_host);
}

void spFieldRead(spField *f, spIOStream *os, char const name[], int flag)
{

}

void spFieldSync(spField *f)
{

}
