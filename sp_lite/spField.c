//
// Created by salmon on 16-7-20.
//


#include "sp_lite_def.h"
#include <stdlib.h>

#include "spParallel.h"
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
        size_type num_of_entities = spMeshGetNumberOfEntity(f->m, f->iform);

        spParallelDeviceAlloc((void **) &(f->device_data), num_of_entities * sizeof(Real));
    }
}

void spFieldClear(spField *f)
{
    spFieldDeploy(f);

    size_type num_of_entities = spMeshGetNumberOfEntity(f->m, f->iform);

    spParallelMemset(f->device_data, 0, num_of_entities * sizeof(Real));
}

void spFieldWrite(spField *f, spIOStream *os, char const name[], int flag)
{
    size_type size_in_byte = spMeshGetNumberOfEntity(f->m, f->iform) * sizeof(Real);

    void *f_host;

    spParallelHostAlloc(&f_host, size_in_byte);

    spParallelMemcpy((f_host), (void *) (f->device_data), size_in_byte);

    int ndims = (f->iform == 1 || f->iform == 2) ? 4 : 3;

    size_type shape[4];
    size_type start[4];
    size_type count[4];

    spMeshGetDomain(f->m, SP_DOMAIN_CENTER, start, count, NULL);

    for (int i = 0; i < 3; ++i)
    {
        shape[i] = spMeshGetShape(f->m)[i];
        count[i] -= start[i];
    }
    shape[3] = 3;
    start[3] = 0;
    count[3] = 3;

    spIOStreamWriteSimple(os, name, SP_TYPE_Real, f_host, ndims, shape, start, NULL, count, NULL, flag);

    spParallelHostFree(&f_host);
}

void spFieldRead(spField *f, spIOStream *os, char const name[], int flag)
{

}

void spFieldSync(spField *f)
{

}
