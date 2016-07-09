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
        if ((**f).device_data != NULL)
        {
            spParallelDeviceFree((void **) ((**f).device_data));
        };

        if ((**f).host_data != NULL)
        {
            free((void **) ((**f).host_data));
        }
        *f = NULL;
    }
}

void spFieldDeploy(spField *f)
{
    if (f->device_data == NULL)
    {
        size_type num_of_entities = spMeshGetNumberOfEntity(f->m, f->iform);
        spParallelDeviceMalloc((void **) &(f->device_data), num_of_entities * sizeof(Real));
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

    void *f_host = malloc(size_in_byte);

    spParallelMemcpy((f_host), (void *) (f->device_data), size_in_byte);

    int ndims = (f->iform == 1 || f->iform == 2) ? 4 : 3;

    size_type shape[4];
    size_type start[4];
    size_type count[4];

    shape[0] = f->m->dims.x;
    shape[1] = f->m->dims.y;
    shape[2] = f->m->dims.z;
    shape[3] = 3;
    start[0] = f->m->i_lower.x;
    start[1] = f->m->i_lower.y;
    start[2] = f->m->i_lower.z;
    start[3] = 0;
    count[0] = f->m->i_upper.x - f->m->i_lower.x;
    count[1] = f->m->i_upper.y - f->m->i_lower.y;
    count[2] = f->m->i_upper.z - f->m->i_lower.z;
    count[3] = 3;

//    spIOStreamWriteSimple(os, name, f_host, ndims, shape, start, count, flag);

    free(f_host);

}

void spFieldRead(spField *f, spIOStream *os, char const name[], int flag)
{

}

void spFieldSync(spField *f)
{

}
