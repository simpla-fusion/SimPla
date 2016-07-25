//
// Created by salmon on 16-7-20.
//


#include "sp_lite_def.h"
#include <stdlib.h>
#include "spObject.h"
#include "spParallel.h"
#include "spMesh.h"
#include "spField.h"
#include "spIO.h"

typedef struct spField_s
{
    SP_OBJECT_HEAD

    int iform;

    const struct spMesh_s *m;

    struct spDataType_s *m_data_type_desc_;

    Real *device_data;
    Real *host_data;
} spField;

int spFieldCreate(spField **f, const struct spMesh_s *mesh, int iform)
{
    *f = (spField *) malloc(sizeof(spField));
    (*f)->m = mesh;
    (*f)->iform = iform;

    (*f)->host_data = NULL;
    (*f)->device_data = NULL;

    spDataTypeCreate(&((*f)->m_data_type_desc_), SP_TYPE_Real, 0);

    return SP_SUCCESS;
}

int spFieldDestroy(spField **f)
{
    if (f != NULL && *f != NULL)
    {
        spParallelDeviceFree((void **) (&((**f).device_data)));
        spParallelHostFree((void **) (&((**f).host_data)));
        spDataTypeDestroy(&((*f)->m_data_type_desc_));
        free(*f);
        *f = NULL;
    }
    return SP_SUCCESS;
}

int spFieldDeploy(spField *f)
{
    if (f->device_data == NULL)
    {
        size_type num_of_entities = spMeshNumberOfEntity(f->m, SP_DOMAIN_ALL, f->iform);
        size_type s = spDataTypeSizeInByte(f->m_data_type_desc_);
        spParallelDeviceAlloc((void **) &(f->device_data), num_of_entities * s);
    }
    return SP_SUCCESS;
}

size_type spFieldId(spField const *f) { return f->id; };

spMesh const *spFieldMesh(spField const *f) { return f->m; }

int spFieldForm(spField const *f) { return f->iform; };

spDataType const *spFieldDataType(spField const *f) { return f->m_data_type_desc_; };

void *spFieldData(spField *f) { return spFieldDeviceData(f); }

void *spFieldDeviceData(spField *f) { return f->device_data; }

void *spFieldHostData(spField *f) { return f->host_data; }

void const *spFieldDataConst(spField const *f) { return spFieldDeviceDataConst(f); }

void const *spFieldDeviceDataConst(spField const *f) { return f->device_data; }

void const *spFieldHostDataConst(spField const *f) { return f->host_data; }

int spFieldClear(spField *f)
{
    spFieldDeploy(f);

    size_type num_of_entities = spMeshNumberOfEntity(f->m, SP_DOMAIN_ALL, f->iform);

    spParallelMemset(f->device_data, 0, num_of_entities * sizeof(Real));
    return SP_SUCCESS;
}
int spFieldFill(spField *f, Real v)
{
    spFieldDeploy(f);

    size_type num_of_entities = spMeshNumberOfEntity(f->m, SP_DOMAIN_ALL, f->iform);

    int iv = *(int *) (&v);
    CHECK_INT(iv);
    spParallelMemset(f->device_data, iv, num_of_entities * sizeof(Real));

    return SP_SUCCESS;
}
int spFieldWrite(spField *f, spIOStream *os, char const name[], int flag)
{
    size_type size_in_byte = spMeshNumberOfEntity(f->m, SP_DOMAIN_ALL, f->iform) * sizeof(Real);

    void *f_host;

    spParallelHostAlloc(&f_host, size_in_byte);

    spParallelMemcpy((f_host), (void *) (f->device_data), size_in_byte);

    int ndims = spMeshNDims(spFieldMesh(f));

    size_type l_dims[ndims + 1];
    size_type l_start[ndims + 1];
    size_type l_count[ndims + 1];

    size_type g_dims[ndims + 1];
    size_type g_start[ndims + 1];


    spMeshGlobalDomain(f->m, g_dims, g_start);
    spMeshDomain(f->m, SP_DOMAIN_CENTER, l_dims, l_start, l_count);
    l_dims[ndims] = 3;
    l_start[ndims] = 0;
    l_count[ndims] = 3;

    g_dims[ndims] = 3;
    g_start[ndims] = 0;

    spIOStreamWriteSimple(os,
                          name,
                          spFieldDataType(f),
                          f_host,
                          (f->iform == 1 || f->iform == 2) ? ndims + 1 : ndims,
                          l_dims, l_start, NULL, l_count, NULL,
                          g_dims, g_start, flag);

    spParallelHostFree(&f_host);
    return SP_SUCCESS;
}

int spFieldRead(spField *f, spIOStream *os, char const name[], int flag)
{
    return SP_SUCCESS;
}

int spFieldSync(spField *f)
{

    size_type start[4];
    size_type count[4];
    size_type dims[4];

    int ndims = (f->iform == 1 || f->iform == 2) ? 4 : 3;

    spMeshDomain(f->m, SP_DOMAIN_CENTER, dims, start, count);

    start[3] = 0;
    count[3] = 3;
    dims[3] = 3;

    spParallelUpdateNdArrayHalo(f->device_data, f->m_data_type_desc_, ndims, dims, start, NULL, count, NULL);

    return SP_SUCCESS;

}
