//
// Created by salmon on 16-7-20.
//

#include "sp_lite_def.h"

#include <assert.h>

#include "spObject.h"
#include "spParallel.h"
#include "spMPI.h"
#include "spDataType.h"
#include "spIOStream.h"

#include "spMesh.h"
#include "spField.h"

#define MAX_NUM_OF_FIELD_ATTR 16

typedef struct spField_s
{
    SP_MESH_ATTR_HEAD

    struct spDataType_s *m_data_type_desc_;

    void *m_data_;

    int is_soa;

} spField;

int spFieldCreate(spField **f, const struct spMesh_s *mesh, int iform, int type_tag)
{
    int error_code = SP_SUCCESS;
    SP_CALL(spMeshAttributeCreate((spMeshAttribute **) f, sizeof(spField), mesh, iform));

    (*f)->m = mesh;
    (*f)->iform = (uint) iform;
    (*f)->is_soa = SP_TRUE;
    (*f)->m_data_ = NULL;


    SP_CALL(spDataTypeCreate(&((*f)->m_data_type_desc_), type_tag, 0));

    return error_code;
}

int spFieldDestroy(spField **f)
{
    int error_code = SP_SUCCESS;

    if (f != NULL && *f != NULL)
    {
        SP_CALL(spParallelDeviceFree(&((**f).m_data_)));

        SP_CALL(spDataTypeDestroy(&((*f)->m_data_type_desc_)));
    }

    SP_CALL(spMeshAttributeDestroy((spMeshAttribute **) f));

    return error_code;
}

int spFieldDeploy(spField *f)
{
    int error_code = SP_SUCCESS;

    if (f->m_data_ == NULL)
    {
        SP_CALL(spParallelDeviceAlloc((void **) &(f->m_data_), spFieldGetSizeInByte(f)));
    }
    return error_code;
}

size_type spFieldGetSizeInByte(spField const *f)
{
    return spDataTypeSizeInByte(f->m_data_type_desc_) *
           spMeshGetNumberOfEntities(f->m, SP_DOMAIN_ALL, f->iform);
}

int spFieldAdd(spField *, void const *);

int spFieldIsSoA(spField const *f) { return f->is_soa; }

spDataType const *spFieldDataType(spField const *f) { return f->m_data_type_desc_; };

void *spFieldData(spField *f) { return spFieldDeviceData(f); }

void *spFieldDeviceData(spField *f) { return f->m_data_; }

const void *spFieldDataConst(spField const *f) { return spFieldDeviceDataConst(f); }

const void *spFieldDeviceDataConst(spField const *f) { return f->m_data_; }

int spFieldNumberOfSub(spField const *f)
{
    int iform = spMeshAttributeGetForm((spMeshAttribute const *) f);

    return (iform == VERTEX || iform == VOLUME) ? 1 : 3;
}

int spFieldAdd(spField *f, void const *v)
{
    return SP_DO_NOTHING;
}

int spFieldSubArray(spField *f, void **data)
{
    int error_code = SP_SUCCESS;

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) f);

    size_type ele_size_in_byte = spDataTypeSizeInByte(spFieldDataType(f));

    void *data_root = spFieldData(f);

    int num_of_sub = spFieldNumberOfSub(f);

    if (num_of_sub == 1) { *data = spFieldData(f); }
    else if (spFieldIsSoA(f))
    {
        size_type offset = ele_size_in_byte * spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL, VERTEX);

        for (int i = 0; i < num_of_sub; ++i) { data[i] = data_root + i * offset; }

    } else
    {
        UNIMPLEMENTED;
//        for (int i = 0; i < num_of_sub; ++i) { data[i] = data_root + i * ele_size_in_byte; }
    }
    return error_code;
};

int spFieldClear(spField *f)
{
    int error_code = SP_SUCCESS;
    SP_CALL(spFieldDeploy(f));

    SP_CALL(spParallelMemset(f->m_data_, 0, spFieldGetSizeInByte(f)));

    return error_code;
}

int spFieldFill(spField *f, Real v)
{
    int error_code = SP_SUCCESS;

    SP_CALL(spFieldDeploy(f));

    SP_CALL(spParallelDeviceFillReal(f->m_data_, v, spMeshGetNumberOfEntities(f->m, SP_DOMAIN_ALL, f->iform)));
    return error_code;

}

int spFieldWrite(spField *f, spIOStream *os, char const name[], int flag)
{
    int error_code = SP_SUCCESS;

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) f);
    int iform = spMeshAttributeGetForm((spMeshAttribute const *) f);

    size_type size_in_byte = spFieldGetSizeInByte(f);

    void *f_host;

    SP_CALL(spFieldCopyToHost(&f_host, f));

    int ndims = spMeshGetNDims(m);
    int array_ndims, mesh_start_dim;

    size_type l_dims[ndims + 1];
    size_type l_start[ndims + 1];
    size_type l_count[ndims + 1];

    size_type g_dims[ndims + 1];
    size_type g_start[ndims + 1];

    size_type num_of_sub = 3;
    SP_CALL(spMeshGetGlobalArrayShape(m, SP_DOMAIN_CENTER,
                                      (iform == VERTEX || iform == VOLUME) ? 0 : 1,
                                      &num_of_sub, &array_ndims, &mesh_start_dim,
                                      g_dims, g_start, l_dims, l_start, l_count, spFieldIsSoA(f)));

    SP_CALL(spIOStreamWriteSimple(os, name, spFieldDataType(f),
                                  f_host, array_ndims, l_dims,
                                  l_start, NULL, l_count, NULL,
                                  g_dims, g_start, flag));


    SP_CALL(spParallelHostFree(&f_host));


    return error_code;
}

int spFieldRead(spField *f, spIOStream *os, char const name[], int flag)
{
    UNIMPLEMENTED;
    return SP_SUCCESS;
}

int spFieldSync(spField *f)
{
    int error_code = SP_SUCCESS;

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) f);
    int iform = spMeshAttributeGetForm((spMeshAttribute const *) f);
    int ndims = spMeshGetNDims(m);
    int array_ndims, mesh_start_dim;

    size_type l_dims[ndims + 1];
    size_type l_start[ndims + 1];
    size_type l_count[ndims + 1];

    int num_of_sub = spFieldNumberOfSub(f);

    void *F[num_of_sub];

    spMPICartUpdater *updater;


    SP_CALL(spMeshGetDomain(m, SP_DOMAIN_CENTER, l_start, NULL, l_count));
    SP_CALL(spFieldSubArray(f, (void **) F));
    SP_CALL(spMPICartUpdaterCreate(&updater,
                                   spMPIComm(),
                                   spFieldDataType(f),
                                   0,
                                   ndims,
                                   l_dims,
                                   l_start,
                                   NULL,
                                   l_count,
                                   NULL,
                                   NULL,
                                   NULL,
                                   NULL));
    SP_CALL(spMPICartUpdateAll(updater, num_of_sub, F));
    SP_CALL(spMPICartUpdaterDestroy(&updater));
    return error_code;
}

int spFeildAssign(spField *f, size_type num_of_points, size_type *offset, Real const **v)
{
    int error_code = SP_SUCCESS;

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) f);

    if (spFieldIsSoA(f))
    {
        int num_of_sub = spFieldNumberOfSub(f);

        Real *data[num_of_sub];

        SP_CALL(spFieldSubArray(f, (void **) data));

        for (int i = 0; i < num_of_sub; ++i) { SP_CALL(spParallelAssign(num_of_points, offset, data[i], v[i])); }
    } else
    {
        UNIMPLEMENTED;
    }
    return error_code;
}

int spFieldCopyToHost(void **d, spField const *f)
{
    int error_code = SP_SUCCESS;

    size_type s = spFieldGetSizeInByte(f);

    SP_CALL(spParallelHostAlloc(d, s));
    SP_CALL(spParallelMemcpy(*d, f->m_data_, s));
    return error_code;
};

int spFieldCopyToDevice(spField *f, void const *d)
{
    int error_code = SP_SUCCESS;

    SP_CALL(spParallelMemcpy(f->m_data_, d, spFieldGetSizeInByte(f)));
    return error_code;
}