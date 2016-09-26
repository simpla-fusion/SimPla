//
// Created by salmon on 16-7-20.
//

#include "sp_lite_def.h"

#include <assert.h>

#include "spObject.h"
#include "spParallel.h"
#include "spDataType.h"
#include "spIOStream.h"

#include "spMesh.h"
#include "spField.h"
#include "spAlogorithm.h"
#include "spMisc.h"

#define MAX_NUM_OF_FIELD_ATTR 16

typedef struct spField_s
{
    SP_MESH_ATTR_HEAD

    int m_data_type_tag_;

    void *m_data_;

    int is_soa;


} spField;

int spFieldCreate(spField **f, const struct spMesh_s *mesh, int iform, int type_tag)
{

    SP_CALL(spMeshAttributeCreate((spMeshAttribute **) f, sizeof(spField), mesh, iform));

    (*f)->m = mesh;
    (*f)->iform = (uint) iform;
    (*f)->is_soa = SP_TRUE;
    (*f)->m_data_ = NULL;
    (*f)->m_data_type_tag_ = type_tag;
    return SP_SUCCESS;
}

int spFieldDestroy(spField **f)
{

    if (f != NULL && *f != NULL) {SP_CALL(spMemoryDeviceFree(&((**f).m_data_))); }

    SP_CALL(spMeshAttributeDestroy((spMeshAttribute **) f));

    return SP_SUCCESS;
}

int spFieldDeploy(spField *f)
{
    if (f->m_data_ == NULL) {SP_CALL(spMemoryDeviceAlloc(&(f->m_data_), spFieldGetSizeInByte(f))); }

    return SP_SUCCESS;
}

size_type spFieldGetSizeInByte(spField const *f)
{
    return spDataTypeSizeInByte(f->m_data_type_tag_) *
           spMeshGetNumberOfEntities(f->m, SP_DOMAIN_ALL, f->iform);
}

int spFieldAddScalar(spField *, void const *);

int spFieldIsSoA(spField const *f) { return f->is_soa; }

int spFieldDataType(spField const *f) { return f->m_data_type_tag_; };

void *spFieldData(spField *f) { return spFieldDeviceData(f); }

void *spFieldDeviceData(spField *f) { return f->m_data_; }

const void *spFieldDataConst(spField const *f) { return spFieldDeviceDataConst(f); }

const void *spFieldDeviceDataConst(spField const *f) { return f->m_data_; }

int spFieldNumberOfSub(spField const *f)
{
    int iform = spMeshAttributeGetForm((spMeshAttribute const *) f);

    return (iform == VERTEX || iform == VOLUME) ? 1 : 3;
}

int spFieldAddScalar(spField *f, void const *v)
{
    return SP_DO_NOTHING;
}


int spFieldSubArray(spField *f, void **data)
{
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
    return SP_SUCCESS;
};

int spFieldClear(spField *f)
{
    SP_CALL(spFieldDeploy(f));

    SP_CALL(spMemSet(f->m_data_, 0, spFieldGetSizeInByte(f)));

    return SP_SUCCESS;
}

int spFieldFillReal(spField *f, Real v)
{
    SP_CALL(spFieldDeploy(f));

    SP_CALL(spParallelDeviceFillReal(f->m_data_, v, spMeshGetNumberOfEntities(f->m, SP_DOMAIN_ALL, f->iform)));
    return SP_SUCCESS;

}

int spFieldShow(const spField *f, char const *name)
{
    if (f == NULL) { return SP_FAILED; }

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) f);

    size_type dims[3];

    spMeshGetDims(m, dims);

    int num_of_sub = spFieldNumberOfSub(f);

    void *d[num_of_sub];

    SP_CALL(spFieldSubArray((spField *) f, d));

    if (name != NULL) { printf("\n [ %s ]", name); }

    for (int i = 0; i < num_of_sub; ++i)
    {
        printf("\n [ %d/%d ]", i, num_of_sub);
        SP_CALL(printArray(d[i], f->m_data_type_tag_, 3, dims));
    }



//    void *buffer;
//    SP_CALL(spMemHostAlloc((void **) &buffer, spFieldGetSizeInByte(f)));
//    SP_CALL(spMemoryCopy(buffer, spFieldData((spField *) f), spFieldGetSizeInByte(f)));
//
//    if (name != NULL) { printf("\n [ %s ]", name); }
//
//    printf("\n %4d|\t", 0);
//    for (int j = 0; j < dims[1]; ++j) { printf(" %8d ", j); }
//    printf("\n-----+--");
//    for (int j = 0; j < dims[1] * 10; ++j) { printf("-"); }
//
//
//    for (int i = 0; i < dims[0]; ++i)
//    {
//        printf("\n %4d|\t", i);
//        for (int j = 0; j < dims[1]; ++j)
//        {
//            if (dims[2] > 1) { printf("{"); }
//            for (int k = 0; k < dims[2]; ++k)
//            {
//                size_type s = i * strides[0] + j * strides[1] + k * strides[2];
//
//                if (f->m_data_type_tag_ == SP_TYPE_Real) { printf(" %8.2f ", ((Real *) buffer)[s]); }
//                else if (f->m_data_type_tag_ == SP_TYPE_size_type) { printf(" %8lu ", ((size_type *) buffer)[s]); }
//            }
//            if (dims[2] > 1) { printf("},"); }
//        }
//
//    }
//
//
//    printf("\n");
//    SP_CALL(spMemHostFree(&buffer));
    return SP_SUCCESS;
}

int spFieldWrite(spField *f, spIOStream *os, char const name[], int flag)
{


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

    size_type num_of_sub = (size_type) spFieldNumberOfSub(f);
    SP_CALL(spMeshGetGlobalArrayShape(m, SP_DOMAIN_CENTER,
                                      (iform == VERTEX || iform == VOLUME) ? 0 : 1,
                                      &num_of_sub, &array_ndims, &mesh_start_dim,
                                      g_dims, g_start, l_dims, l_start, l_count, spFieldIsSoA(f)));

    SP_CALL(spIOStreamWriteSimple(os, name, spFieldDataType(f),
                                  f_host, array_ndims, l_dims,
                                  l_start, NULL, l_count, NULL,
                                  g_dims, g_start, flag));


    SP_CALL(spMemHostFree(&f_host));


    return SP_SUCCESS;
}

int spFieldRead(spField *f, spIOStream *os, char const name[], int flag)
{
    UNIMPLEMENTED;
    return SP_SUCCESS;
}

int spFieldSync(spField *f)
{
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) f);

    int num_of_sub = spFieldNumberOfSub(f);

    void *d[num_of_sub];

    SP_CALL(spFieldSubArray(f, (void **) d));

    spMPIUpdater *updater;

    SP_CALL(spMeshGetMPIUpdater(m, &updater));

    for (int i = 0; i < num_of_sub; ++i)
    {
        SP_CALL(spMPIUpdateHalo(updater, f->m_data_type_tag_, d[i]));
    }


    return SP_SUCCESS;
}

int spFeildAssign(spField *f, size_type num_of_points, size_type *offset, Real const **v)
{
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) f);

    if (spFieldIsSoA(f))
    {
        int num_of_sub = spFieldNumberOfSub(f);

        Real *data[num_of_sub];

        SP_CALL(spFieldSubArray(f, (void **) data));

        for (int i = 0; i < num_of_sub; ++i) {SP_CALL(spParallelAssign(num_of_points, offset, data[i], v[i])); }
    } else
    {
        UNIMPLEMENTED;
    }
    return SP_SUCCESS;
}

int spFieldCopyToHost(void **d, spField const *f)
{
    size_type s = spFieldGetSizeInByte(f);
    SP_CALL(spMemHostAlloc(d, s));
    SP_CALL(spMemoryCopy(*d, f->m_data_, s));
    return SP_SUCCESS;
};

int spFieldCopyToDevice(spField *f, void const *d)
{
    SP_CALL(spMemoryCopy(f->m_data_, d, spFieldGetSizeInByte(f)));
    return SP_SUCCESS;
}


int spFieldTestSync(spMesh const *m, int type_tag)
{
    spField *f;

    SP_CALL(spFieldCreate(&f, m, VERTEX, type_tag));

    SP_CALL(spFieldClear(f));

    size_type *data;

    SP_CALL(spFieldSubArray(f, (void **) &data));

    SP_CALL(spFieldFillSeq(f, SP_DOMAIN_CENTER));

    if (spMPIRank() == 0) {SHOW_FIELD(f); }

    SP_CALL(spFieldSync(f));

    if (spMPIRank() == 0) {SHOW_FIELD(f); }

    return SP_SUCCESS;
};
