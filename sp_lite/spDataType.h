//
// Created by salmon on 16-9-12.
//

#ifndef SIMPLA_SPDATATYPE_H
#define SIMPLA_SPDATATYPE_H


enum
{
    SP_TYPE_NULL,
    SP_TYPE_float,
    SP_TYPE_double,
    SP_TYPE_int,
    SP_TYPE_uint,
    SP_TYPE_long,
    SP_TYPE_int64_t,
    SP_TYPE_OPAQUE
};
#ifdef REAL_IS_FLOAT
#   define SP_TYPE_Real SP_TYPE_float
#else
#   define SP_TYPE_Real SP_TYPE_double
#endif
#define SP_TYPE_MeshEntityId SP_TYPE_int64_t


struct spDataType_s;

typedef struct spDataType_s spDataType;

int spDataTypeCreate(spDataType **, int type_tag, int s);

int spDataTypeDestroy(spDataType **);

int spDataTypeCopy(spDataType *, spDataType const *);

int spDataTypeSizeInByte(spDataType const *dtype);

void spDataTypeSetSizeInByte(spDataType *dtype, int s);

int spDataTypeIsValid(spDataType const *);

int spDataTypeExtent(spDataType *, int rank, const int *d);

int spDataTypeAdd(spDataType *dtype, int offset, char const *name, spDataType const *other);

int spDataTypeAddArray(spDataType *dtype,
                       int offset,
                       char const *name,
                       int type_tag,
                       int n,
                       int const *dims);

#endif //SIMPLA_SPDATATYPE_H
