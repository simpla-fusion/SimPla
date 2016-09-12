//
// Created by salmon on 16-9-12.
//

#ifndef SIMPLA_SPDATATYPE_H
#define SIMPLA_SPDATATYPE_H
#include "sp_config.h"
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

int spDataTypeCreate(spDataType **, int type_tag, size_type s);

int spDataTypeDestroy(spDataType **);

int spDataTypeCopy(spDataType *, spDataType const *);

size_type spDataTypeSizeInByte(spDataType const *dtype);

void spDataTypeSetSizeInByte(spDataType *dtype, size_type s);

int spDataTypeIsValid(spDataType const *);

int spDataTypeExtent(spDataType *, int rank, const size_type *d);

int spDataTypeAdd(spDataType *dtype, size_type offset, char const *name, spDataType const *other);

int spDataTypeAddArray(spDataType *dtype,
                       size_type offset,
                       char const *name,
                       int type_tag,
                       size_type n,
                       size_type const *dims);


#endif //SIMPLA_SPDATATYPE_H
