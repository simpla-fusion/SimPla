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
    SP_TYPE_unsigned_int,
    SP_TYPE_long,
    SP_TYPE_unsigned_long,
    SP_TYPE_OPAQUE,
    SP_TYPE_CUSTOM = 0xFF
};
#ifdef REAL_IS_FLOAT
#   define SP_TYPE_Real SP_TYPE_float
#else
#   define SP_TYPE_Real SP_TYPE_double
#endif
#define SP_TYPE_MeshEntityId SP_TYPE_int64_t
#define SP_TYPE_size_type  SP_TYPE_unsigned_long

size_type spDataTypeSizeInByte(int type_tag);


#endif //SIMPLA_SPDATATYPE_H
