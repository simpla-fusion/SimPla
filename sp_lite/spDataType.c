//
// Created by salmon on 16-9-19.
//
#include "sp_lite_config.h"
#include "spDataType.h"

size_type spDataTypeSizeInByte(int type_tag)
{
    size_type res = 0;
    if (type_tag != SP_TYPE_NULL)
    {

        switch (type_tag)
        {
            case SP_TYPE_float:
                res = sizeof(float);
                break;
            case SP_TYPE_double:
                res = sizeof(double);
                break;

            case SP_TYPE_int:
                res = sizeof(int);
                break;
            case SP_TYPE_uint:
                res = sizeof(unsigned int);
                break;
            case SP_TYPE_long:
                res = sizeof(long);
                break;
//            case SP_TYPE_int64_t:
//                res = sizeof(int64_t);
//                break;
            case SP_TYPE_size_type:
                res = sizeof(size_type);
                break;
            default:
                UNIMPLEMENTED;
                break;
        }
    }

    return res;
}