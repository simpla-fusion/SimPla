//
// Created by salmon on 16-9-25.
//
#include <stdio.h>
#include "spMisc.h"
#include "spDataType.h"
#include "spParallel.h"
#include "sp_lite_def.h"

int printArray(const void *d, int type_tag, int ndims, size_type const *dims)
{

    size_type size_in_byte = spDataTypeSizeInByte(type_tag);
    for (int i = 0; i < ndims; ++i)
    {
        size_in_byte *= dims[i];
    }
    void *buffer;
    SP_CALL(spMemHostAlloc(&buffer, size_in_byte));
    SP_CALL(spMemCopy(buffer, d, size_in_byte));


    printf("\n %4d|\t", 0);
    for (int j = 0; j < dims[1]; ++j) { printf(" %8d ", j); }
    printf("\n-----+--");
    for (int j = 0; j < dims[1] * 10; ++j) { printf("-"); }


    for (int i = 0; i < dims[0]; ++i)
    {
        printf("\n %4d|\t", i);
        for (int j = 0; j < dims[1]; ++j)
        {
            if (dims[2] > 1) { printf("{"); }
            for (int k = 0; k < dims[2]; ++k)
            {
                size_type s = (i * dims[1] + j) * dims[2] + k;

                if (type_tag == SP_TYPE_Real) { printf(" %8f ", ((Real *) buffer)[s]); }
                else if (type_tag == SP_TYPE_size_type) { printf(" %8lu ", ((size_type *) buffer)[s]); }
            }
            if (dims[2] > 1) { printf("},"); }
        }

    }


    printf("\n");
    SP_CALL(spMemHostFree(&buffer));
    return SP_SUCCESS;
};
