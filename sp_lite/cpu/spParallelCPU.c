//
// Created by salmon on 16-7-28.
//
#include <stdlib.h>
#include <string.h>
#include "../spParallel.h"


int spParallelDeviceInitialize(int argc, char **argv)
{

}

int spParallelDeviceFinalize()
{

}

int spParallelDeviceAlloc(void **p, size_type _S_)
{
    *p = malloc(_S_);
    return SP_SUCCESS;
}

int spParallelDeviceFree(void **_P_)
{
    if (*_P_ != NULL)
    {
        (free(*_P_));
        *_P_ = NULL;
    };
    return SP_SUCCESS;
}

int spParallelMemcpy(void **dest, void const **src, size_type s)
{
    memcpy(dest, src, (s));
    return SP_SUCCESS;
}

int spParallelMemcpyToSymbol(void **dest, void const **src, size_type s)
{
    memcpy(dest, src, (s));
    return SP_SUCCESS;
}

int spParallelMemset(void *dest, int v, size_type s)
{
    memset(dest, v, s);
    return SP_SUCCESS;
}

int spParallelDeviceSync()
{
    SP_CALL(spParallelGlobalBarrier());
    return SP_SUCCESS;
}

int spParallelHostAlloc(void **p, size_type s)
{
    *p = malloc(s);
    return SP_SUCCESS;
};

int spParallelHostFree(void **p)
{
    if (*p != NULL)
    {
        free(*p);
        *p = NULL;
    }

    return SP_SUCCESS;
}