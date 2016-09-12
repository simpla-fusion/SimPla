//
// Created by salmon on 16-7-28.
//
#include <stdlib.h>
#include <string.h>
#include "../../spParallel.h"
#include "spParallelOPENMP.h"

#include <omp.h>

int spParallelDeviceInitialize(int argc, char **argv)
{
#ifndef NDEBUG
    omp_set_num_threads(1);
#endif
    return SP_SUCCESS;
}

int spParallelDeviceFinalize() { return SP_SUCCESS; }

int spParallelDeviceAlloc(void **p, int s)
{
    *p = malloc(s);
    return SP_SUCCESS;
}

int spParallelDeviceFree(void **p)
{
    if (*p != NULL)
    {
        free(*p);
        *p = NULL;
    }
    return SP_SUCCESS;
};

int spParallelMemcpy(void *dest, void const *src, int s)
{
    memcpy(dest, src, s);
    return SP_SUCCESS;
}

int spParallelMemcpyToCache(const void *dest, void const *src, int s)
{
    memcpy((void *) dest, src, s);
    return SP_SUCCESS;
}

int spParallelMemset(void *dest, int v, int s)
{
    memset(dest, v, s);
    return SP_SUCCESS;
}

int spParallelDeviceSync()
{
    SP_CALL(spParallelGlobalBarrier());
    return SP_SUCCESS;
}

int spParallelHostAlloc(void **p, int s)
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

int spParallelDeviceFillInt(int *d, int v, int s)
{
    UNIMPLEMENTED;
    return SP_SUCCESS;
};


int spParallelDeviceFillReal(Real *d, Real v, int s)
{
    UNIMPLEMENTED;
    return SP_SUCCESS;
};


int spParallelAssign(int num_of_point, int *offset, Real *d, Real const *v)
{
    UNIMPLEMENTED;
    return SP_SUCCESS;
};

int spMemoryDeviceToHost(void **p, void *src, int size_in_byte)
{
    *p = src;
    return SP_SUCCESS;
}
int spMemoryHostFree(void **p)
{
    *p = NULL;
    return SP_SUCCESS;
};

