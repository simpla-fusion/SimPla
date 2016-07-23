//
// Created by salmon on 16-7-23.
//

#ifndef SIMPLA_SPPAGE_CU_H
#define SIMPLA_SPPAGE_CU_H
#include "spPage.h"
#include </usr/local/cuda/include/cuda_runtime_api.h>
#include "../../../../../usr/local/cuda/include/device_launch_parameters.h"

#define THREAD_X (blockIdx.x * blockDim.x + threadIdx.x)
#define THREAD_Y (blockIdx.y * blockDim.y + threadIdx.y)
#define THREAD_Z (blockIdx.z * blockDim.z + threadIdx.z)

#define DIMS_X (blockDim.x * gridDim.x)
#define DIMS_Y (blockDim.y * gridDim.y)
#define DIMS_Z (blockDim.z * gridDim.z)

__device__ spPage *spPageAtomicPush(spPage **pg, spPage *v)
{
    spPage *old = (*pg);
    spPage **address_as_ull = pg;
    spPage *assumed, *next;
    assert(sizeof(unsigned long long int) == sizeof(struct spPapge_s *));
    do
    {
        if (old == NULL)
        {
            break;
        }

        assumed = old;

        next = (spPage *) (old->next);

        old = (spPage *) atomicCAS((unsigned long long int *) address_as_ull, (unsigned long long int) (assumed),
                                   (unsigned long long int) (next));
    }
    while (assumed != old);

    return old;

}
__device__ spPage *spPageAtomicPop(spPage **pg)
{
    spPage *old = (*pg);
    spPage **address_as_ull = pg;
    spPage *assumed, *next;
    assert(sizeof(unsigned long long int) == sizeof(struct spPapge_s *));
    do
    {
        if (old == NULL)
        {
            break;
        }

        assumed = old;

        next = (spPage *) (old->next);

        old = (spPage *) atomicCAS((unsigned long long int *) address_as_ull, (unsigned long long int) (assumed),
                                   (unsigned long long int) (next));
    }
    while (assumed != old);

    return old;

}

__device__ void spPageLinkResize(spPage **p, spPage **pool, size_type max)
{
    int count = 0;

    while (count < max)
    {
        if (*p == NULL) { *p = spPageAtomicPop(pool); }

        p = &((*p)->next);
        ++count;

    }

    while (*p != NULL)
    {
        spPage *t = *p;
        p = &((*p)->next);
        spPageAtomicPush(pool, t);
    }
}


#endif //SIMPLA_SPPAGE_CU_H
