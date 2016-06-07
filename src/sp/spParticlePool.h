//
// Created by salmon on 16-6-7.
//

#ifndef SIMPLA_SPPARTICLEPOOL_H
#define SIMPLA_SPPARTICLEPOOL_H
#ifdef __cplusplus
extern "C" {
#endif
#include "SmallObjPool.h"

struct spParticlePool
{
    spPagePool *pool;
    spPage **pages;
};


#ifdef __cplusplus
}
#endif
#endif //SIMPLA_SPPARTICLEPOOL_H
