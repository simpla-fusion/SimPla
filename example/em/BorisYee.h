//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_BORISYEE_H
#define SIMPLA_BORISYEE_H

#include "../../src/particle/ParticleInterface.h"

#ifdef __cplusplus
extern "C" {
#endif


#define CACHE_EXTENT_X 4
#define CACHE_EXTENT_Y 4
#define CACHE_EXTENT_Z 4
#define CACHE_SIZE (CACHE_EXTENT_X*CACHE_EXTENT_Y*CACHE_EXTENT_Z)


struct spPage;


spBorisYeeIntegralRho(struct spPage *pg, Real tf[CACHE_SIZE], size_type iform);


void spBorisYeeIntegralRho(struct spPage *pg, Real *cf, size_type iform, size_type sub_index);


void spBorisYeeIntegralJ(struct spPage *pg, Real *cJ, size_type iform, size_type sub_index);


void spBorisYeeIntegralE(struct spPage *pg, Real *cE, size_type iform, size_type sub_index);

#ifdef __cplusplus
};
#endif


#endif //SIMPLA_BORISYEE_H
