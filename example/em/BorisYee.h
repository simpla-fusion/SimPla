//
// Created by salmon on 16-6-11.
//

#ifndef SIMPLA_BORISYEE_H
#define SIMPLA_BORISYEE_H

#include "../../src/sp_cuda_config.h"
#include "../../src/particle/ParticleInterface.h"

#ifdef __cplusplus
extern "C" {
#endif


#define CACHE_EXTENT_X 4
#define CACHE_EXTENT_Y 4
#define CACHE_EXTENT_Z 4
#define CACHE_SIZE (CACHE_EXTENT_X*CACHE_EXTENT_Y*CACHE_EXTENT_Z)


struct spPage;

void spBorisYeePush(struct spPage *pg, Real cmr, double dt, const Real *E, const Real *B, size_type const *i_self_,
                    size_type const *i_lower_, size_type const *i_upper_, const Real *inv_dx);

void spBorisYeeIntegralRho(struct spPage *pg, Real *cf, size_type iform, size_type sub_index);


void spBorisYeeIntegralJ(struct spPage *pg, Real *cJ, size_type iform, size_type sub_index);


void spBorisYeeIntegralE(struct spPage *pg, Real *cE, size_type iform, size_type sub_index);

#ifdef __cplusplus
};
#endif


#endif //SIMPLA_BORISYEE_H
