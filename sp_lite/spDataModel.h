//
// Created by salmon on 16-7-24.
//

#ifndef SIMPLA_SPDATAMODEL_H
#define SIMPLA_SPDATAMODEL_H

#include "sp_lite_def.h"
#include "../src/sp_capi.h"


#define SP_PARTICLE_HEAD                                \
     int   id[SP_NUMBER_OF_ENTITIES_IN_PAGE];           \
     Real  rx[SP_NUMBER_OF_ENTITIES_IN_PAGE];           \
     Real  ry[SP_NUMBER_OF_ENTITIES_IN_PAGE];           \
     Real  rz[SP_NUMBER_OF_ENTITIES_IN_PAGE];

#define SP_PARTICLE_ATTR(_T_, _N_)       _T_ _N_[SP_NUMBER_OF_ENTITIES_IN_PAGE];

#define SP_PARTICLE_ADD_ATTR_HEAD(_SP_, _CLS_)  \
     spParticleAddAttribute(_SP_, "id", SP_TYPE_int, sizeof(int),offsetof(_CLS_,id));  \
     spParticleAddAttribute(_SP_, "rx", SP_TYPE_Real, sizeof(Real),offsetof(_CLS_,rx));            \
     spParticleAddAttribute(_SP_, "ry", SP_TYPE_Real, sizeof(Real),offsetof(_CLS_,ry));            \
     spParticleAddAttribute(_SP_, "rz", SP_TYPE_Real, sizeof(Real),offsetof(_CLS_,rz));

#define SP_PARTICLE_ADD_ATTR(_SP_, _CLS_, _T_, _N_)  \
     spParticleAddAttribute(_SP_, __STRING(_N_), SP_TYPE_##_T_, sizeof(_T_),offsetof(_CLS_,_N_));

#define ADD_PARTICLE_ATTRIBUTE(_SP_, _T_, _N_) spParticleAddAttribute(_SP_, __STRING(_N_), SP_TYPE_##_T_, sizeof(_T_),0ul-1);

//int spParticleAddAttribute(struct spParticle_s *pg, char const *name, int type_tag, size_type size_in_byte,
//                           size_type offset);
//int spParticleNumberOfAttributes(struct spParticle_s const *sp);

#endif //SIMPLA_SPDATAMODEL_H
