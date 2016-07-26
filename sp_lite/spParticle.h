/**
 * spParticle.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPPARTICLE_H_
#define SPPARTICLE_H_


#include "sp_lite_def.h"


#define SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE 256

#define SP_PARTICLE_HEAD                                \
     int   id[SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE];           \
     Real  rx[SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE];           \
     Real  ry[SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE];           \
     Real  rz[SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE];

#define SP_PARTICLE_ATTR(_T_, _N_)       _T_ _N_[SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE];


#define SP_PARTICLE_DATA_DESC_ADD(_DESC_, _CLS_, _T_, _N_)                           \
    spParticleAddAttribute(_DESC_, __STRING(_N_), SP_TYPE_##_T_,sizeof(_T_),offsetof(_CLS_,_N_));

#define SP_PARTICLE_CREATE_DATA_DESC(_DESC_, _CLS_)     \
    SP_PARTICLE_DATA_DESC_ADD(_DESC_,_CLS_,int ,id)   \
    SP_PARTICLE_DATA_DESC_ADD(_DESC_,_CLS_,Real ,rx)  \
    SP_PARTICLE_DATA_DESC_ADD(_DESC_,_CLS_,Real ,ry)  \
    SP_PARTICLE_DATA_DESC_ADD(_DESC_,_CLS_,Real ,rz)


struct spDataType_s;
struct spIOStream_s;

struct spMesh_s;
struct spParticle_s;

typedef struct spParticle_s spParticle;

int spParticleCreate(spParticle **sp, struct spMesh_s const *m);

int spParticleDestroy(spParticle **sp);

int spParticleDeploy(spParticle *sp);

Real spParticleMass(spParticle const *);

Real spParticleCharge(spParticle const *);

int spParticleAddAttribute(spParticle *sp, char const name[], int tag, size_type size, size_type offset);

int spParticlePIC(spParticle *sp, size_type s);

size_type spParticleMaxFiberLength(const spParticle *sp);

void **spParticleData(spParticle *sp);

void const **spParticleDataConst(spParticle const *sp);

int spParticleWrite(spParticle const *sp, struct spIOStream_s *os, const char *url, int flag);

int spParticleRead(spParticle *sp, struct spIOStream_s *os, const char *url, int flag);

int spParticleSync(spParticle *sp);


#endif /* SPPARTICLE_H_ */
