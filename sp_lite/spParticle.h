/**
 * spParticle.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPPARTICLE_H_
#define SPPARTICLE_H_


#include "sp_lite_def.h"


#define SP_NUMBER_OF_ENTITIES_IN_PAGE 128

#define SP_PARTICLE_HEAD                                \
     int   id[SP_NUMBER_OF_ENTITIES_IN_PAGE];           \
     Real  rx[SP_NUMBER_OF_ENTITIES_IN_PAGE];           \
     Real  ry[SP_NUMBER_OF_ENTITIES_IN_PAGE];           \
     Real  rz[SP_NUMBER_OF_ENTITIES_IN_PAGE];

#define SP_PARTICLE_ATTR(_T_, _N_)       _T_ _N_[SP_NUMBER_OF_ENTITIES_IN_PAGE];


#define SP_PARTICLE_CREATE_DATA_DESC_ADD(_DESC_, _CLS_, _T_, _N_)                           \
    spDataTypeAddArray(_DESC_,offsetof(_CLS_,_N_), __STRING(_N_), SP_TYPE_##_T_,SP_NUMBER_OF_ENTITIES_IN_PAGE,NULL );

#define SP_PARTICLE_CREATE_DATA_DESC(_DESC_, _CLS_)     \
    spDataType *data_desc;                                   \
    spDataTypeCreate(&data_desc, SP_TYPE_NULL);              \
    spDataTypeSetSizeInByte(_DESC_, sizeof(_CLS_));          \
    SP_PARTICLE_CREATE_DATA_DESC_ADD(_DESC_,_CLS_,int ,id)   \
    SP_PARTICLE_CREATE_DATA_DESC_ADD(_DESC_,_CLS_,Real ,rx)  \
    SP_PARTICLE_CREATE_DATA_DESC_ADD(_DESC_,_CLS_,Real ,ry)  \
    SP_PARTICLE_CREATE_DATA_DESC_ADD(_DESC_,_CLS_,Real ,rz)


struct spDataType_s;
struct spIOStream_s;

struct spMesh_s;
struct spParticle_s;

typedef struct spParticle_s spParticle;

int spParticleCreate(spParticle **sp, struct spMesh_s const *m, struct spDataType_s const *);

int spParticleDestroy(spParticle **sp);

int spParticleDeploy(spParticle *sp, size_type PIC);

Real spParticleMass(spParticle const *);

Real spParticleCharge(spParticle const *);

struct spDataType_s const *spParticleDataTypeDesc(spParticle const *sp);

struct spMesh_s const *spParticleMesh(spParticle const *sp);

void *spParticleData(spParticle *sp);

void const *spParticleDataConst(spParticle *sp);

size_type spParticleFiberLength(spParticle const *);

int spParticleWrite(spParticle const *sp, struct spIOStream_s *os, const char *url, int flag);

int spParticleRead(spParticle *sp, struct spIOStream_s *os, const char *url, int flag);

int spParticleSync(spParticle *sp);


#endif /* SPPARTICLE_H_ */
