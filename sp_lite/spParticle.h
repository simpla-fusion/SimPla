/**
 * spParticle.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPPARTICLE_H_
#define SPPARTICLE_H_


#include "sp_lite_def.h"


#define SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE 128

#define SP_PARTICLE_HEAD                                \
     int   *id;           \
     Real  *rx;           \
     Real  *ry;           \
     Real  *rz;

#define SP_PARTICLE_ATTR(_T_, _N_)       _T_ * _N_;


#define SP_PARTICLE_DATA_DESC_ADD(_DESC_, _CLS_, _T_, _N_)                           \
    spParticleAddAttribute(_DESC_, __STRING(_N_), SP_TYPE_##_T_,sizeof(_T_),-1);

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

int spParticleInitialize(spParticle *sp, int const *dist_types);

int spParticleSetPIC(spParticle *sp, size_type pic, size_type max_pic);

size_type spParticleGetPIC(spParticle const *sp);

size_type spParticleGetPIC(spParticle const *sp);

int spParticleSetMass(spParticle *, Real m);

int spParticleSetCharge(spParticle *, Real e);

Real spParticleGetMass(spParticle const *);

Real spParticleGetCharge(spParticle const *);

int spParticleAddAttribute(spParticle *sp, char const name[], int tag, size_type size, size_type offset);

int spParticleGetNumberOfAttributes(spParticle const *sp);

int spParticleGetAttributeName(spParticle *sp, int i, char *);

size_type spParticleGetAttributeTypeSizeInByte(spParticle *sp, int i);

void *spParticleGetAttributeData(spParticle *sp, int i);

int spParticleGetAllAttributeData(spParticle *sp, void **res);

int spParticleGetAllAttributeData_device(spParticle *sp, void ***current_data, void ***next_data);

size_type spParticleGetNumberOfEntities(spParticle const *sp);

size_type spParticleGetMaxPIC(const spParticle *sp);

int spParticleWrite(spParticle const *sp, struct spIOStream_s *os, const char *url, int flag);

int spParticleRead(spParticle *sp, struct spIOStream_s *os, const char *url, int flag);

int spParticleUpdate(spParticle *sp);

int spParticleSync(spParticle *sp);


#endif /* SPPARTICLE_H_ */
