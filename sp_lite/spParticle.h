/**
 * spParticle.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPPARTICLE_H_
#define SPPARTICLE_H_


#include "sp_config.h"


#define SP_PARTICLE_HEAD  \
     uint  *id;           \
     Real  *rx;           \
     Real  *ry;           \
     Real  *rz;


typedef struct
{
    SP_PARTICLE_HEAD
    Real *attrs[];

} particle_head;


#define SP_PARTICLE_DATA_DESC_ADD(_DESC_, _CLS_, _T_, _N_)                           \
    spParticleAddAttribute(_DESC_, __STRING(_N_), SP_TYPE_##_T_,sizeof(_T_),-1);

#define SP_PARTICLE_CREATE_DATA_DESC(_DESC_, _CLS_)     \
    SP_PARTICLE_DATA_DESC_ADD(_DESC_,_CLS_,uint ,id)   \
    SP_PARTICLE_DATA_DESC_ADD(_DESC_,_CLS_,Real ,rx)  \
    SP_PARTICLE_DATA_DESC_ADD(_DESC_,_CLS_,Real ,ry)  \
    SP_PARTICLE_DATA_DESC_ADD(_DESC_,_CLS_,Real ,rz)

#define SP_PARTICLE_ATTR(_N_)  Real * _N_;

#define SP_PARTICLE_ADD_ATTR(_DESC_, _N_)   spParticleAddAttribute(_DESC_, __STRING(_N_), SP_TYPE_Real,sizeof(Real),-1);


struct spDataType_s;

struct spIOStream_s;

struct spMesh_s;

struct spParticle_s;

typedef struct spParticle_s spParticle;

int spParticleCreate(spParticle **sp, struct spMesh_s const *m);

int spParticleDestroy(spParticle **sp);

int spParticleDeploy(spParticle *sp);

int spParticleInitialize(spParticle *sp, int const *dist_types);

int spParticleIsSorted(spParticle const *sp);

int spParticleSetPIC(spParticle *sp, unsigned int pic);

uint spParticleGetPIC(spParticle const *sp);

size_type spParticleGetNumOfParticle(const spParticle *sp);

size_type spParticleGetMaxNumOfParticle(const spParticle *sp);

int spParticleSetMass(spParticle *, Real m);

Real spParticleGetMass(spParticle const *);

int spParticleSetCharge(spParticle *, Real e);

Real spParticleGetCharge(spParticle const *);

size_type spParticleGetSize(spParticle const *);

size_type spParticleGetCapacity(spParticle const *);

size_type spParticlePush(spParticle *sp, size_type s);

//const uint *spParticleGetStartPos(spParticle const *);
//
//const uint *spParticleGetEndPos(spParticle const *);
//
//const uint *spParticleGetSortedIndex(spParticle const *);

int spParticleGetIndexArray(spParticle *sp, uint **start_pos, uint **end_pos, uint **index);

int spParticleAddAttribute(spParticle *sp, char const name[], int tag, size_type size, size_type offset);

int spParticleGetNumberOfAttributes(spParticle const *sp);

int spParticleGetAttributeName(spParticle *sp, int i, char *);

size_type spParticleGetAttributeTypeSizeInByte(spParticle *sp, int i);

void *spParticleGetAttributeData(spParticle *sp, int i);

int spParticleSetAttributeData(spParticle *sp, int i, void *data);

int spParticleGetAllAttributeData(spParticle *sp, void **res);

int spParticleGetAllAttributeData_device(spParticle *sp, void ***data);

int spParticleWrite(spParticle const *sp, struct spIOStream_s *os, const char *url, int flag);

int spParticleRead(spParticle *sp, struct spIOStream_s *os, const char *url, int flag);

int spParticleSort(spParticle *sp);

int spParticleDeepSort(spParticle *sp);

int spParticleSync(spParticle *sp);


#endif /* SPPARTICLE_H_ */
