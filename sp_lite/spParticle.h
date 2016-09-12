/**
 * spParticle.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPPARTICLE_H_
#define SPPARTICLE_H_


#include "sp_lite_def.h"
#include "spDataType.h"

#define SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE 128

#define SP_PARTICLE_HEAD  \
     uint  *id;           \
     Real  *rx;           \
     Real  *ry;           \
     Real  *rz;


typedef struct { SP_PARTICLE_HEAD } particle_head;

#define SP_PARTICLE_ATTR(_T_, _N_)       _T_ * _N_;


#define SP_PARTICLE_DATA_DESC_ADD(_DESC_, _CLS_, _T_, _N_)                           \
    spParticleAddAttribute(_DESC_, __STRING(_N_), SP_TYPE_##_T_,sizeof(_T_),-1);

#define SP_PARTICLE_CREATE_DATA_DESC(_DESC_, _CLS_)     \
    SP_PARTICLE_DATA_DESC_ADD(_DESC_,_CLS_,uint ,id)   \
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

int spParticleIsSorted(spParticle const *sp);

int spParticleSetPIC(spParticle *sp, unsigned int pic);

uint spParticleGetPIC(spParticle const *sp);

int spParticleGetNumOfParticle(const spParticle *sp);

int spParticleGetMaxNumOfParticle(const spParticle *sp);

int spParticleSetMass(spParticle *, Real m);

Real spParticleGetMass(spParticle const *);

int spParticleSetCharge(spParticle *, Real e);

Real spParticleGetCharge(spParticle const *);

int spParticleGetSize(spParticle const *);

int spParticleGetCapacity(spParticle const *);

int spParticlePush(spParticle *sp, int s);

const uint *spParticleGetStartPos(spParticle const *);

const uint *spParticleGetEndPos(spParticle const *);

const uint *spParticleGetSortedIndex(spParticle const *);

int spParticleGetIndexArray(spParticle *sp, uint **start_pos, uint **end_pos, uint **index);

int spParticleAddAttribute(spParticle *sp, char const name[], int tag, int size, int offset);

int spParticleGetNumberOfAttributes(spParticle const *sp);

int spParticleGetAttributeName(spParticle *sp, int i, char *);

int spParticleGetAttributeTypeSizeInByte(spParticle *sp, int i);

void *spParticleGetAttributeData(spParticle *sp, int i);

int spParticleSetAttributeData(spParticle *sp, int i, void *data);

int spParticleGetAllAttributeData(spParticle *sp, void **res);

int spParticleGetAllAttributeData_device(spParticle *sp, void ***data);

int spParticleWrite(spParticle const *sp, struct spIOStream_s *os, const char *url, int flag);

int spParticleRead(spParticle *sp, struct spIOStream_s *os, const char *url, int flag);

int spParticleSort(spParticle *sp);

int spParticleSync(spParticle *sp);


#endif /* SPPARTICLE_H_ */
