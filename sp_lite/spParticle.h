/**
 * spParticle.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPPARTICLE_H_
#define SPPARTICLE_H_


#include "sp_config.h"

#ifndef SP_MAX_NUMBER_OF_PARTICLE_ATTR
#    define SP_MAX_NUMBER_OF_PARTICLE_ATTR 16
#endif

#define SP_PARTICLE_HEAD  \
     size_type  *id;           \
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
    SP_PARTICLE_DATA_DESC_ADD(_DESC_,_CLS_,size_type ,id)   \
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

/**  constructure and destructure @{*/
int spParticleCreate(spParticle **sp, struct spMesh_s const *m);

int spParticleDestroy(spParticle **sp);

int spParticleDeploy(spParticle *sp);

int spParticleInitialize(spParticle *sp, int const *dist_types);

/**    @}*/
/**  meta data @{*/

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

/**    @}*/
/**  ID @{*/

int spParticleSort(spParticle *sp);

int spParticleBuildBucket(spParticle *sp);


int spParticleResetHash(spParticle *sp);

int spParticleGetBucketIndex(spParticle *sp, size_type **start_pos, size_type **end_pos, size_type **index);

int spParticleRearrange(spParticle *sp);

int spParticleSync(spParticle *sp);


/**    @}*/
/**  attribute @{*/
int spParticleAddAttribute(spParticle *sp, char const name[], int tag, size_type size, size_type offset);

int spParticleGetNumberOfAttributes(spParticle const *sp);

int spParticleGetAttributeName(spParticle *sp, int i, char *);

size_type spParticleGetAttributeTypeSizeInByte(spParticle *sp, int i);

void *spParticleGetAttributeData(spParticle *sp, int i);

int spParticleSetAttributeData(spParticle *sp, int i, void *data);

int spParticleGetAllAttributeData(spParticle *sp, void **res);

int spParticleGetAllAttributeData_device(spParticle *sp, void ***data);
/** @}*/

int spParticleWrite(spParticle const *sp, struct spIOStream_s *os, const char *url, int flag);

int spParticleRead(spParticle *sp, struct spIOStream_s *os, const char *url, int flag);


#endif /* SPPARTICLE_H_ */
