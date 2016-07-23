/**
 * spParticle.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPPARTICLE_H_
#define SPPARTICLE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "sp_lite_def.h"
#include "spMesh.h"
#include "spPage.h"

#ifdef __cplusplus
}
#endif

typedef size_type **spPageHaed;

struct spMesh_s;

struct spParticle_s;

typedef struct spParticle_s spParticle;

#define SP_PARTICLE_HEAD                                \
     SP_PAGE_HEAD(struct spParticlePage_s)              \
     MeshEntityId  flag[SP_NUMBER_OF_ENTITIES_IN_PAGE]; \
     Real  rx[SP_NUMBER_OF_ENTITIES_IN_PAGE];           \
     Real  ry[SP_NUMBER_OF_ENTITIES_IN_PAGE];           \
     Real  rz[SP_NUMBER_OF_ENTITIES_IN_PAGE];

#define SP_PARTICLE_ATTR(_T_, _N_)       _T_ _N_[SP_NUMBER_OF_ENTITIES_IN_PAGE];

#define SP_PARTICLE_ATTR_HEAD(_SP_, _CLS_)  \
     spParticleAddAttribute(_SP_, "flag", SP_TYPE_int64_t, sizeof(int64_t), offsetof(_CLS_, flag));  \
     spParticleAddAttribute(_SP_, "rx", SP_TYPE_Real, sizeof(Real), offsetof(_CLS_, rx));            \
     spParticleAddAttribute(_SP_, "ry", SP_TYPE_Real, sizeof(Real), offsetof(_CLS_, ry));            \
     spParticleAddAttribute(_SP_, "rz", SP_TYPE_Real, sizeof(Real), offsetof(_CLS_, rz));

#define SP_PARTICLE_ADD_ATTR(_SP_, _CLS_, _T_, _N_)  \
     spParticleAddAttribute(_SP_, __STRING(_N_), SP_TYPE_##_T_, sizeof(_T_), offsetof(_CLS_, _N_));  \


typedef struct spParticlePage_s
{
    SP_PARTICLE_HEAD
    byte_type __others[];
} spParticlePage;

int spParticleCreate(spParticle **pg, const spMesh *ctx);

int spParticleDestroy(spParticle **sp);

int spParticleDeploy(spParticle *sp, size_type PIC);

int spParticlePushPageToField(spParticlePage **b,
                              spParticlePage **pool,
                              size_type const *shape,
                              size_type const *lower,
                              size_type const *upper,
                              size_type const *num_of_page,
                              size_type default_num_page);

spMesh const *spParticleMesh(spParticle const *sp);

size_type **spParticleBuckets(spParticle *);

size_type **spParticlePagePool(spParticle *);

Real spParticleMass(spParticle const *);

Real spParticleCharge(spParticle const *);

#define SP_MAX_NUMBER_OF_PARTICLE_ATTR 16

#define ADD_PARTICLE_ATTRIBUTE(_SP_, _T_, _N_) spParticleAddAttribute(_SP_, __STRING(_N_), SP_TYPE_##_T_, sizeof(_T_),0ul-1);

int spParticleAddAttribute(struct spParticle_s *pg, char const *name, int type_tag,
                           size_type size_in_byte, size_type offset);

void *spParticleAttributeData(struct spParticle_s *pg, int i);

void **spParticleAttributeDeviceData(struct spParticle_s *pg);

int spParticleGetibuteTypeTag(struct spParticle_s *pg, int i);

size_type spParticleAttibuteSizeInByte(struct spParticle_s *pg, int i);

void spParticleAttributeName(struct spParticle_s *pg, int i, char *name);

void spParticleWrite(spParticle const *f, spIOStream *os, const char url[], int flag);

void spParticleRead(struct spParticle_s *f, spIOStream *os, char const url[], int flag);

void spParticleSync(struct spParticle_s *f);

void spParticleSyncStart(struct spParticle_s *f);

void spParticleSyncEnd(struct spParticle_s *f);


#endif /* SPPARTICLE_H_ */
