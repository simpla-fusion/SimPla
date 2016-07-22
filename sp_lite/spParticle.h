/**
 * spParticle.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPPARTICLE_H_
#define SPPARTICLE_H_

#include "sp_lite_def.h"
#include "spMesh.h"
#include "spPage.h"


typedef size_type **spPageHaed;

struct spMesh_s;

struct spParticle_s;

typedef struct spParticle_s spParticle;


#define SP_PARTICLE_DATA_HEAD     MeshEntityId * id; Real * rx;Real* ry;Real* rz;

struct spParticleData_s
{
    SP_PARTICLE_DATA_HEAD
    void *attrs[];
};

void spParticleCreate(const struct spMesh_s *ctx, struct spParticle_s **pg);

void spParticleDestroy(struct spParticle_s **sp);

void spParticleDeploy(struct spParticle_s *sp, size_type PIC);

spMesh const *spParticleMesh(spParticle const *sp);

size_type **spParticleBuckets(spParticle *);

size_type **spParticlePagePool(spParticle *);

Real spParticleMass(spParticle const *);

Real spParticleCharge(spParticle const *);

#define SP_MAX_NUMBER_OF_PARTICLE_ATTR 16

#define ADD_PARTICLE_ATTRIBUTE(_SP_, _T_, _N_) spParticleAddAttribute(_SP_, __STRING(_N_), SP_TYPE_##_T_, sizeof(_T_),0ul-1);

void spParticleAddAttribute(struct spParticle_s *pg, char const *name, int type_tag,
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
