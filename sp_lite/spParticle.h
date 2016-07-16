/**
 * spParticle.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPPARTICLE_H_
#define SPPARTICLE_H_

#include "sp_lite_def.h"
#include "spPage.h"
#include "spMesh.h"

#define SP_MAX_NUMBER_OF_PARTICLE_ATTR 32

struct spPage_s;
struct spMesh_s;


#define SP_PARTICLE_DATA_HEAD     MeshEntityId * id; Real * rx;Real* ry;Real* rz;

struct spParticleData_s;

struct spParticle_s;

typedef struct spParticlePage_s
{
    SP_PAGE_HEAD(struct spParticlePage_s)
    MeshEntityId id;
    size_type offset;
} spParticlePage;


#define ADD_PARTICLE_ATTRIBUTE(_SP_, _T_, _N_) spParticleAddAttribute(_SP_, __STRING(_N_), SP_TYPE_##_T_, sizeof(_T_),int(-1));


typedef struct spParticle_s spParticle;

void spParticleCreate(const struct spMesh_s *ctx, struct spParticle_s **pg);

void spParticleInitialize(spParticle *sp);

void spParticleDestroy(struct spParticle_s **sp);

void spParticleAddAttribute(struct spParticle_s *pg, char const *name, int type_tag,
                            int size_in_byte, int offset);

void spParticleDeploy(struct spParticle_s *sp, size_type PIC);

spParticlePage **spParticleBuckets(spParticle *);

spParticlePage **spParticlePagePool(spParticle *);

spMesh const *spParticleMesh(spParticle const *sp);

void *spParticleAttributeData(struct spParticle_s *pg, int i);

void **spParticleAttributeDeviceData(struct spParticle_s *pg);

int spParticleGetibuteTypeTag(struct spParticle_s *pg, int i);

int spParticleAttibuteSizeInByte(struct spParticle_s *pg, int i);

void spParticleAttributeName(struct spParticle_s *pg, int i, char *name);

void spParticleWrite(spParticle const *f, spIOStream *os, const char url[], int flag);

void spParticleRead(struct spParticle_s *f, spIOStream *os, char const url[], int flag);

void spParticleSync(struct spParticle_s *f);

void spParticleStart(struct spParticle_s *f);

void spParticleEnd(struct spParticle_s *f);


#endif /* SPPARTICLE_H_ */
