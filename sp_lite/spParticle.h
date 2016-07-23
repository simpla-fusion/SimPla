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


struct spMesh_s;

struct spParticle_s;

typedef struct spParticle_s spParticle;

#ifndef SP_MAX_NUMBER_OF_PARTICLE_ATTR
#    define SP_MAX_NUMBER_OF_PARTICLE_ATTR 16
#endif

#define SP_PARTICLE_HEAD                                \
     MeshEntityId  flag[]; \
     Real  rx[];           \
     Real  ry[];           \
     Real  rz[];

#define SP_PARTICLE_ATTR(_T_, _N_)       _T_ _N_[];

#define SP_PARTICLE_ADD_ATTR_HEAD(_SP_, _CLS_)  \
     spParticleAddAttribute(_SP_, "flag", SP_TYPE_int64_t, sizeof(int64_t));  \
     spParticleAddAttribute(_SP_, "rx", SP_TYPE_Real, sizeof(Real));            \
     spParticleAddAttribute(_SP_, "ry", SP_TYPE_Real, sizeof(Real));            \
     spParticleAddAttribute(_SP_, "rz", SP_TYPE_Real, sizeof(Real));

#define SP_PARTICLE_ADD_ATTR(_SP_, _CLS_, _T_, _N_)  \
     spParticleAddAttribute(_SP_, __STRING(_N_), SP_TYPE_##_T_, sizeof(_T_));  \



int spParticleCreate(const spMesh *ctx, spParticle **pg);

int spParticleDestroy(spParticle **sp);

int spParticleDeploy(spParticle *sp, size_type PIC);

int spParticleResizePageLink(spParticle *sp);

spMesh const *spParticleMesh(spParticle const *sp);

void **spParticleDataRoot(spParticle *sp);

spPage **spParticleBaseField(spParticle *sp);

spPage **spParticlePagePool(spParticle *sp);

int *spParticlePageCount(spParticle *sp);

size_type spParticleCountPageNum(spParticle *sp, int domain_tag, size_type *displs);

Real spParticleMass(spParticle const *);

Real spParticleCharge(spParticle const *);

size_type spParticleNumOfEntitiesInPage(spParticle const *);

#define ADD_PARTICLE_ATTRIBUTE(_SP_, _T_, _N_) spParticleAddAttribute(_SP_, __STRING(_N_), SP_TYPE_##_T_, sizeof(_T_),0ul-1);

int spParticleAddAttribute(struct spParticle_s *pg, char const *name, int type_tag, size_type size_in_byte);

int spParticleNumberOfAttributes(struct spParticle_s const *sp);

int spParticleWrite(spParticle const *f, spIOStream *os, const char *url, int flag);

void spParticleRead(struct spParticle_s *f, spIOStream *os, char const url[], int flag);

void spParticleSync(struct spParticle_s *f);

int spParticleResizePageLink(spParticle *sp);

int
spParticleGetPageOffset(spParticle *sp,
                        size_type const lower[3],
                        size_type const upper[3],
                        size_type *num_of_page,
                        MeshEntityId **page_id,
                        size_type **data_displs);
#endif /* SPPARTICLE_H_ */
