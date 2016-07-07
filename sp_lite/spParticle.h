/**
 * spParticle.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPPARTICLE_H_
#define SPPARTICLE_H_

#include "sp_lite_def.h"

#define SP_MAX_NUMBER_OF_PARTICLE_ATTR 32

struct spPage_s;
struct spMesh_s;
struct spParticleAttrEntity_s
{
    int type_tag;
    size_type size_in_byte;
    size_type offsetof;
    char name[255];
};

struct spParticle_s
{
    SP_OBJECT_HEAD
    struct spMesh_s const *m;

    Real mass;
    Real charge;

    size_type max_number_of_pages;
    size_type entity_size_in_byte;
    size_type page_size_in_byte;

    int number_of_attrs;
    struct spParticleAttrEntity_s attrs[SP_MAX_NUMBER_OF_PARTICLE_ATTR];

    void *data;
    struct spPage_s *m_free_page;
    struct spPage_s *m_pages_holder;
    struct spPage_s **buckets;

};

#define SP_PARTICLE_POINT_HEAD  int flag; Real rx;Real ry;Real rz;

struct spParticlePoint_s
{
    SP_PARTICLE_POINT_HEAD
    void __others[];
};


#define ADD_PARTICLE_ATTRIBUTE(_SP_, _S_, _T_, _N_) spParticleAddAttribute(_SP_, __STRING(_N_), SP_TYPE_##_T_, sizeof(_T_), offsetof(_S_,_N_));

#ifdef PARTICLE_IS_AOS
#   define P_GET(_PG_, _N_, _S_) _PG_[_S_]._N_
#   define P_GETp(_D_, _STRUCT_, _T_, _N_, _S_) (_STRUCT_*)(_D_+sizeof(_STRUCT_)*SP_NUMBER_OF_ENTITIES_IN_PAGE)->_N_
#else
#   define P_GET(_D_, _STRUCT_, _T_, _N_, _S_) *(_T_*)(_D_+offsetof(_STRUCT_,_N_)*SP_NUMBER_OF_ENTITIES_IN_PAGE+sizeof(_T_)*_S_)
#   define P_GET_FLAG(_D_, _S_) *(int*)(_D_+offsetof(struct spParticlePoint_s,flag)*SP_NUMBER_OF_ENTITIES_IN_PAGE+sizeof(int)*_S_)

#endif

typedef struct spParticle_s spParticle;

void spParticleCreate(const struct spMesh_s *ctx, struct spParticle_s **pg);

void spParticleDestroy(struct spParticle_s **sp);

struct spParticleAttrEntity_s *spParticleAddAttribute(struct spParticle_s *pg, char const *name, int type_tag,
                                                      size_type size_in_byte, size_type offsetof);

void spParticleDeploy(struct spParticle_s *sp, int PIC);

void spParticleWrite(spParticle const *f, spIOStream *os, const char url[], int flag);

void spParticleRead(struct spParticle_s *f, char const url[], int flag);

void spParticleSync(struct spParticle_s *f);

void spParticleInitialize(spParticle *sp);

#endif /* SPPARTICLE_H_ */
