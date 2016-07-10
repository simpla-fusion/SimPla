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

#define SP_PARTICLE_POINT_HEAD  MeshEntityId flag; Real rx;Real ry;Real rz;

typedef struct spParticlePoint_s
{
	SP_PARTICLE_POINT_HEAD
	byte_type __others[];
} spParticlePoint;

typedef struct spParticlePage_s
{
	SP_PAGE_HEAD(struct spParticlePage_s)
	MeshEntityId id;
	byte_type data[];
} spParticlePage;

struct spParticleAttrEntity_s
{
	int type_tag;
	size_type size_in_byte;
	size_type offsetof;
	char name[255];
};

struct spParticle_s
{
	struct spMesh_s const *m;
	int iform;
	Real mass;
	Real charge;
	int num_of_attrs;
	struct spParticleAttrEntity_s attrs[SP_MAX_NUMBER_OF_PARTICLE_ATTR];

	size_type number_of_pages;
	size_type entity_size_in_byte;
	size_type page_size_in_byte;

	struct spParticlePage_s *m_pages_;
	struct spParticlePage_s **m_page_pool_; //DEVICE
	struct spParticlePage_s **m_buckets_;
};

#define SP_MP_SUCCESS SP_SUCCESS
#define SP_MP_ERROR_POOL_IS_OVERFLOW 0xF000|0x1
#define SP_MP_FINISHED 0xFFFF

MC_DEVICE extern int spParticleMapAndPack(spParticlePage **dest, spParticlePage **src, int *d_tail, int *g_d_tail,
		int *s_tail, int *g_s_tail, spParticlePage **pool, MeshEntityId tag);

#define ADD_PARTICLE_ATTRIBUTE(_SP_, _S_, _T_, _N_) spParticleAddAttribute(_SP_, __STRING(_N_), SP_TYPE_##_T_, sizeof(_T_), offsetof(_S_,_N_));

#ifdef PARTICLE_IS_AOS
#   define P_GET(_PG_, _N_, _S_) _PG_[_S_]._N_
#   define P_GETp(_D_, _STRUCT_, _T_, _N_, _S_) (_STRUCT_*)((byte_type*) (_D_)+sizeof(_STRUCT_)*SP_NUMBER_OF_ENTITIES_IN_PAGE)->_N_
#else
#   define P_GET(_D_, _STRUCT_, _T_, _N_, _S_) *(_T_*)((byte_type*) (_D_)+offsetof(_STRUCT_,_N_)*SP_NUMBER_OF_ENTITIES_IN_PAGE+sizeof(_T_)*_S_)
#   define P_GET_FLAG(_D_, _S_) (*(MeshEntityId*)((byte_type*) (_D_)+offsetof(struct spParticlePoint_s,flag)*SP_NUMBER_OF_ENTITIES_IN_PAGE+sizeof(MeshEntityId)*_S_))

#endif

typedef struct spParticle_s spParticle;

void spParticleCreate(const struct spMesh_s *ctx, struct spParticle_s **pg);

void spParticleDestroy(struct spParticle_s **sp);

struct spParticleAttrEntity_s *spParticleAddAttribute(struct spParticle_s *pg, char const *name, int type_tag,
		size_type size_in_byte, int offsetof);

void *spParticleGetAttribute(spParticle *sp, char const *name);

void spParticleDeploy(struct spParticle_s *sp, int PIC);

void spParticleWrite(spParticle const *f, spIOStream *os, const char url[], int flag);

void spParticleRead(struct spParticle_s *f, spIOStream *os, char const url[], int flag);

void spParticleSync(struct spParticle_s *f);

void spParticleInitialize(spParticle *sp);

#endif /* SPPARTICLE_H_ */
