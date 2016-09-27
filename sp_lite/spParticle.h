/**
 * spParticle.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPPARTICLE_H_
#define SPPARTICLE_H_


#include "sp_lite_config.h"
#include "spField.h"

#ifndef SP_MAX_NUMBER_OF_PARTICLE_ATTR
#    define SP_MAX_NUMBER_OF_PARTICLE_ATTR 16
#endif

#define SP_PARTICLE_HEAD  \
     Real  *rx;           \
     Real  *ry;           \
     Real  *rz;


typedef struct
{
    SP_PARTICLE_HEAD
    Real *attrs[];

} particle_head;

#define SP_PARTICLE_DEFAULT_NUM_OF_PIC 256
#define SP_PARTICLE_ATTR(_N_)  Real * _N_;

#define SP_PARTICLE_ATTR_ADD(_P_, _N_)   SP_CALL(spParticleAddAttribute(_P_, __STRING(_N_), SP_TYPE_Real));

#define SP_PARTICLE_ATTR_HEAD(_P_)     \
    SP_PARTICLE_ATTR_ADD(_P_,rx)  \
    SP_PARTICLE_ATTR_ADD(_P_,ry)  \
    SP_PARTICLE_ATTR_ADD(_P_,rz)
struct spDataType_s;

struct spMesh_s;

struct spParticle_s;

typedef struct spParticle_s spParticle;

/**  constructure and destructure @{*/
int spParticleCreate(spParticle **sp, struct spMesh_s const *m);

int spParticleDestroy(spParticle **sp);

int spParticleDeploy(spParticle *sp);

int spParticleInitialize(spParticle *sp, int const *dist_types);

int spParticleNextStep(spParticle *sp);

int spParticleEnableSorting(spParticle *sp);

int spParticleNeedSorting(spParticle const *sp);

int spParticleCoordinateLocalToGlobal(spParticle *sp);

int spParticleCoordinateGlobalToLocal(spParticle *sp);
/**    @}*/
/**  meta-data @{*/

int spParticleSetPIC(spParticle *sp, unsigned int pic);

uint spParticleGetPIC(spParticle const *sp);

int spParticleSetMass(spParticle *, Real m);

Real spParticleGetMass(spParticle const *);

int spParticleSetCharge(spParticle *, Real e);

Real spParticleGetCharge(spParticle const *);

size_type spParticleSize(spParticle const *);

size_type spParticleCapacity(spParticle const *);

int spParticleResize(spParticle *, size_type);

size_type spParticleGlobalSize(spParticle const *sp);

size_type spParticleLocalSizeInDomain(spParticle const *sp, int domain_tag);

/**    @}*/

/**  ID @{*/


int spParticleSetDefragmentFreq(spParticle *sp, size_t n);

int spParticleSort(spParticle *sp);

int spParticleSync(spParticle *sp);

int spParticleDefragment(spParticle *sp);

int spParticleGetBucket(spParticle *sp, size_type **start_pos, size_type **count, size_type **sorted_idx,
                        size_type **cell_hash);

int spParticleGetBucket2(spParticle *sp, spField **start_pos, spField **count,
                         size_type **sorted_idx, size_type **cell_hash);

int spParticleCollectIndex(spParticle const *sp, size_type const cell_start[3], size_type const cell_count[3],
                           size_type **index, size_type *mem_size);


/**    @}*/

/**  attribute @{*/
int spParticleAddAttribute(spParticle *sp, const char name[], int type_tag);

int spParticleGetNumberOfAttributes(spParticle const *sp);

int spParticleGetAttributeName(spParticle *sp, int i, char *);

size_type spParticleGetAttributeTypeSizeInByte(spParticle *sp, int i);

void *spParticleGetAttributeData(spParticle *sp, int i);

int spParticleSetAttributeData(spParticle *sp, int i, void *data);

int spParticleGetAllAttributeData(spParticle *sp, void **res);

int spParticleGetAllAttributeData_device(spParticle *sp, void ***current_data, void ***next_data);

/** @}*/

struct spIOStream_s;

int spParticleWrite(const spParticle *sp, struct spIOStream_s *os, const char *url, int flag);

int spParticleRead(spParticle *sp, struct spIOStream_s *os, const char *url, int flag);

int spParticleDiagnose(spParticle const *sp, struct spIOStream_s *os, char const *path, int flag);


#endif /* SPPARTICLE_H_ */
