/**
 * sp_def.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SP_DEF_H_
#define SP_DEF_H_
#include <stdio.h>
#include "../sp_config.h"
#if !defined(__CUDA_ARCH__)
#define MC_HOST_DEVICE_PREFIX
#define MC_HOST_PREFIX
#define MC_DEVICE
#define MC_SHARED
#define MC_CONSTANT
#define MC_GLOBAL

#else

#define MC_HOST_DEVICE_PREFIX __host__ __device__
#define MC_HOST_PREFIX __host__
#define MC_DEVICE  __device__
#define MC_SHARED __shared__
#define MC_CONSTANT __constant__
#define MC_GLOBAL  __global__

#endif

struct spPage_s;
typedef struct spPage_s spPage;
typedef spPage * bucket_type;

struct spPagePool_s;
typedef struct spPagePool_s spPagePool;

struct spMesh_s
{

  Real inv_dx[3];
  size_type number_of_cell;
  size_type i_lower[3];
  size_type i_upper[3];
  size_type i_dims[3];
  size_type number_of_idx;
  size_type *cell_idx;

  dim3 numBlocks;

  dim3 threadsPerBlock;
};

struct spField_s
{
  Real * data;
};
struct spParticleSpecies_s
{
  Real mass;
  Real charge;
  size_type entity_size_in_byte;
  spPagePool * pool;
  bucket_type * buckets;

};

#endif /* SP_DEF_H_ */
