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
#define MC_HOST_DEVICE
#define MC_HOST
#define MC_DEVICE
#define MC_SHARED
#define MC_CONSTANT static const
#define MC_GLOBAL

#else

#define MC_HOST_DEVICE __host__ __device__
#define MC_HOST __host__
#define MC_DEVICE  __device__
#define MC_SHARED __shared__
#define MC_CONSTANT __constant__
#define MC_GLOBAL  __global__

#endif

struct spPage_s;
typedef struct spPage_s spPage;
typedef spPage bucket_type;

struct spPagePool_s;
typedef struct spPagePool_s spPagePool;



struct spField_s
{
	int iform;
	Real * data;
};


#endif /* SP_DEF_H_ */
