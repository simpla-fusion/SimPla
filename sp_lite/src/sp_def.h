/**
 * sp_def.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SP_DEF_H_
#define SP_DEF_H_
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#define  AUTHOR " YU Zhi <yuzhi@ipp.ac.cn> "
#define  COPYRIGHT "All rights reserved. (2016 )"

typedef int8_t byte_type; // int8_t
typedef double Real;
typedef long Integral;
typedef int64_t id_type;
typedef int64_t index_type;
typedef uint64_t size_type;

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

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#define CUDA_CHECK(_CMD_)  											\
		fprintf(stderr, "[line %d in file %s]\n %s = %u \n",					\
				 __LINE__, __FILE__,__STRING(_CMD_),(_CMD_));		\

inline bool sp_is_device_ptr(void const *p)
{
	cudaPointerAttributes attribute;
	CUDA_CHECK_RETURN(cudaPointerGetAttributes(&attribute, p));
	return (attribute.device == cudaMemoryTypeDevice);

}

#endif /* SP_DEF_H_ */
