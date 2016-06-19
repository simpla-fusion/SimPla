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
typedef float Real;
typedef int Integral;
typedef int64_t id_type;
typedef int64_t index_type;
typedef uint64_t size_type;
enum
{
	SP_NEW = 1UL << 1, SP_APPEND = 1UL << 2, SP_BUFFER = (1UL << 3), SP_RECORD = (1UL << 4)
};

enum
{
	SP_INT, SP_LONG, SP_DOUBLE, SP_FLOAT, SP_OPAQUE
};
#ifdef __CUDACC__
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

#define CUDA_CHECK_RETURN(_CMD_) {											\
	cudaError_t _m_cudaStat = _CMD_;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
#if !defined(__CUDA_ARCH__)
#define CUDA_CHECK(_CMD_)  											\
		fprintf(stderr, "[line %d in file %s]\n %s = %d \n",					\
				 __LINE__, __FILE__,__STRING(_CMD_),(_CMD_));
#else
#define CUDA_CHECK(_CMD_)
#endif
#define DONE	 	fprintf(stderr, "====== DONE ======\n" );
#define CHECK	 	fprintf(stderr, "[ line %d in file%s]====== CHECK ======\n", __LINE__, __FILE__ );

inline bool sp_is_device_ptr(void const *p)
{
	cudaPointerAttributes attribute;
	CUDA_CHECK_RETURN(cudaPointerGetAttributes(&attribute, p));
	return (attribute.device == cudaMemoryTypeDevice);

}
inline int sp_pointer_type(void const *p)
{
	cudaPointerAttributes attribute;
	CUDA_CHECK_RETURN(cudaPointerGetAttributes(&attribute, p));
	return (attribute.device);

}
#endif //__CUDACC__
#endif /* SP_DEF_H_ */
