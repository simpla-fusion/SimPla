/**
 * sp_def.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SP_DEF_H_
#define SP_DEF_H_
#include <stdio.h>
#include "sp_config.h"

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
#ifndef __CUDA_ARCH__
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
#endif

#define CUDA_CHECK(_CMD_)  											\
		fprintf(stderr, "[line %d in file %s]\n %s = %d \n",					\
				 __LINE__, __FILE__,__STRING(_CMD_),(_CMD_));		\

inline int sp_is_device_ptr(void const *p)
{
	cudaPointerAttributes attribute;
	CUDA_CHECK_RETURN(cudaPointerGetAttributes(&attribute, p));
	return (attribute.device == cudaMemoryTypeDevice) ? 1 : 0;

}

inline void sp_memcpy(void *dest, const void *src, size_type s)
{
	CUDA_CHECK_RETURN(cudaMemcpy(dest, src, s, cudaMemcpyDefault));
}

typedef struct spObject_s
{
	struct spObject_s * self;
	size_type size_in_byte;
	byte_type __others[];
} spObject;
#define SP_OBJECT_HEAD	void * self;

struct spPage_s;
typedef struct spPage_s spPage;
typedef spPage bucket_type;

struct spPagePool_s;
typedef struct spPagePool_s spPagePool;

MC_HOST_DEVICE void spCreateObject(spObject ** obj, size_type size_in_byte)
{
	if (sp_is_device_ptr(*obj))
	{
		CUDA_CHECK_RETURN(cudaMalloc(obj, size_in_byte));
		(*obj)->self = 0x0;
		(*obj)->size_in_byte = size_in_byte;
	}
#if !defined(__CUDA_ARCH__)
	else
	{
		*obj=(spObject *)malloc(size_in_byte);
		(*obj)->size_in_byte = size_in_byte;
		(*obj)->self=0x0;
	}
#endif
}
void spDestroyObject(spObject ** obj)
{
	if (obj != 0x0 && sp_is_device_ptr(*obj))
	{
		CUDA_CHECK_RETURN(cudaFree(*obj));
		*obj = 0x0;
	}
	else
	{
		spDestroyObject(&((*obj)->self));
		free(*obj);
		*obj = 0x0;
	}
}
MC_HOST void spObjectHostToDevice(spObject * obj)
{
	if (obj->self == 0x0)
	{
		cudaMalloc(&(obj->self), obj->size_in_byte);
	}
	cudaMemcpy(obj->self, obj, obj->size_in_byte, cudaMemcpyDefault);
}
MC_HOST void spObjectDeviceToHost(spObject * obj)
{
	if (obj != 0x0 && obj->self != 0x0)
	{
		spObject * tmp = obj->self;
		cudaMemcpy(obj, obj->self, obj->size_in_byte, cudaMemcpyDefault);
		obj->self = tmp;
	}
}
#endif /* SP_DEF_H_ */
