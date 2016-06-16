/*
 * sp_cuda_common.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SP_CUDA_COMMON_H_
#define SP_CUDA_COMMON_H_

#include <stdio.h>
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

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

#endif /* SP_CUDA_COMMON_H_ */
