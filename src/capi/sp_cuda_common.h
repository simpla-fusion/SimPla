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

#endif /* SP_CUDA_COMMON_H_ */
