/**
 * sp_def_cuda.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SP_INTERAL_DEF_H_
#define SP_INTERAL_DEF_H_
#include <stdio.h>
#include "../sp_config.h"

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

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

#endif /* SP_INTERAL_DEF_H_ */
