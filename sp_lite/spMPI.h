/*
 * spMPI.h
 *
 *  Created on: 2016年6月18日
 *      Author: salmon
 */

#ifndef SPMPI_H_
#define SPMPI_H_


void mpi_sync(void *, size_type ele_size_in_byte, int d_type, int ndims,
		size_type const * dims, size_type *offset, size_type const *count,
		int flag);


#endif /* SPMPI_H_ */
