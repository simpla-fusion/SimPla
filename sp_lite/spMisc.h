/*
 * spIO.h
 *
 *  Created on: 2016年6月16日
 *      Author: salmon
 */

#ifndef SPIO_H_
#define SPIO_H_

enum
{
	SP_NEW = 1UL << 1, SP_APPEND = 1UL << 2, SP_BUFFER = (1UL << 3), SP_RECORD = (1UL << 4)
};

enum
{
	SP_INT, SP_LONG, SP_DOUBLE, SP_FLOAT, SP_OPAQUE
};
void hdf5_write_field(char const url[], void *d, int ndims, size_type const * dims, ptrdiff_t const*offset,
		size_type const *count, int flag);
void hdf5_write_particle(char const url[], void *d, size_type ele_size_in_byte, size_type ndims, int flag);

void mpi_sync_field(void *, size_type ele_size_in_byte, int d_type, int ndims, size_type const * dims,
		size_type *offset, size_type const *count, int flag);
void mpi_sync_particle(char const url[], void *d, size_type ele_size_in_byte, size_type ndims, int flag);

#endif /* SPIO_H_ */
