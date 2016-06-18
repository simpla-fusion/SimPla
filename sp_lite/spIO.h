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
	SP_NEW, SP_RECORD, SP_APPEND
};
enum
{
	SP_INT, SP_LONG, SP_DOUBLE, SP_FLOAT, SP_OPAQUE
};
void hdf5_write(char const url[], void *, size_type ele_size_in_byte,
		int d_type, int ndims, size_type const * dims, size_type *offset,
		size_type const *count, int flag);
#endif /* SPIO_H_ */
