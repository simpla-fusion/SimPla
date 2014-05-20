/*
 * decompose.h
 *
 *  Created on: 2014年5月20日
 *      Author: salmon
 */

#ifndef DECOMPOSE_H_
#define DECOMPOSE_H_

#include <stddef.h>
#include <utility>

#include "../fetl/ntuple.h"

namespace simpla
{
template<int NDIMS>
void RectangleDecompose(

nTuple<NDIMS, size_t> const & num_process,

nTuple<NDIMS, size_t> const & process_num,

nTuple<NDIMS, size_t> const & gw,

nTuple<NDIMS, size_t> const & dims,

size_t *global_start, size_t *global_count, size_t *local_start,
		size_t *local_count)
{

	for (int i = 0; i < NDIMS; ++i)
	{

		if (num_process[i] <= 1)
		{
			global_start[i] = 0;
			global_count[i] = dims[i];
			local_start[i] = 0;
			local_count[i] = dims[i];

		}
		else if (2 * gw[i] * num_process[i] > dims[i])
		{
			ERROR << "Mesh is too small to decompose! dims[" << i << "]="
					<< dims[i]

					<< " process[" << i << "]=" << num_process[i]
					<< " ghost_width=" << gw[i];
		}
		else
		{
			global_start[i] = dims[i] * process_num[i] / (num_process[i]);
			global_count[i] = dims[i] * (process_num[i] + 1) / (num_process[i])
					- global_start[i];

			local_start[i] = global_start[i] - gw[i];
			local_count[i] = global_count[i] + 2 * gw[i];
		}

	}

}

}  // namespace simpla

#endif /* DECOMPOSE_H_ */
