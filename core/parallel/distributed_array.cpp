/*
 * distributed_array.cpp
 *
 *  Created on: 2014-11-13
 *      Author: salmon
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_ARRAY_CPP_
#define CORE_PARALLEL_DISTRIBUTED_ARRAY_CPP_

#include "../gtl/utilities/log.h"

#include "mpi_comm.h"
#include "mpi_aux_functions.h"
#include "mpi_update.h"

#include "distributed_array.h"

namespace simpla
{

template<>
void Distributed<DataSet>::deploy()
{

//
//	auto global_shape = ds.dataspace.global_shape();
//
//	auto local_shape = ds.dataspace.local_shape();
//
//	int ndims = global_shape.ndims;
//
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> l_dims, l_offset, l_stride, l_count, l_block, ghost_width;
//
//	l_dims = local_shape.dimensions;
//	l_offset = local_shape.offset;
//	l_stride = local_shape.stride;
//	l_count = local_shape.count;
//	l_block = local_shape.block;
//
//	ghost_width = l_offset;
//
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_count, send_offset;
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_count, recv_offset;
//
//	for (unsigned int tag = 0, tag_e = (1U << (ndims * 2)); tag < tag_e; ++tag)
//	{
//		nTuple<int, 3> coord_shift;
//
//		bool tag_is_valid = true;
//
//		for (int n = 0; n < ndims; ++n)
//		{
//			if (((tag >> (n * 2)) & 3UL) == 3UL)
//			{
//				tag_is_valid = false;
//				break;
//			}
//
//			coord_shift[n] = ((tag >> (n * 2)) & 3U) - 1;
//
//			switch (coord_shift[n])
//			{
//			case 0:
//				send_count[n] = l_count[n];
//				send_offset[n] = l_offset[n];
//				recv_count[n] = l_count[n];
//				recv_offset[n] = l_offset[n];
//				break;
//			case -1: //left
//
//				send_count[n] = ghost_width[n];
//				send_offset[n] = l_offset[n];
//
//				recv_count[n] = ghost_width[n];
//				recv_offset[n] = l_offset[n] - ghost_width[n];
//
//				break;
//			case 1: //right
//				send_count[n] = ghost_width[n];
//				send_offset[n] = l_offset[n] + l_count[n] - ghost_width[n];
//
//				recv_count[n] = ghost_width[n];
//				recv_offset[n] = l_offset[n] + l_count[n];
//				break;
//			default:
//				tag_is_valid = false;
//				break;
//			}
//
//			if (send_count[n] == 0 || recv_count[n] == 0)
//			{
//				tag_is_valid = false;
//				break;
//			}
//
//		}
//
//		if (tag_is_valid && (coord_shift[0] != 0 || coord_shift[1] != 0 || coord_shift[2] != 0))
//		{
//			DistributedObject::add_link_send(coord_shift,
//					DataSpace::create(ndims, &l_dims[0], &send_offset[0], nullptr, &send_count[0], nullptr),
//					ds.datatype, &ds.data & ds.data);
//
//			DistributedObject::add_link_recv(coord_shift,
//					DataSpace::create(ndims, &l_dims[0], &recv_offset[0], nullptr, &recv_count[0], nullptr),
//					ds.datatype, &ds.data & ds.data);
//
//
//		}
//	}

}

}// namespace simpla


#endif /* CORE_PARALLEL_DISTRIBUTED_ARRAY_CPP_ */
