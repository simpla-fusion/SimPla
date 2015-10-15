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

struct DistributedArray::pimpl_s
{
	std::vector<mpi_send_recv_s> m_send_recv_list_;
	std::vector<mpi_send_recv_buffer_s> m_send_recv_buffer_;
	std::vector<MPI_Request> m_mpi_requests_;

	int m_object_id_;

};

//! Default constructor
DistributedArray::DistributedArray(DataType const &d_type, DataSpace const &d_space)
		: pimpl_(new pimpl_s)
{
	pimpl_->m_object_id_ = SingletonHolder<simpla::MPIComm>::instance().generate_object_id();

	DataSet::datatype = d_type;

	DataSet::dataspace = d_space;

	DataSet::data = nullptr;

}

DistributedArray::DistributedArray(DistributedArray const &other) : pimpl_(new pimpl_s),
		DataSet(other)
{
}

DistributedArray::~DistributedArray()
{
}

void DistributedArray::swap(DistributedArray &other)
{
	std::swap(pimpl_, other.pimpl_);
	DataSet::swap(other);
}


void DistributedArray::sync()
{
	if (GLOBAL_COMM.is_valid() && !pimpl_->m_send_recv_list_.empty())
	{
		sync_update_continue(pimpl_->m_send_recv_list_, DataSet::data.get());
	}
}

void DistributedArray::async()
{
	if (GLOBAL_COMM.is_valid() && !pimpl_->m_send_recv_list_.empty())
	{
		sync_update_continue(pimpl_->m_send_recv_list_, DataSet::data.get(), &(pimpl_->m_mpi_requests_));
	}
}


void DistributedArray::wait()
{

	DataSet::deploy();

	if (GLOBAL_COMM.is_valid() && !pimpl_->m_mpi_requests_.empty())
	{
		wait_all_request(const_cast<std::vector<MPI_Request> *>(&pimpl_->m_mpi_requests_));
	}
}

//bool DistributedArray::is_valid() const
//{
//	return DataSet::is_valid() && (!pimpl_);
//}

bool DistributedArray::is_ready() const
{
	//! FIXME this is not multi-threads safe

	if (!is_valid())
	{
		return false;
	}

	if (pimpl_->m_mpi_requests_.size() > 0)
	{
		int flag = 0;
		MPI_ERROR(MPI_Testall(static_cast<int>( pimpl_->m_mpi_requests_.size()), //
				const_cast<MPI_Request *>(&pimpl_->m_mpi_requests_[0]),//
				&flag, MPI_STATUSES_IGNORE));

		return flag != 0;
	}

	return true;

}


void DistributedArray::deploy()
{
	DataSet::deploy();

	if ((!GLOBAL_COMM.is_valid()))
	{
		return;
	}

	pimpl_->m_send_recv_list_.clear();

	auto global_shape = DataSet::dataspace.global_shape();

	auto local_shape = DataSet::dataspace.local_shape();

	int ndims = global_shape.ndims;

	nTuple<size_t, MAX_NDIMS_OF_ARRAY> l_dims, l_offset, l_stride, l_count, l_block, ghost_width;

	l_dims = local_shape.dimensions;
	l_offset = local_shape.offset;
	l_stride = local_shape.stride;
	l_count = local_shape.count;
	l_block = local_shape.block;

	ghost_width = l_offset;

	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_count, send_offset;
	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_count, recv_offset;

	for (unsigned int tag = 0, tag_e = (1U << (ndims * 2)); tag < tag_e; ++tag)
	{
		nTuple<int, 3> coord_shift;

		bool tag_is_valid = true;

		for (int n = 0; n < ndims; ++n)
		{
			if (((tag >> (n * 2)) & 3UL) == 3UL)
			{
				tag_is_valid = false;
				break;
			}

			coord_shift[n] = ((tag >> (n * 2)) & 3U) - 1;

			switch (coord_shift[n])
			{
			case 0:
				send_count[n] = l_count[n];
				send_offset[n] = l_offset[n];
				recv_count[n] = l_count[n];
				recv_offset[n] = l_offset[n];
				break;
			case -1: //left

				send_count[n] = ghost_width[n];
				send_offset[n] = l_offset[n];

				recv_count[n] = ghost_width[n];
				recv_offset[n] = l_offset[n] - ghost_width[n];

				break;
			case 1: //right
				send_count[n] = ghost_width[n];
				send_offset[n] = l_offset[n] + l_count[n] - ghost_width[n];

				recv_count[n] = ghost_width[n];
				recv_offset[n] = l_offset[n] + l_count[n];
				break;
			default:
				tag_is_valid = false;
				break;
			}

			if (send_count[n] == 0 || recv_count[n] == 0)
			{
				tag_is_valid = false;
				break;
			}

		}

		if (tag_is_valid && (coord_shift[0] != 0 || coord_shift[1] != 0 || coord_shift[2] != 0))
		{

			int dest, send_tag, recv_tag;

			std::tie(dest, send_tag, recv_tag) = GLOBAL_COMM.make_send_recv_tag(pimpl_->m_object_id_, &coord_shift[0]);

			pimpl_->m_send_recv_list_.emplace_back(

					mpi_send_recv_s {

							dest, send_tag, recv_tag,

							MPIDataType::create(DataSet::datatype, ndims, &l_dims[0], &send_offset[0], nullptr,
									&send_count[0], nullptr),

							MPIDataType::create(DataSet::datatype, ndims, &l_dims[0], &recv_offset[0], nullptr,
									&recv_count[0], nullptr)

					}

			);
		}
	}

}


}// namespace simpla


#endif /* CORE_PARALLEL_DISTRIBUTED_ARRAY_CPP_ */
