/*
 * dataspace.cpp
 *
 *  Created on: 2014年11月13日
 *      Author: salmon
 */

#include "data_space.h"
#include "data_type.h"
#include "../utilities/utilities.h"

#if  !NO_MPI || USE_MPI
#include "../parallel/mpi_comm.h"
#include "../parallel/mpi_datatype.h"
#endif

namespace simpla
{
struct DataSpace::pimpl_s
{
	typedef nTuple<size_t, MAX_NDIMS_OF_ARRAY> dims_type;

	size_t ndims = 3;

	dims_type dimensions;
	dims_type offset;
	dims_type stride;
	dims_type count;
	dims_type block;

	std::shared_ptr<DataSpace> global_space_;

	bool is_valid() const;
};

//===================================================================
DataSpace::DataSpace() :
		pimpl_(nullptr)
{

}
DataSpace::DataSpace(int rank, const size_t * dims) :
		pimpl_(nullptr)
{
	init(rank, dims);
}

DataSpace::DataSpace(const DataSpace& other) :
		pimpl_(nullptr)
{
	if (other.pimpl_ != nullptr)
	{
		init(other.pimpl_->ndims, &other.pimpl_->dimensions[0]);
		select_hyperslab(&other.pimpl_->offset[0], &other.pimpl_->count[0],
				&other.pimpl_->stride[0], &other.pimpl_->block[0]);
	}

}

DataSpace& DataSpace::operator=(const DataSpace& rhs)
{
	DataSpace(rhs).swap(*this);
	return *this;
}

DataSpace::~DataSpace()
{
	if (pimpl_ != nullptr)
		delete pimpl_;
}

void DataSpace::swap(DataSpace &other)
{
	std::swap(pimpl_->ndims, other.pimpl_->ndims);
	std::swap(pimpl_->dimensions, other.pimpl_->dimensions);
	std::swap(pimpl_->count, other.pimpl_->count);
	std::swap(pimpl_->offset, other.pimpl_->offset);
	std::swap(pimpl_->stride, other.pimpl_->stride);
	std::swap(pimpl_->block, other.pimpl_->block);
	std::swap(pimpl_->global_space_, other.pimpl_->global_space_);
}

void DataSpace::init(int rank, const size_t * dims)
{
	if (pimpl_ == nullptr)
		pimpl_ = new pimpl_s;

	pimpl_->ndims = rank;
	pimpl_->dimensions = dims;
	pimpl_->count = dims;
	pimpl_->offset = 0;
	pimpl_->stride = 1;
	pimpl_->block = 1;
}

bool DataSpace::is_valid() const
{
	return pimpl_ != nullptr && pimpl_->is_valid();
}

bool DataSpace::pimpl_s::is_valid() const
{ //TODO check the legality of hyperslab   ;
	return true;
}
bool DataSpace::is_distributed() const
{
	return pimpl_->global_space_ != nullptr;
}

DataSpace const & DataSpace::global_space() const
{
	return (pimpl_->global_space_ == nullptr) ?
			(*this) : (*pimpl_->global_space_);
}

std::tuple<size_t, size_t const *, size_t const *, size_t const *,
		size_t const *, size_t const *> DataSpace::shape() const
{
	return std::forward_as_tuple(pimpl_->ndims, &pimpl_->dimensions[0],
			&pimpl_->offset[0], &pimpl_->count[0], &pimpl_->stride[0],
			&pimpl_->block[0]);
}

bool DataSpace::select_hyperslab(size_t const * start, size_t const * count,
		size_t const * stride, size_t const * block)
{
	if (pimpl_ == nullptr)
		return false;

	if (start != nullptr)
		pimpl_->offset = start;
	if (count != nullptr)
		pimpl_->count = count;
	if (stride != nullptr)
		pimpl_->stride = stride;
	if (block != nullptr)
		pimpl_->block = block;

	return pimpl_->is_valid();
}

//
//void decomposer_(size_t num_process, size_t process_num, size_t gw,
//		size_t ndims, size_t const *global_start, size_t const * global_count,
//		size_t * local_outer_start, size_t * local_outer_count,
//		size_t * local_inner_start, size_t * local_inner_count)
//{
//
////FIXME this is wrong!!!
//	for (int i = 0; i < ndims; ++i)
//	{
//		local_outer_count[i] = global_count[i];
//		local_outer_start[i] = global_start[i];
//		local_inner_count[i] = global_count[i];
//		local_inner_start[i] = global_start[i];
//	}
//
//	if (num_process <= 1)
//		return;
//
//	int n = 0;
//	long L = 0;
//	for (int i = 0; i < ndims; ++i)
//	{
//		if (global_count[i] > L)
//		{
//			L = global_count[i];
//			n = i;
//		}
//	}
//
//	if ((2 * gw * num_process > global_count[n] || num_process > global_count[n]))
//	{
//
//		RUNTIME_ERROR("Array is too small to split");
//
////		if (process_num > 0)
////			local_outer_end = local_outer_begin;
//	}
//	else
//	{
//		local_inner_start[n] = (global_count[n] * process_num) / num_process
//				+ global_start[n];
//		local_inner_count[n] = (global_count[n] * (process_num + 1))
//				/ num_process + global_start[n];
//		local_outer_start[n] = local_inner_start[n] - gw;
//		local_outer_count[n] = local_inner_count[n] + gw;
//	}
//
//}
//
//void DataSpace::pimpl_s::decompose()
//{
//
////	local_shape_.dimensions = global_shape_.dimensions;
////	local_shape_.count = global_shape_.count;
////	local_shape_.offset = global_shape_.offset;
////	local_shape_.stride = global_shape_.stride;
////	local_shape_.block = global_shape_.block;
//
////	if (!GLOBAL_COMM.is_valid()) return;
////
////	int num_process = GLOBAL_COMM.get_size();
////	unsigned int process_num = GLOBAL_COMM.get_rank();
////
////	decomposer_(num_process, process_num, gw_, ndims_,  //
////			&global_shape_.offset[0], &global_shape_.count[0],  //
////			&local_outer_shape_.offset[0], &local_outer_shape_.count[0],  //
////			&local_inner_shape_.offset[0], &local_inner_shape_.count[0]);
////
////	self_id_ = (process_num);
////
////	for (int dest = 0; dest < num_process; ++dest)
////	{
////		if (dest == self_id_)
////			continue;
////
////		sub_array_s node;
////
////		decomposer_(num_process, dest, gw, ndims_, &global_shape_.offset[0],
////				&global_shape_.count[0], &node.outer_offset[0],
////				&node.outer_count[0], &node.inner_offset[0],
////				&node.inner_count[0]
////
////				);
////
////		sub_array_s remote;
////
////		for (unsigned long s = 0, s_e = (1UL << (ndims_ * 2)); s < s_e; ++s)
////		{
////			remote = node;
////
////			bool is_duplicate = false;
////
////			for (int i = 0; i < ndims_; ++i)
////			{
////
////				int n = (s >> (i * 2)) & 3UL;
////
////				if (n == 3)
////				{
////					is_duplicate = true;
////					continue;
////				}
////
////				auto L = global_shape_.count[i] * ((n + 1) % 3 - 1);
////
////				remote.outer_offset[i] += L;
////				remote.inner_offset[i] += L;
////
////			}
////			if (!is_duplicate)
////			{
////				bool f_inner = Clipping(ndims_, local_outer_shape_.offset,
////						local_outer_shape_.count, remote.inner_offset,
////						remote.inner_count);
////				bool f_outer = Clipping(ndims_, local_inner_shape_.offset,
////						local_inner_shape_.count, remote.outer_offset,
////						remote.outer_count);
////
////				bool flag = f_inner && f_outer;
////
////				for (int i = 0; i < ndims_; ++i)
////				{
////					flag = flag && (remote.outer_count[i] != 0);
////				}
////				if (flag)
////				{
////					send_recv_.emplace_back(
////							send_recv_s(
////									{ dest, hash(&remote.outer_offset[0]), hash(
////											&remote.inner_offset[0]),
////											remote.outer_offset,
////											remote.outer_count,
////											remote.inner_offset,
////											remote.inner_count }));
////				}
////			}
////		}
////	}
//
//	is_valid_ = true;
//}

//bool DataSpace::sync(std::shared_ptr<void> data, DataType const & datatype,
//		size_t flag)
//{
//#if  !NO_MPI || USE_MPI
//	if (!GLOBAL_COMM.is_valid() || pimpl_->send_recv_.size() == 0)
//	{
//		return true;
//	}
//
//	MPI_Comm comm = GLOBAL_COMM.comm();
//
//	MPI_Request request[pimpl_->send_recv_.size() * 2];
//
//	int count = 0;
//
//	for (auto const & item : pimpl_->send_recv_)
//	{
//
//		MPIDataType send_type = MPIDataType::create(datatype, pimpl_->local_shape_.ndims ,
//		&pimpl_->local_shape_.dimensions[0], & item.send.offset[0],
//		&item.send.stride[0], &item.send.count[0], &item.send.block[0]);
//
//		dims_type recv_offset;
//		recv_offset = item.recv.offset - pimpl_->local_shape_.offset;
//
//		MPIDataType recv_type = MPIDataType::create(datatype, pimpl_->local_shape_.ndims ,
//		&pimpl_->local_shape_.dimensions[0], & item.recv.offset[0],
//		&item.recv.stride[0], &item.recv.count[0], &item.recv.block[0]);
//
//		MPI_Isend(data.get(), 1, send_type.type(), item.dest, item.send_tag,
//		comm, &request[count * 2]);
//		MPI_Irecv(data.get(), 1, recv_type.type(), item.dest, item.recv_tag,
//		comm, &request[count * 2 + 1]);
//
//		++count;
//	}
//
//	MPI_Waitall(pimpl_->send_recv_.size() * 2, request, MPI_STATUSES_IGNORE);
//
//#endif //#if  !NO_MPI || USE_MPI
//
//	return true;
//}

}
// namespace simpla
