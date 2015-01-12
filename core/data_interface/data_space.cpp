/**
 * @file dataspace.cpp
 *
 *  Created on: 2014年11月13日
 *      Author: salmon
 */

#include "../data_interface/data_space.h"

#include "../data_interface/data_type.h"
#include "../utilities/utilities.h"
#include "../numeric/geometric_algorithm.h"
#if  !NO_MPI || USE_MPI
#include "../parallel/mpi_comm.h"
#include "../parallel/mpi_datatype.h"
#endif

namespace simpla
{
struct DataSpace::pimpl_s
{
	typedef nTuple<size_t, MAX_NDIMS_OF_ARRAY> dims_type;

	pimpl_s();

	pimpl_s(pimpl_s const &);

	~pimpl_s();

	void swap(pimpl_s & other);

	bool is_valid() const;

	void init(int rank, const size_t * dims);

	size_t ndims = 3;

	dims_type dimensions;
	dims_type offset;
	dims_type stride;
	dims_type count;
	dims_type block;

	DataSpace * local_space_;

};

//===================================================================

DataSpace::DataSpace() :
		pimpl_(nullptr)
{
}
DataSpace::DataSpace(const DataSpace& other) :
		pimpl_(nullptr), neighgours_(other.neighgours_)
{
	if (other.pimpl_ != nullptr)
	{
		pimpl_ = new pimpl_s(*other.pimpl_);
	}

}

DataSpace::DataSpace(int rank, size_t const * dims, const size_t * gw)
{
	init(rank, dims, gw);
}
DataSpace::~DataSpace()
{
	if (pimpl_ != nullptr)
	{
		delete pimpl_;
	}
}

DataSpace& DataSpace::operator=(const DataSpace& rhs)
{
	DataSpace(rhs).swap(*this);
	return *this;
}

void DataSpace::swap(DataSpace &other)
{
	pimpl_->swap(*other.pimpl_);
	neighgours_.swap(other.neighgours_);
}
void DataSpace::init(int rank, const size_t * dims, const size_t * gw)
{
	if (pimpl_ == nullptr)
		pimpl_ = new pimpl_s;

	pimpl_->init(rank, dims);

	if (gw != nullptr)
		decompose(0, gw);

}

void DataSpace::decompose(int num_procs, size_t const * gw)
{
#if !NO_MPI || USE_MPI
	pimpl_s::dims_type g_begin, g_end, count, l_begin, l_end, l_dims,
			ghost_width;

	if (num_procs == 0)
		num_procs = GLOBAL_COMM.get_size();

	if (gw == nullptr)
	{
		ghost_width = 2;
	}
	else
	{
		ghost_width = gw;
	}

	for (int proc_num = 0; proc_num < num_procs; ++proc_num)
	{
		g_begin = pimpl_->offset;
		g_end = pimpl_->offset + pimpl_->count;

		std::tie(g_begin, g_begin) = block_decompose(g_begin, g_begin,
				num_procs, proc_num);

		count = (g_end - g_begin);

		l_dims = (count + ghost_width * 2) * pimpl_->stride;

		neighgours_.emplace(
				std::make_pair(proc_num,
						DataSpace(pimpl_->ndims, &pimpl_->dimensions[0])));

		neighgours_[proc_num].select_hyperslab(&g_begin[0], &count[0],
				&(pimpl_->stride[0]), &(pimpl_->block[0]));

		neighgours_[proc_num].pimpl_->local_space_ = new DataSpace(
				pimpl_->ndims, &l_dims[0]);

		neighgours_[proc_num].pimpl_->local_space_->select_hyperslab(
				&ghost_width[0], &count[0], &(pimpl_->stride[0]),
				&(pimpl_->block[0]));

	}

	if (pimpl_->local_space_ != nullptr)
		delete pimpl_->local_space_;

	pimpl_->local_space_ = new DataSpace(neighgours_[GLOBAL_COMM.get_rank()]);

#endif
}
void DataSpace::compose(size_t flag)
{

}

bool DataSpace::is_valid() const
{
	return pimpl_ != nullptr && pimpl_->is_valid();
}

bool DataSpace::is_distributed() const
{
	return neighgours_.size() > 0;
}

DataSpace const & DataSpace::local_space() const
{
	if (pimpl_->local_space_ != nullptr)
	{
		return *(pimpl_->local_space_);
	}
	else
	{
		return *this;
	}
}

DataSpace::pimpl_s::pimpl_s() :
		local_space_(nullptr)
{

}
DataSpace::pimpl_s::pimpl_s(pimpl_s const &other) :
		local_space_(nullptr)
{
	ndims = other.ndims;
	dimensions = other.dimensions;
	count = other.count;
	offset = other.offset;
	stride = other.stride;
	block = other.block;
	if (other.local_space_ != nullptr)
		local_space_ = new DataSpace(*other.local_space_);

}

DataSpace::pimpl_s::~pimpl_s()
{
	if (local_space_ != nullptr)
		delete local_space_;
}
void DataSpace::pimpl_s::swap(pimpl_s & other)
{
	std::swap(ndims, other.ndims);
	std::swap(dimensions, other.dimensions);
	std::swap(count, other.count);
	std::swap(offset, other.offset);
	std::swap(stride, other.stride);
	std::swap(block, other.block);
	std::swap(local_space_, other.local_space_);
}
void DataSpace::pimpl_s::init(int rank, const size_t * dims)
{
	ndims = rank;
	dimensions = dims;
	count = dims;
	offset = 0;
	stride = 1;
	block = 1;

}

size_t DataSpace::size() const
{
	size_t res = 1;

	for (int i = 0; i < pimpl_->ndims; ++i)
	{
		res *= pimpl_->dimensions[i];
	}

	return res;
}

bool DataSpace::pimpl_s::is_valid() const
{
// TODO valid the data space
	return true;
}

std::tuple<size_t, size_t const *, size_t const *, size_t const *,
		size_t const *, size_t const *> DataSpace::shape() const
{
	return std::forward_as_tuple(pimpl_->ndims, //
			&pimpl_->dimensions[0], //
			&pimpl_->offset[0], //
			&pimpl_->count[0], //
			&pimpl_->stride[0], //
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
