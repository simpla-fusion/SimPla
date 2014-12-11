/*
 * dataspace.cpp
 *
 *  Created on: 2014年11月13日
 *      Author: salmon
 */

#include "dataspace.h"
#include "data_type.h"
#include "data_set.h"
#include "../utilities/utilities.h"

#if  !NO_MPI || USE_MPI
#include "../parallel/mpi_comm.h"
#include "../parallel/mpi_datatype.h"
#endif

namespace simpla
{
struct DataSpace::pimpl_s
{
	pimpl_s();

	pimpl_s(pimpl_s const &);

	~pimpl_s();

	void swap(pimpl_s &);

	bool is_valid() const;

	void init(size_t nd, size_t const * dims, size_t const * gw = nullptr);

	bool sync(DataSet *ds, size_t flag);

	size_t num_of_dims() const;

	/**
	 * dimensions of global data
	 * @return <global start, global count>
	 */
	std::tuple<size_t const *, size_t const *, size_t const *, size_t const *> global_shape() const;

	/**
	 * dimensions of data in local memory
	 * @return <local start, local count>
	 */
	std::tuple<size_t const *, size_t const *, size_t const *, size_t const *> local_shape() const;

	/**
	 * logical shape of data in local memory, which  is the result of select_hyperslab
	 * @return <strat,count,strides,block>
	 */
	std::tuple<size_t const *, size_t const *, size_t const *, size_t const *> shape() const;

	/**
	 *  select a hyper rectangle from local data
	 * @param start
	 * @param count
	 * @param strides
	 * @param block
	 * @return
	 */
	bool select_hyperslab(size_t const * start, size_t const * count,
			size_t const * stride = nullptr, size_t const * block = nullptr);

	/**
	 *
	 * @param data
	 * @param dtype
	 * @param flag
	 * @return
	 */
	bool sync(std::shared_ptr<void> data, DataType dtype, size_t flag) const;

	Properties properties;

private:

	bool is_valid_ = false;

	size_t ndims_ = 3;

	typedef nTuple<size_t, MAX_NDIMS_OF_ARRAY> dims_type;

	dims_type gw_;
	dims_type dims_;
	dims_type start_;
	dims_type count_;
	dims_type stride_;
	dims_type block_;

	void decompose();

	struct hyperslab_s
	{
		dims_type offset;
		dims_type stride;
		dims_type count;
		dims_type block;
	};

	hyperslab_s global_shape_;

	hyperslab_s local_inner_shape_;
	hyperslab_s local_outer_shape_;

	struct send_recv_s
	{
		int src;
		int dest;
		int send_tag;
		int recv_tag;
		hyperslab_s send;
		hyperslab_s recv;

	};

	std::vector<send_recv_s> send_recv_; // dest, send_tag,recv_tag, sub_array_s

};
DataSpace::pimpl_s::pimpl_s()
{
}
DataSpace::pimpl_s::pimpl_s(pimpl_s const & other)
{
	dims_ = other.dims_;
	start_ = other.start_;
	count_ = other.count_;
	stride_ = other.stride_;
	block_ = other.block_;
}
DataSpace::pimpl_s::~pimpl_s()
{
}
void DataSpace::pimpl_s::swap(pimpl_s &)
{
	UNIMPLEMENTED;
}

bool DataSpace::pimpl_s::is_valid() const
{
	return is_valid_;
}

void DataSpace::pimpl_s::init(size_t nd, size_t const * dims, size_t const * gw)
{
	ndims_ = nd;
	dims_ = dims;
	global_shape_.count = dims;
	global_shape_.offset = 0;
	global_shape_.stride = 1;
	global_shape_.block = 1;

	if (gw != nullptr)
	{
		gw_ = gw;
	}
	else
	{
		gw_ = 0;
	}

	decompose();

}

size_t DataSpace::pimpl_s::num_of_dims() const
{
	return ndims_;
}

std::tuple<size_t const *, size_t const *, size_t const *, size_t const *> DataSpace::pimpl_s::global_shape() const
{

}
std::tuple<size_t const *, size_t const *, size_t const *, size_t const *> DataSpace::pimpl_s::local_shape() const
{

}

std::tuple<size_t const *, size_t const *, size_t const *, size_t const *> DataSpace::pimpl_s::shape() const
{
	return std::forward_as_tuple(&start_[0], &count_[0], &stride_[0],
			&block_[0]);
}
bool DataSpace::pimpl_s::select_hyperslab(size_t const * start,
		size_t const * count, size_t const * stride, size_t const * block)
{
	start_ = start;
	count_ = count;
	stride_ = stride;
	block_ = block;
	return true;
}

void decomposer_(size_t num_process, size_t process_num, size_t gw,
		size_t ndims, size_t const *global_start, size_t const * global_count,
		size_t * local_outer_start, size_t * local_outer_count,
		size_t * local_inner_start, size_t * local_inner_count)
{

	//FIXME this is wrong!!!
	for (int i = 0; i < ndims; ++i)
	{
		local_outer_count[i] = global_count[i];
		local_outer_start[i] = global_start[i];
		local_inner_count[i] = global_count[i];
		local_inner_start[i] = global_start[i];
	}

	if (num_process <= 1)
		return;

	int n = 0;
	long L = 0;
	for (int i = 0; i < ndims; ++i)
	{
		if (global_count[i] > L)
		{
			L = global_count[i];
			n = i;
		}
	}

	if ((2 * gw * num_process > global_count[n] || num_process > global_count[n]))
	{

		RUNTIME_ERROR("Array is too small to split");

//		if (process_num > 0)
//			local_outer_end = local_outer_begin;
	}
	else
	{
		local_inner_start[n] = (global_count[n] * process_num) / num_process
				+ global_start[n];
		local_inner_count[n] = (global_count[n] * (process_num + 1))
				/ num_process + global_start[n];
		local_outer_start[n] = local_inner_start[n] - gw;
		local_outer_count[n] = local_inner_count[n] + gw;
	}

}

void DataSpace::pimpl_s::decompose()
{

	local_outer_shape_.offset = global_shape_.offset;
	local_outer_shape_.count = global_shape_.count;
	local_inner_shape_.offset = global_shape_.offset;
	local_inner_shape_.count = global_shape_.count;

//	if (!GLOBAL_COMM.is_valid()) return;
//
//	int num_process = GLOBAL_COMM.get_size();
//	unsigned int process_num = GLOBAL_COMM.get_rank();
//
//	decomposer_(num_process, process_num, gw_, ndims_,  //
//			&global_shape_.offset[0], &global_shape_.count[0],  //
//			&local_outer_shape_.offset[0], &local_outer_shape_.count[0],  //
//			&local_inner_shape_.offset[0], &local_inner_shape_.count[0]);
//
//	self_id_ = (process_num);
//
//	for (int dest = 0; dest < num_process; ++dest)
//	{
//		if (dest == self_id_)
//			continue;
//
//		sub_array_s node;
//
//		decomposer_(num_process, dest, gw, ndims_, &global_shape_.offset[0],
//				&global_shape_.count[0], &node.outer_offset[0],
//				&node.outer_count[0], &node.inner_offset[0],
//				&node.inner_count[0]
//
//				);
//
//		sub_array_s remote;
//
//		for (unsigned long s = 0, s_e = (1UL << (ndims_ * 2)); s < s_e; ++s)
//		{
//			remote = node;
//
//			bool is_duplicate = false;
//
//			for (int i = 0; i < ndims_; ++i)
//			{
//
//				int n = (s >> (i * 2)) & 3UL;
//
//				if (n == 3)
//				{
//					is_duplicate = true;
//					continue;
//				}
//
//				auto L = global_shape_.count[i] * ((n + 1) % 3 - 1);
//
//				remote.outer_offset[i] += L;
//				remote.inner_offset[i] += L;
//
//			}
//			if (!is_duplicate)
//			{
//				bool f_inner = Clipping(ndims_, local_outer_shape_.offset,
//						local_outer_shape_.count, remote.inner_offset,
//						remote.inner_count);
//				bool f_outer = Clipping(ndims_, local_inner_shape_.offset,
//						local_inner_shape_.count, remote.outer_offset,
//						remote.outer_count);
//
//				bool flag = f_inner && f_outer;
//
//				for (int i = 0; i < ndims_; ++i)
//				{
//					flag = flag && (remote.outer_count[i] != 0);
//				}
//				if (flag)
//				{
//					send_recv_.emplace_back(
//							send_recv_s(
//									{ dest, hash(&remote.outer_offset[0]), hash(
//											&remote.inner_offset[0]),
//											remote.outer_offset,
//											remote.outer_count,
//											remote.inner_offset,
//											remote.inner_count }));
//				}
//			}
//		}
//	}

	is_valid_ = true;
}

bool DataSpace::pimpl_s::sync(DataSet * ds, size_t flag)
{
#if  !NO_MPI || USE_MPI
	if (!GLOBAL_COMM.is_valid() || send_recv_.size() == 0)
	{
		return true;
	}

	MPI_Comm comm = GLOBAL_COMM.comm();

	MPI_Request request[send_recv_.size() * 2];

	int count = 0;

	for (auto const & item : send_recv_)
	{

		dims_type send_offset;
		send_offset = item.send.offset - local_outer_shape_.offset;

		MPIDataType send_type = MPIDataType::create(ds->datatype, ndims_,
		&local_outer_shape_.count[0], &send_offset[0],
		&item.send.stride[0], &item.send.count[0], &item.send.block[0]);

		dims_type recv_offset;
		recv_offset = item.recv.offset - local_outer_shape_.offset;

		MPIDataType recv_type = MPIDataType::create(ds->datatype, ndims_,
		&local_outer_shape_.count[0], &send_offset[0],
		&item.recv.stride[0], &item.recv.count[0], &item.recv.block[0]);

		MPI_Isend(ds->data.get(), 1, send_type.type(), item.dest, item.send_tag,
		comm, &request[count * 2]);
		MPI_Irecv(ds->data.get(), 1, recv_type.type(), item.dest, item.recv_tag,
		comm, &request[count * 2 + 1]);

		++count;
	}

	MPI_Waitall(send_recv_.size() * 2, request, MPI_STATUSES_IGNORE);

#endif //#if  !NO_MPI || USE_MPI

	return true;
}
//===================================================================
DataSpace::DataSpace() :
		pimpl_(new pimpl_s)
{
}
DataSpace::DataSpace(DataSpace const & other) :
		pimpl_(new pimpl_s(*other.pimpl_))
{
}
DataSpace::~DataSpace()
{
	delete pimpl_;
}

void DataSpace::swap(DataSpace &other)
{
	pimpl_->swap(*other.pimpl_);
}

DataSpace DataSpace::create_simple(size_t rank, size_t const* count)
{
	DataSpace res;
	size_t start[rank];
	for (int i = 0; i < rank; ++i)
	{
		start[i] = 0;
	}
	res.init(rank, start, count);
	return std::move(res);
}

bool DataSpace::is_valid() const
{
	return pimpl_->is_valid();
}
Properties & DataSpace::properties(std::string const& key)
{
	return pimpl_->properties(key);
}
Properties const& DataSpace::properties(std::string const& key) const
{
	return pimpl_->properties(key);
}

void DataSpace::init(size_t nd, size_t const* dims, size_t const* gw)
{
	if (pimpl_ == nullptr)
		pimpl_ = new pimpl_s;

	pimpl_->init(nd, dims, gw);
}

bool DataSpace::sync(DataSet *ds, size_t flag)
{
	return pimpl_->sync(ds, flag);
}

size_t DataSpace::num_of_dims() const
{
	return pimpl_->num_of_dims();
}

std::tuple<size_t const *, size_t const *, size_t const *, size_t const *> DataSpace::global_shape() const
{
	return pimpl_->global_shape();
}

std::tuple<size_t const *, size_t const *, size_t const *, size_t const *> DataSpace::local_shape() const
{
	return pimpl_->local_shape();
}

std::tuple<size_t const *, size_t const *, size_t const *, size_t const *> DataSpace::shape() const
{
	return pimpl_->shape();
}

bool DataSpace::select_hyperslab(size_t const * start, size_t const * count,
		size_t const * strides, size_t const * block)
{
	return pimpl_->select_hyperslab(start, count, strides, block);
}

}
// namespace simpla
