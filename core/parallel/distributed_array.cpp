/*
 * distributed_array.cpp
 *
 *  Created on: 2014年11月13日
 *      Author: salmon
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_ARRAY_CPP_
#define CORE_PARALLEL_DISTRIBUTED_ARRAY_CPP_

#include "distributed_array.h"

#include <mpi.h>
#include <memory>
#include <vector>

#include "../data_representation/data_set.h"
#include "../numeric/geometric_algorithm.h"
#include "../utilities/utilities.h"

#include "mpi_comm.h"
#include "mpi_datatype.h"

namespace simpla
{
struct DistributedArray::pimpl_s
{

	typedef nTuple<size_t, MAX_NDIMS_OF_ARRAY> dims_type;

	bool is_valid() const;

	bool sync_ghosts(DataSet * ds, size_t flag) const;

	void decompose();

	void init(size_t nd, size_t const * dims, size_t const * gw_p = nullptr);

	Properties & properties(std::string const &key);

	Properties const& properties(std::string const &key) const;

	size_t size() const
	{
		size_t res = 1;

		for (int i = 0; i < ndims_; ++i)
		{
			res *= local_inner_shape_.count[i];
		}
		return res;
	}
	size_t memory_size() const
	{
		size_t res = 1;

		for (int i = 0; i < ndims_; ++i)
		{
			res *= local_outer_shape_.count[i];
		}
		return res;
	}

	size_t num_of_dims() const;

	std::tuple<size_t const*, size_t const*> local_shape() const;
	std::tuple<size_t const*, size_t const*> global_shape() const;
	std::tuple<size_t const*, size_t const*> shape() const;

private:

	Properties prop_;

	size_t ndims_ = 3;

	int self_id_ = 0;

	dims_type gw;

	bool is_valid_ = false;

	dims_type dimensions;

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

	int hash(size_t const *d) const
	{
		dims_type g_stride;
		g_stride[0] = 1;

		for (int i = 1; i < ndims_; ++i)
		{
			g_stride[i] = global_shape_.count[i] * g_stride[i - 1];
		}
		int res = 0;
		for (int i = 0; i < ndims_; ++i)
		{
			res += ((d[i] - global_shape_.offset[i] + global_shape_.count[i])
					% global_shape_.count[i]) * g_stride[i];
		}
		return res;
	}

};

bool DistributedArray::pimpl_s::is_valid() const
{
	return is_valid_;
}

Properties & DistributedArray::pimpl_s::properties(std::string const &key)
{
	return prop_(key);
}
Properties const& DistributedArray::pimpl_s::properties(
		std::string const &key) const
{
	return prop_(key);
}

size_t DistributedArray::pimpl_s::num_of_dims() const
{
	return ndims_;
}

std::tuple<size_t const*, size_t const*> DistributedArray::pimpl_s::local_shape() const
{
	return std::forward_as_tuple(&local_outer_shape_.offset[0],
			&local_outer_shape_.count[0]);
}
std::tuple<size_t const*, size_t const*> DistributedArray::pimpl_s::global_shape() const
{
	return std::forward_as_tuple(&global_shape_.offset[0],
			&global_shape_.count[0]);
}
std::tuple<size_t const*, size_t const*> DistributedArray::pimpl_s::shape() const
{
	return std::forward_as_tuple(&local_inner_shape_.offset[0],
			&local_inner_shape_.count[0]);
}

void DistributedArray::pimpl_s::init(size_t nd, size_t const * dims,
		size_t const * gw_p)
{
	ndims_ = nd;

	dimensions = dims;
	global_shape_.count = dims;
	global_shape_.offset = 0;
	global_shape_.stride = 1;
	global_shape_.block = 1;

	if (gw_p != nullptr)
	{
		gw = gw_p;
	}
	else
	{
		gw = 0;
	}

	decompose();
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

void DistributedArray::pimpl_s::decompose()
{

	local_outer_shape_.offset = global_shape_.offset;
	local_outer_shape_.count = global_shape_.count;
	local_inner_shape_.offset = global_shape_.offset;
	local_inner_shape_.count = global_shape_.count;

	if (!GLOBAL_COMM.is_valid()) return;

	int num_process = GLOBAL_COMM.get_size();
	unsigned int process_num = GLOBAL_COMM.get_rank();

	decomposer_(num_process, process_num, gw, ndims_,  //
			&global_shape_.offset[0], &global_shape_.count[0],  //
			&local_outer_shape_.offset[0], &local_outer_shape_.count[0],  //
			&local_inner_shape_.offset[0], &local_inner_shape_.count[0]);

	self_id_ = (process_num);

	for (int dest = 0; dest < num_process; ++dest)
	{
		if (dest == self_id_)
			continue;

		sub_array_s node;

		decomposer_(num_process, dest, gw, ndims_, &global_shape_.offset[0],
				&global_shape_.count[0], &node.outer_offset[0],
				&node.outer_count[0], &node.inner_offset[0],
				&node.inner_count[0]

				);

		sub_array_s remote;

		for (unsigned long s = 0, s_e = (1UL << (ndims_ * 2)); s < s_e; ++s)
		{
			remote = node;

			bool is_duplicate = false;

			for (int i = 0; i < ndims_; ++i)
			{

				int n = (s >> (i * 2)) & 3UL;

				if (n == 3)
				{
					is_duplicate = true;
					continue;
				}

				auto L = global_shape_.count[i] * ((n + 1) % 3 - 1);

				remote.outer_offset[i] += L;
				remote.inner_offset[i] += L;

			}
			if (!is_duplicate)
			{
				bool f_inner = Clipping(ndims_, local_outer_shape_.offset,
						local_outer_shape_.count, remote.inner_offset,
						remote.inner_count);
				bool f_outer = Clipping(ndims_, local_inner_shape_.offset,
						local_inner_shape_.count, remote.outer_offset,
						remote.outer_count);

				bool flag = f_inner && f_outer;

				for (int i = 0; i < ndims_; ++i)
				{
					flag = flag && (remote.outer_count[i] != 0);
				}
				if (flag)
				{
					send_recv_.emplace_back(
							send_recv_s(
									{ dest, hash(&remote.outer_offset[0]), hash(
											&remote.inner_offset[0]),
											remote.outer_offset,
											remote.outer_count,
											remote.inner_offset,
											remote.inner_count }));
				}
			}
		}
	}

	is_valid_ = true;
}

bool DistributedArray::pimpl_s::sync_ghosts(DataSet * ds, size_t flag) const
{
//#ifdef USE_MPI
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
		&local_outer_shape_.count[0],
		&send_offset[0],
		&item.send.stride[0],
		&item.send.count[0],
		&item.send.block[0] );

		dims_type recv_offset;
		recv_offset = item.recv.offset - local_outer_shape_.offset;

		MPIDataType recv_type = MPIDataType::create(ds->datatype, ndims_,
		&local_outer_shape_.count[0],
		&send_offset[0],
		&item.recv.stride[0],
		&item.recv.count[0],
		&item.recv.block[0] ));

		MPI_Isend(ds->data.get(), 1, send_type.type(), item.dest, item.send_tag,
		comm, &request[count * 2]);
		MPI_Irecv(ds->data.get(), 1, recv_type.type(), item.dest, item.recv_tag,
		comm, &request[count * 2 + 1]);

		++count;
	}

	MPI_Waitall(send_recv_.size() * 2, request, MPI_STATUSES_IGNORE);
//#endif
	return true;
}

//********************************************************************

DistributedArray::DistributedArray() :
		pimpl_(new pimpl_s)
{
}
DistributedArray::~DistributedArray()
{
	delete pimpl_;
}
bool DistributedArray::is_valid() const
{
	return pimpl_ != nullptr && pimpl_->is_valid();
}
Properties & DistributedArray::properties(std::string const &key)
{
	return pimpl_->properties(key);
}
Properties const& DistributedArray::properties(std::string const &key) const
{
	return pimpl_->properties(key);
}

void DistributedArray::init(size_t nd, size_t const * dims, size_t const *gw)
{
	if (pimpl_ == nullptr)
		pimpl_ = (new pimpl_s);

	pimpl_->init(nd, dims, gw);
}

bool DistributedArray::sync_ghosts(DataSet* ds, size_t flag) const
{
	return pimpl_->sync_ghosts(ds, flag);
}

size_t DistributedArray::num_of_dims() const
{
	return pimpl_->num_of_dims();
}

std::tuple<size_t const *, size_t const *> DistributedArray::global_shape() const
{
	return pimpl_->global_shape();
}

std::tuple<size_t const *, size_t const *> DistributedArray::local_shape() const
{
	return pimpl_->local_shape();
}

}  // namespace simpla

#endif /* CORE_PARALLEL_DISTRIBUTED_ARRAY_CPP_ */
