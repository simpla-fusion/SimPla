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

#include "../data_structure/data_set.h"
#include "../numeric/geometric_algorithm.h"
#include "../utilities/log.h"
#include "../utilities/ntuple.h"
#include "../utilities/primitives.h"
#include "../utilities/properties.h"
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

	void init(size_t nd, size_t const * b, size_t const * e, size_t gw_p = 2);

	Properties & properties(std::string const &key);

	Properties const& properties(std::string const &key) const;

	size_t size() const
	{
		size_t res = 1;

		for (int i = 0; i < ndims_; ++i)
		{
			res *= local_.inner_count[i];
		}
		return res;
	}
	size_t memory_size() const
	{
		size_t res = 1;

		for (int i = 0; i < ndims_; ++i)
		{
			res *= local_.outer_count[i];
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

	size_t gw;

	bool is_valid_ = false;

	dims_type global_start_;
	dims_type global_count_;
	dims_type global_strides_;

	struct sub_array_s
	{
		dims_type outer_start;
		dims_type outer_count;
		dims_type inner_start;
		dims_type inner_count;
	};
	sub_array_s local_;

	struct send_recv_s
	{
		int dest;
		int send_tag;
		int recv_tag;
		dims_type send_start;
		dims_type send_count;
		dims_type recv_start;
		dims_type recv_count;
	};

	std::vector<send_recv_s> send_recv_; // dest, send_tag,recv_tag, sub_array_s

	int hash(size_t const *d) const
	{
		int res = 0;
		for (int i = 0; i < ndims_; ++i)
		{
			res += ((d[i] - global_start_[i] + global_count_[i])
					% global_count_[i]) * global_strides_[i];
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
	return std::forward_as_tuple(&local_.outer_start[0], &local_.outer_count[0]);
}
std::tuple<size_t const*, size_t const*> DistributedArray::pimpl_s::global_shape() const
{
	return std::forward_as_tuple(&global_start_[0], &global_count_[0]);
}
std::tuple<size_t const*, size_t const*> DistributedArray::pimpl_s::shape() const
{
	return std::forward_as_tuple(&local_.inner_start[0], &local_.inner_count[0]);
}

void DistributedArray::pimpl_s::init(size_t nd, size_t const * start,
		size_t const * count, size_t gw_p)
{
	ndims_ = nd;
	global_start_ = start;
	global_count_ = count;
	gw = gw_p;
	decompose();
}

void decomposer_(size_t num_process, size_t process_num, size_t gw,
		size_t ndims, size_t const *global_begin, size_t const * global_count,
		size_t * local_outer_begin, size_t * local_outer_end,
		size_t * local_inner_begin, size_t * local_inner_end)
{
	//FIXME this is wrong!!!
//	local_outer_end = global_end;
//	local_outer_begin = global_begin;
//	local_inner_end = global_end;
//	local_inner_begin = global_begin;

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
		local_inner_begin[n] = (global_count[n] * process_num) / num_process
				+ global_begin[n];
		local_inner_end[n] = (global_count[n] * (process_num + 1)) / num_process
				+ global_begin[n];
		local_outer_begin[n] = local_inner_begin[n] - gw;
		local_outer_end[n] = local_inner_end[n] + gw;
	}

}

void DistributedArray::pimpl_s::decompose()
{
	if (!GLOBAL_COMM.is_valid()) return;

	int num_process = GLOBAL_COMM.get_size();
	unsigned int process_num = GLOBAL_COMM.get_rank();

	decomposer_(num_process, process_num, gw, ndims_,  //
	&global_start_ [0] ,
	&global_count_ [0] ,//
	&local_.outer_start [0] ,
	&local_.outer_count[0] ,//
	&local_.inner_start [0] ,
	&local_.inner_count [0]
	);

	self_id_ = (process_num);

	if (num_process <= 1)
	return;

	global_strides_[0] = 1;

	for (int i = 1; i < ndims_; ++i)
	{
		global_strides_[i] =global_count_[i] * global_strides_[i - 1];
	}

	for (int dest = 0; dest < num_process; ++dest)
	{
		if (dest == self_id_)
		continue;

		sub_array_s node;

		decomposer_(num_process, dest, gw, ndims_,
		&global_start_ [0],
		&global_count_ [0],
		&node.outer_start [0],
		&node.outer_count [0],
		&node.inner_start [0],
		&node.inner_count [0]

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

				auto L =global_count_[i] * ((n + 1) % 3 - 1);

				remote.outer_start[i] += L;
				remote.inner_start[i] += L;

			}
			if (!is_duplicate)
			{
				bool f_inner = Clipping(ndims_, local_.outer_start, local_.outer_count, remote.inner_start,
				remote.inner_count);
				bool f_outer = Clipping(ndims_, local_.inner_start, local_.inner_count, remote.outer_start,
				remote.outer_count);

				bool flag = f_inner && f_outer;

				for (int i = 0; i < ndims_; ++i)
				{
					flag = flag && (remote.outer_count[i] != 0);
				}
				if (flag)
				{
					send_recv_.emplace_back(send_recv_s(
							{	dest, hash(&remote.outer_start[0]), hash(&remote.inner_start[0]),
								remote.outer_start, remote.outer_count,
								remote.inner_start, remote.inner_count}));
				}
			}
		}
	}

	is_valid_=true;
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
		dims_type g_outer_count;
		dims_type send_count;
		dims_type recv_count;
		dims_type send_start;
		dims_type recv_start;

		g_outer_count = local_.outer_count;
		send_count = item.send_count;
		recv_count = item.recv_count;
		send_start = item.send_start - local_.outer_start;
		recv_start = item.recv_start - local_.outer_start;

		MPIDataType send_type = MPIDataType::create(ds->datatype, ndims_,
		&g_outer_count[0], &send_count[0], &send_start[0]);
		MPIDataType recv_type = MPIDataType::create(ds->datatype, ndims_,
		&g_outer_count[0], &recv_count[0], &recv_start[0]);

		MPI_Isend(ds->data.get(), 1, send_type.type(), item.dest, item.send_tag,
		comm, &request[count * 2]);
		MPI_Irecv(ds->data.get(), 1, recv_type.type(), item.dest, item.recv_tag,
		comm, &request[count * 2 + 1]);

		++count;
	}

	MPI_Waitall(send_recv_.size() * 2, request,
	MPI_STATUSES_IGNORE);
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
	return pimpl_->is_valid();
}
Properties & DistributedArray::properties(std::string const &key)
{
	return pimpl_->properties(key);
}
Properties const& DistributedArray::properties(std::string const &key) const
{
	return pimpl_->properties(key);
}

void DistributedArray::init(size_t nd, size_t const * b, size_t const* e,
		size_t gw)
{
	pimpl_->init(nd, b, e, gw);
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
