/*
 * distributed_array.cpp
 *
 *  Created on: 2014年11月13日
 *      Author: salmon
 */

#ifndef CORE_PARALLEL_DISTRIBUTED_ARRAY_CPP_
#define CORE_PARALLEL_DISTRIBUTED_ARRAY_CPP_
#include <string>

#include "distributed_array.h"
#include "message_comm.h"
#include "../utilities/log.h"
#include "../numeric/geometric_algorithm.h"
#include "mpi_datatype.h"

namespace simpla
{
struct DistributedArray::pimpl_s
{
	size_t ndims_ = 3;

	int self_id_ = 0;

	Properties prop_;

	size_t gw;

	bool is_valid() const;

	void update_ghosts(void * data, DataType const & data_type, size_t *block =
	        nullptr);

	void decompose();

	void init(size_t nd, size_t const * b, size_t const * e, size_t gw_p = 2);

	size_t size() const
	{
		size_t res = 1;

		for (int i = 0; i < ndims_; ++i)
		{
			res *= (local_.inner_end[i] - local_.inner_begin[i]);
		}
		return res;
	}
	size_t memory_size() const
	{
		size_t res = 1;

		for (int i = 0; i < ndims_; ++i)
		{
			res *= (local_.outer_end[i] - local_.outer_begin[i]);
		}
		return res;
	}

	size_t num_of_dims() const;
	size_t local_shape(size_t * dims, size_t * offset, size_t * strides,
	        size_t * count, size_t * block) const;
	size_t global_shape(size_t * dims, size_t * offset, size_t * strides,
	        size_t * count, size_t * block) const;

private:

	static constexpr size_t MAX_NUM_OF_DIMS = 10;
	size_t global_begin_[MAX_NUM_OF_DIMS];
	size_t global_end_[MAX_NUM_OF_DIMS];
	size_t global_strides_[MAX_NUM_OF_DIMS];

	struct sub_array_s
	{
		size_t outer_begin[MAX_NUM_OF_DIMS];
		size_t outer_end[MAX_NUM_OF_DIMS];
		size_t inner_begin[MAX_NUM_OF_DIMS];
		size_t inner_end[MAX_NUM_OF_DIMS];
	};
	sub_array_s local_;

	struct send_recv_s
	{
		int dest;
		int send_tag;
		int recv_tag;
		size_t send_begin[MAX_NUM_OF_DIMS];
		size_t send_end[MAX_NUM_OF_DIMS];
		size_t recv_begin[MAX_NUM_OF_DIMS];
		size_t recv_end[MAX_NUM_OF_DIMS];
	};

	std::vector<send_recv_s> send_recv_; // dest, send_tag,recv_tag, sub_array_s

	int hash(size_t const *d) const
	        {
		int res = 0;
		for (int i = 0; i < ndims_; ++i)
		{
			res += ((d[i] - global_begin_[i]
			        + (global_end_[i] - global_begin_[i]))
			        % (global_end_[i] - global_begin_[i])) * global_strides_[i];
		}
		return res;
	}

};

void DistributedArray::pimpl_s::init(size_t nd, size_t const * b,
        size_t const * e, size_t gw_p = 2)
{
	ndims_ = nd;

	for (int i = 0; i < nd; ++i)
	{
		global_begin_[i] = b[i];
		global_end_[i] = e[i];
	}
	gw = gw_p;
	decompose();
}

void DistributedArray::pimpl_s::update_ghosts(void * data,
        DataType const & data_type, size_t * block)
{
	if (send_recv_.size() == 0)
	{
		return;
	}

	MPI_Comm comm = GLOBAL_COMM.comm();

	MPI_Request request[send_recv_.size() * 2];

	int count = 0;

	for (auto const & item : send_recv_)
	{
		size_t g_outer_count[ndims_];
		size_t send_count[ndims_];
		size_t recv_count[ndims_];
		size_t send_begin[ndims_];
		size_t recv_begin[ndims_];

		for (int i = 0; i < ndims_; ++i)
		{

			g_outer_count[i] = local_.outer_end[i]
			        - local_.outer_begin[i];
			send_count[i] = item.send_end[i] - item.send_begin[i];
			recv_count[i] = item.recv_end[i] - item.recv_begin[i];
			send_begin[i] = item.send_begin[i]
			        - local_.outer_begin[i];
			recv_begin[i] = item.recv_begin[i]
			        - local_.outer_begin[i];
		}
		auto send_type = MPIDataType::create(data_type, ndims_, g_outer_count,
		        send_count, send_begin);
		auto recv_type = MPIDataType::create(data_type, ndims_, g_outer_count,
		        recv_count, recv_begin);

		MPI_Isend(data, 1, send_type.type(), item.dest, item.send_tag, comm,
		        &request[count * 2]);
		MPI_Irecv(data, 1, recv_type.type(), item.dest, item.recv_tag, comm,
		        &request[count * 2 + 1]);

		++count;
	}

	MPI_Waitall(send_recv_.size() * 2, request,
	MPI_STATUSES_IGNORE);

}

void decomposer_(size_t num_process, size_t process_num, size_t gw,
        size_t ndims, size_t const *global_begin, size_t const * global_end,
        size_t * local_outer_begin, size_t * local_outer_end,
        size_t * local_inner_begin, size_t * local_inner_end)
{
	local_outer_end = global_end;
	local_outer_begin = global_begin;
	local_inner_end = global_end;
	local_inner_begin = global_begin;

	if (num_process <= 1)
		return;

	int n = 0;
	long L = 0;
	for (int i = 0; i < ndims; ++i)
	{
		if ((global_end[i] - global_begin[i]) > L)
		{
			L = (global_end[i] - global_begin[i]);
			n = i;
		}
	}

	if ((2 * gw * num_process > (global_end[n] - global_begin[n])
	        || num_process > (global_end[n] - global_begin[n])))
	{

		RUNTIME_ERROR("Array is too small to split");

//		if (process_num > 0)
//			local_outer_end = local_outer_begin;
	}
	else
	{
		local_inner_begin[n] = ((global_end[n] - global_begin[n]) * process_num)
		        / num_process + global_begin[n];
		local_inner_end[n] = ((global_end[n] - global_begin[n])
		        * (process_num + 1)) / num_process + global_begin[n];
		local_outer_begin[n] = local_inner_begin[n] - gw;
		local_outer_end[n] = local_inner_end[n] + gw;
	}

}

void DistributedArray::pimpl_s::decompose()
{
	int num_process = GLOBAL_COMM.get_size();
	unsigned int process_num = GLOBAL_COMM.get_rank();

	decomposer_(num_process, process_num, gw, ndims_,  //
			global_begin_, global_end_,//
			local_.outer_begin, local_.outer_end,//
			local_.inner_begin, local_.inner_end);

	self_id_ = (process_num);

	if (num_process <= 1)
	return;

	global_strides_[0] = 1;

	for (int i = 1; i < ndims_; ++i)
	{
		global_strides_[i] = (global_end_[i] - global_begin_[i])
		* global_strides_[i - 1];
	}

	for (int dest = 0; dest < num_process; ++dest)
	{
		if (dest == self_id_)
		continue;

		sub_array_s node;

		decomposer_(num_process, dest, gw, ndims_, global_begin_,
				global_end_, node.outer_begin, node.outer_end,
				node.inner_begin, node.inner_end);

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

				auto L = (global_end_[i] - global_begin_[i]) * ((n + 1) % 3 - 1);

				remote.outer_begin[i] += L;
				remote.outer_end[i] += L;
				remote.inner_begin[i] += L;
				remote.inner_end[i] += L;

			}
			if (!is_duplicate)
			{
				bool f_inner = Clipping(ndims_, local_.outer_begin, local_.outer_end, remote.inner_begin,
						remote.inner_end);
				bool f_outer = Clipping(ndims_, local_.inner_begin, local_.inner_end, remote.outer_begin,
						remote.outer_end);

				bool flag = f_inner && f_outer;

				for (int i = 0; i < ndims_; ++i)
				{
					flag = flag && (remote.outer_begin[i] != remote.outer_end[i]);
				}
				if (flag)
				{
					send_recv_.emplace_back(send_recv_s(
									{	dest, hash(remote.outer_begin), hash(remote.inner_begin),
										remote.outer_begin, remote.outer_end, remote.inner_begin, remote.inner_end}));
				}
			}
		}
	}

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
Properties & DistributedArray::properties()
{
	return pimpl_->prop_;
}
Properties const& DistributedArray::properties() const
{
	return pimpl_->prop_;
}

void DistributedArray::init(size_t nd, size_t const * b, size_t const* e,
        size_t gw = 2)
{
	pimpl_->init(nd, b, e, gw);
}

void DistributedArray::update_ghosts(void* data, DataType const &dtype,
        size_t *block = nullptr)
{
	pimpl_->update_ghosts(data, dtype, block);
}
}  // namespace simpla

#endif /* CORE_PARALLEL_DISTRIBUTED_ARRAY_CPP_ */
