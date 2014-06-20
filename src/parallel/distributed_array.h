/*
 * distributed_array.h
 *
 *  Created on: 2014年5月30日
 *      Author: salmon
 */

#ifndef DISTRIBUTED_ARRAY_H_
#define DISTRIBUTED_ARRAY_H_

#include <vector>
#include <functional>
#include <tuple>
#include "../utilities/ntuple.h"
#include "../utilities/singleton_holder.h"
#include "../model/geometric_algorithm.h"
#include "message_comm.h"

#ifdef USE_MPI
#	include <mpi.h>
#	include "mpi_datatype.h"
#endif

namespace simpla
{
template<int N>
struct DistributedArray
{
public:
	static constexpr int NDIMS = N;
	unsigned int array_order_ = MPI_ORDER_C;
	int self_id_;
	struct sub_array_s
	{
		nTuple<NDIMS, long> outer_start;
		nTuple<NDIMS, long> outer_count;
		nTuple<NDIMS, long> inner_start;
		nTuple<NDIMS, long> inner_count;
	};
	DistributedArray()
			: self_id_(0)
	{
	}

	template<typename ...Args>
	DistributedArray(nTuple<NDIMS, long> global_start, nTuple<NDIMS, long> global_count, Args && ... args)

	{

		global_count_ = global_count;
		global_start_ = global_start;

		Decompose(std::forward<Args >(args)...);
	}

	~DistributedArray()
	{
	}
	size_t size() const
	{
		return NProduct(local_.inner_count);
	}
	size_t memory_size() const
	{
		return NProduct(local_.outer_count);
	}

	void Decompose(int num_process, int process_num, long gw);

	nTuple<NDIMS, long> global_start_;
	nTuple<NDIMS, long> global_count_;

	nTuple<NDIMS, long> global_strides_;

	sub_array_s local_;

	struct send_recv_s
	{
		int dest;
		int send_tag;
		int recv_tag;
		nTuple<NDIMS, long> send_start;
		nTuple<NDIMS, long> send_count;
		nTuple<NDIMS, long> recv_start;
		nTuple<NDIMS, long> recv_count;
	};

	std::vector<send_recv_s> send_recv_; // dest, send_tag,recv_tag, sub_array_s

	void Decomposer_(int num_process, int process_num, unsigned int gw, sub_array_s *) const;

	int hash(nTuple<NDIMS, long> const & d) const
	{
		int res = 0;
		for (int i = 0; i < NDIMS; ++i)
		{
			res += ((d[i] - global_start_[i] + global_count_[i]) % global_count_[i]) * global_strides_[i];
		}
		return res;
	}
}
;

template<int N>
void DistributedArray<N>::Decomposer_(int num_process, int process_num, unsigned int gw, sub_array_s * local) const
{
	local->outer_count = global_count_;
	local->outer_start = global_start_;
	local->inner_count = global_count_;
	local->inner_start = global_start_;

	if (num_process <= 1)
		return;

	int n = 0;
	long L = 0;
	for (int i = 0; i < NDIMS; ++i)
	{
		if (global_count_[i] > L)
		{
			L = global_count_[i];
			n = i;
		}
	}

	nTuple<NDIMS, long> start, count;

	if ((2 * gw * num_process > global_count_[n] || num_process > global_count_[n]))
	{
		if (process_num > 0)
			local->outer_count = 0;
	}
	else
	{
		local->inner_start[n] += (global_count_[n] * process_num) / num_process;
		local->inner_count[n] = (global_count_[n] * (process_num + 1)) / num_process
		        - (global_count_[n] * process_num) / num_process;
		local->outer_start[n] = local->inner_start[n] - gw;
		local->outer_count[n] = local->inner_count[n] + gw * 2;
	}

}

template<int N>
void DistributedArray<N>::Decompose(int num_process, int process_num, long gw)
{
	Decomposer_(num_process, process_num, gw, &local_);
	self_id_ = (process_num);
	if (num_process <= 1)
		return;

	if (array_order_ == MPI_ORDER_C)
	{
		global_strides_[NDIMS - 1] = 1;
		for (int i = NDIMS - 2; i >= 0; --i)
		{
			global_strides_[i] = global_count_[i] * global_strides_[i + 1];
		}
	}
	else
	{
		global_strides_[0] = 1;
		for (int i = 1; i < NDIMS; ++i)
		{
			global_strides_[i] = global_count_[i] * global_strides_[i - 1];
		}

	}

	for (int dest = 0; dest < num_process; ++dest)
	{

		sub_array_s node;

		Decomposer_(num_process, dest, gw, &(node));

		// assume no overlap in inner area
		// consider periodic boundary condition, traversal all neighbour

		sub_array_s remote;
		for (unsigned long s = 0, s_e = (1UL << (NDIMS * 2)); s < s_e; ++s)
		{
			remote = node;

			for (int i = 0; i < NDIMS; ++i)
			{
				int n = (s >> (i * 2)) & 3UL;

				if (n == 3)
					continue;

				n = (n + 1) % 3 - 1; // 0 1 2 => 0 1 -1

				remote.outer_start[i] += global_count_[i] * n;
				remote.inner_start[i] += global_count_[i] * n;
			}

			bool f_inner = Clipping(local_.outer_start, local_.outer_count, &remote.inner_start, &remote.inner_count);
			bool f_outer = Clipping(local_.inner_start, local_.inner_count, &remote.outer_start, &remote.outer_count);

			if (f_inner && f_outer)
			{
				send_recv_.emplace_back(
				        send_recv_s( { dest, hash(remote.outer_start), hash(remote.inner_start), remote.outer_start,
				                remote.outer_count, remote.inner_start, remote.inner_count }));
			}
		}

	}

}

}
// namespace simpla

#endif /* DISTRIBUTED_ARRAY_H_ */
