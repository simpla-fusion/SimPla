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
		nTuple<NDIMS, long> outer_begin;
		nTuple<NDIMS, long> outer_end;
		nTuple<NDIMS, long> inner_begin;
		nTuple<NDIMS, long> inner_end;
	};
	DistributedArray()
			: self_id_(0)
	{
	}

	template<typename ...Args>
	DistributedArray(nTuple<NDIMS, long> global_begin, nTuple<NDIMS, long> global_end, Args const & ... args)

	{

		global_end_ = global_end;
		global_begin_ = global_begin;

		Decompose(std::forward<Args const &>(args)...);
	}

	~DistributedArray()
	{
	}
	size_t size() const
	{
		return NProduct(local_.inner_end - local_.inner_begin);
	}
	size_t memory_size() const
	{
		return NProduct(local_.outer_end - local_.outer_begin);
	}

	void Decompose(int num_process, int process_num, long gw);

	nTuple<NDIMS, long> global_begin_;
	nTuple<NDIMS, long> global_end_;

	nTuple<NDIMS, long> global_strides_;

	sub_array_s local_;

	struct send_recv_s
	{
		int dest;
		int send_tag;
		int recv_tag;
		nTuple<NDIMS, long> send_begin;
		nTuple<NDIMS, long> send_end;
		nTuple<NDIMS, long> recv_begin;
		nTuple<NDIMS, long> recv_end;
	};

	std::vector<send_recv_s> send_recv_; // dest, send_tag,recv_tag, sub_array_s

	void Decomposer_(int num_process, int process_num, unsigned int gw, sub_array_s *) const;

	int hash(nTuple<NDIMS, long> const & d) const
	{
		int res = 0;
		for (int i = 0; i < NDIMS; ++i)
		{
			res += ((d[i] - global_begin_[i] + (global_end_[i] - global_begin_[i]))
			        % (global_end_[i] - global_begin_[i])) * global_strides_[i];
		}
		return res;
	}
}
;

template<int N>
void DistributedArray<N>::Decomposer_(int num_process, int process_num, unsigned int gw, sub_array_s * local) const
{
	local->outer_end = global_end_;
	local->outer_begin = global_begin_;
	local->inner_end = global_end_;
	local->inner_begin = global_begin_;

	if (num_process <= 1)
		return;

	int n = 0;
	long L = 0;
	for (int i = 0; i < NDIMS; ++i)
	{
		if ((global_end_[i] - global_begin_[i]) > L)
		{
			L = (global_end_[i] - global_begin_[i]);
			n = i;
		}
	}

	if ((2 * gw * num_process > (global_end_[n] - global_begin_[n]) || num_process > (global_end_[n] - global_begin_[n])))
	{
		if (process_num > 0)
			local->outer_end = local->outer_begin;
	}
	else
	{
		local->inner_begin[n] = ((global_end_[n] - global_begin_[n]) * process_num) / num_process + global_begin_[n];
		local->inner_end[n] = ((global_end_[n] - global_begin_[n]) * (process_num + 1)) / num_process
		        + global_begin_[n];
		local->outer_begin[n] = local->inner_begin[n] - gw;
		local->outer_end[n] = local->inner_end[n] + gw;
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
			global_strides_[i] = (global_end_[i] - global_begin_[i]) * global_strides_[i + 1];
		}
	}
	else
	{
		global_strides_[0] = 1;
		for (int i = 1; i < NDIMS; ++i)
		{
			global_strides_[i] = (global_end_[i] - global_begin_[i]) * global_strides_[i - 1];
		}

	}

	for (int dest = 0; dest < num_process; ++dest)
	{
		if (dest == self_id_)
			continue;

		sub_array_s node;

		Decomposer_(num_process, dest, gw, &(node));

		// assume no overlap in inner area
		// consider periodic boundary condition, traversal all neighbour

		sub_array_s remote;
		for (unsigned long s = 0, s_e = (1UL << (NDIMS * 2)); s < s_e; ++s)
		{
			remote = node;

			bool is_duplicate = false;

			for (int i = 0; i < NDIMS; ++i)
			{

				int n = (s >> (i * 2)) & 3UL;

				if (n == 3)
				{
					is_duplicate = true;
					continue;
				}

				auto L = (global_end_[i] - global_begin_[i]) * ((n + 1) % 3 - 1);		// 0 1 2 => 0 1 -1

				remote.outer_begin[i] += L;
				remote.outer_end[i] += L;
				remote.inner_begin[i] += L;
				remote.inner_end[i] += L;

			}

			if (!is_duplicate)
			{
				bool f_inner = Clipping(local_.outer_begin, local_.outer_end, &remote.inner_begin, &remote.inner_end);
				bool f_outer = Clipping(local_.inner_begin, local_.inner_end, &remote.outer_begin, &remote.outer_end);

				if (f_inner && f_outer && (remote.outer_begin != remote.outer_end))
				{
					send_recv_.emplace_back(send_recv_s( { dest, hash(remote.outer_begin), hash(remote.inner_begin),
					        remote.outer_begin, remote.outer_end, remote.inner_begin, remote.inner_end }));
				}
			}
		}

	}

}

}
// namespace simpla

#endif /* DISTRIBUTED_ARRAY_H_ */
