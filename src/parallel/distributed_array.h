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
#include "../fetl/ntuple.h"
#include "../utilities/singleton_holder.h"
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
		nTuple<NDIMS, size_t> outer_start;
		nTuple<NDIMS, size_t> outer_count;
		nTuple<NDIMS, size_t> inner_start;
		nTuple<NDIMS, size_t> inner_count;
	};
	DistributedArray() :
			self_id_(0)
	{
	}

	template<typename ...Args>
	DistributedArray(nTuple<NDIMS, size_t> global_start, nTuple<NDIMS, size_t> global_count, Args const & ... args)

	{

		global_count_ = global_count;
		global_start_ = global_start;

		Decompose(std::forward<Args const &>(args)...);
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

	void Decompose(int num_process, int process_num, size_t gw);

#ifdef USE_MPI
	template<typename TV> void UpdateGhosts(TV* data, MPI_Comm comm = MPI_COMM_NULL) const;
#else
	template<typename TV> void UpdateGhost(TV* data) const
	{};
#endif

	nTuple<NDIMS, size_t> global_start_;
	nTuple<NDIMS, size_t> global_count_;

	nTuple<NDIMS, size_t> global_strides_;

	sub_array_s local_;
private:
	struct send_recv_s
	{
		int dest;
		int send_tag;
		int recv_tag;
		nTuple<NDIMS, size_t> send_start;
		nTuple<NDIMS, size_t> send_count;
		nTuple<NDIMS, size_t> recv_start;
		nTuple<NDIMS, size_t> recv_count;
	};
	std::vector<send_recv_s> send_recv_; // dest, send_tag,recv_tag, sub_array_s

	void Decomposer_(int num_process, int process_num, unsigned int gw, sub_array_s *) const;

	int hash(nTuple<NDIMS, size_t> const & d) const
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
	size_t L = 0;
	for (int i = 0; i < NDIMS; ++i)
	{
		if (global_count_[i] > L)
		{
			L = global_count_[i];
			n = i;
		}
	}

	nTuple<NDIMS, size_t> start, count;

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
template<int NDIMS>
bool Clipping(nTuple<NDIMS, size_t> const & l_start, nTuple<NDIMS, size_t> const &l_count,
		nTuple<NDIMS, size_t> *pr_start, nTuple<NDIMS, size_t> *pr_count)
{
	bool has_overlap = false;

	nTuple<NDIMS, size_t> & r_start = *pr_start;
	nTuple<NDIMS, size_t> & r_count = *pr_count;

	for (int i = 0; i < NDIMS; ++i)
	{
		if (r_start[i] + r_count[i] <= l_start[i] || r_start[i] >= l_start[i] + l_count[i])
			return false;

		size_t start = std::max(l_start[i], r_start[i]);
		size_t end = std::min(l_start[i] + l_count[i], r_start[i] + r_count[i]);

		if (end > start)
		{
			r_start[i] = start;
			r_count[i] = end - start;

			has_overlap = true;
		}
	}

	return has_overlap;
}
template<int N>
void DistributedArray<N>::Decompose(int num_process, int process_num, size_t gw)
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
				send_recv_.emplace_back(send_recv_s(
				{ dest, hash(remote.outer_start), hash(remote.inner_start), remote.outer_start, remote.outer_count,
						remote.inner_start, remote.inner_count }));
			}
		}

	}

}

#ifdef USE_MPI

template<int N>
template<typename TV>
void DistributedArray<N>::UpdateGhosts(TV* data, MPI_Comm comm) const
{
	if (send_recv_.size() == 0)
		return;

	if (comm == MPI_COMM_NULL)
	{
		comm = GLOBAL_COMM.GetComm();
	}

	MPI_Request request[send_recv_.size() * 2];

	int count = 0;
	for (auto const & item : send_recv_)
	{

		MPIDataType<TV> send_type(comm, local_.outer_count, item.send_count, item.send_start - local_.outer_start);
		MPIDataType<TV> recv_type(comm, local_.outer_count, item.recv_count, item.recv_start - local_.outer_start);

		MPI_Isend(data, 1, send_type.type(), item.dest, item.send_tag, comm, &request[count * 2]);
		MPI_Irecv(data, 1, recv_type.type(), item.dest, item.recv_tag, comm, &request[count * 2 + 1]);
		++count;
	}

	MPI_Waitall(send_recv_.size() * 2, request, MPI_STATUSES_IGNORE);

}

#endif
}
// namespace simpla

#endif /* DISTRIBUTED_ARRAY_H_ */
