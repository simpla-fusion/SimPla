/**
 * \file distributed_array.cpp
 *
 * \date    2014年7月29日  上午8:32:26 
 * \author salmon
 */
#include "distributed_array.h"
#include "message_comm.h"
namespace simpla
{
void DistributedArray::Decomposer_(int num_process, unsigned int process_num, unsigned int gw,
        sub_array_s * local) const
{
	local->outer_end = global_end_;
	local->outer_begin = global_begin_;
	local->inner_end = global_end_;
	local->inner_begin = global_begin_;

	if (num_process <= 1)
		return;

	int n = 0;
	long L = 0;
	for (int i = 0; i < ndims; ++i)
	{
		if ((global_end_[i] - global_begin_[i]) > L)
		{
			L = (global_end_[i] - global_begin_[i]);
			n = i;
		}
	}

	if ((2 * gw * num_process > (global_end_[n] - global_begin_[n]) || num_process > (global_end_[n] - global_begin_[n])))
	{

		RUNTIME_ERROR("Array is too small to split");

//		if (process_num > 0)
//			local->outer_end = local->outer_begin;
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

void DistributedArray::Decompose(size_t gw)
{
	Decomposer_(GLOBAL_COMM.get_size(),GLOBAL_COMM.get_rank(), gw, &local_);
	self_id_ = (GLOBAL_COMM.get_rank());

	if (GLOBAL_COMM.get_size() <= 1)
	return;

	global_strides_[0] = 1;

	for (int i = 1; i < ndims; ++i)
	{
		global_strides_[i] = (global_end_[i] - global_begin_[i]) * global_strides_[i - 1];
	}

	for (int dest = 0; dest < GLOBAL_COMM.get_size(); ++dest)
	{
		if (dest == self_id_)
		continue;

		sub_array_s node;

		Decomposer_(GLOBAL_COMM.get_size(), dest, gw, &(node));

		// assume no overlap in inner area
		// consider periodic boundary condition, traversal all neighbor

		    sub_array_s remote;
		    for (unsigned long s = 0, s_e = (1UL << (ndims * 2)); s < s_e; ++s)
		    {
			    remote = node;

			    bool is_duplicate = false;

			    for (int i = 0; i < ndims; ++i)
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
					    send_recv_.emplace_back(send_recv_s(
							    {	dest, hash(&remote.outer_begin[0]), hash(&remote.inner_begin[0]),ndims,
								    remote.outer_begin, remote.outer_end, remote.inner_begin, remote.inner_end}));
				    }
			    }
		    }
	    }

    }

}
// namespace simpla
