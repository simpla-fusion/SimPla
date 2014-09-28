/**
 * \file distributed_array.cpp
 *
 * \date    2014年7月29日  上午8:32:26 
 * \author salmon
 */
#include "distributed_array.h"
#include "message_comm.h"
#include "../utilities/log.h"
#include "../numeric/geometric_algorithm.h"
#include "mpi_datatype.h"
namespace simpla
{
template<typename TI, typename TO>
void Decomposer_(int num_process, unsigned int process_num, unsigned int gw, int ndims, TI const & global_begin,
        TI const & global_end, TO & local_outer_begin, TO & local_outer_end, TO & local_inner_begin,
        TO & local_inner_end)
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

	if ((2 * gw * num_process > (global_end[n] - global_begin[n]) || num_process > (global_end[n] - global_begin[n])))
	{

		RUNTIME_ERROR("Array is too small to split");

//		if (process_num > 0)
//			local_outer_end = local_outer_begin;
	}
	else
	{
		local_inner_begin[n] = ((global_end[n] - global_begin[n]) * process_num) / num_process + global_begin[n];
		local_inner_end[n] = ((global_end[n] - global_begin[n]) * (process_num + 1)) / num_process + global_begin[n];
		local_outer_begin[n] = local_inner_begin[n] - gw;
		local_outer_end[n] = local_inner_end[n] + gw;
	}

}

void DistributedArray::Decompose(long gw)
{
	int num_process = GLOBAL_COMM.get_size();
	unsigned int process_num = GLOBAL_COMM.get_rank();

	Decomposer_(num_process, process_num, gw, ndims, global_begin_, global_end_, local_.outer_begin, local_.outer_end,
			local_.inner_begin, local_.inner_end);
	self_id_ = (process_num);

	if (num_process <= 1)
	return;

	global_strides_[0] = 1;

	for (int i = 1; i < ndims; ++i)
	{
		global_strides_[i] = (global_end_[i] - global_begin_[i]) * global_strides_[i - 1];
	}

	for (int dest = 0; dest < num_process; ++dest)
	{
		if (dest == self_id_)
		continue;

		sub_array_s node;

		Decomposer_(num_process, dest, gw, ndims, global_begin_, global_end_, node.outer_begin, node.outer_end,
				node.inner_begin, node.inner_end);

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

				auto L = (global_end_[i] - global_begin_[i]) * ((n + 1) % 3 - 1);

				remote.outer_begin[i] += L;
				remote.outer_end[i] += L;
				remote.inner_begin[i] += L;
				remote.inner_end[i] += L;

			}
			if (!is_duplicate)
			{
				bool f_inner = Clipping(ndims, local_.outer_begin, local_.outer_end, remote.inner_begin,
						remote.inner_end);
				bool f_outer = Clipping(ndims, local_.inner_begin, local_.inner_end, remote.outer_begin,
						remote.outer_end);

				bool flag = f_inner && f_outer;

				for (int i = 0; i < ndims; ++i)
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

void update_ghosts(void * data, DataType const & data_type, DistributedArray const & global_array)
{
	if (global_array.send_recv_.size() == 0)
	{
		return;
	}
	unsigned int ndims = global_array.ndims;

	MPI_Comm comm = GLOBAL_COMM.comm();

	MPI_Request request[global_array.send_recv_.size() * 2];

	int count = 0;

	for (auto const & item : global_array.send_recv_)
	{
		size_t g_outer_count[ndims];
		size_t send_count[ndims];
		size_t recv_count[ndims];
		size_t send_begin[ndims];
		size_t recv_begin[ndims];

		for (int i = 0; i < ndims; ++i)
		{

			g_outer_count[i] = global_array.local_.outer_end[i] - global_array.local_.outer_begin[i];
			send_count[i] = item.send_end[i] - item.send_begin[i];
			recv_count[i] = item.recv_end[i] - item.recv_begin[i];
			send_begin[i] = item.send_begin[i] - global_array.local_.outer_begin[i];
			recv_begin[i] = item.recv_begin[i] - global_array.local_.outer_begin[i];
		}
		auto send_type = MPIDataType::create(data_type, ndims, g_outer_count, send_count, send_begin);
		auto recv_type = MPIDataType::create(data_type, ndims, g_outer_count, recv_count, recv_begin);

		MPI_Isend(data, 1, send_type.type(), item.dest, item.send_tag, comm, &request[count * 2]);
		MPI_Irecv(data, 1, recv_type.type(), item.dest, item.recv_tag, comm, &request[count * 2 + 1]);

		++count;
	}

	MPI_Waitall(global_array.send_recv_.size() * 2, request, MPI_STATUSES_IGNORE);

}
}
// namespace simpla
