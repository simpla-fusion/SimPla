/*
 * sync_array.h
 *
 *  Created on: 2014年5月26日
 *      Author: salmon
 */

#ifndef SYNC_ARRAY_H_
#define SYNC_ARRAY_H_

#include <mpi.h>
#include <stddef.h>
#include <memory>
#include <map>
#include <pair>
#include "message_comm.h"
#include "mpi_datatype.h"

namespace simpla
{
template<int N>
struct DistributedArray
{
public:
	static constexpr int NDIMS = N;
	MPI_Comm comm_;

	unsigned int array_order_ = MPI_ORDER_C;

	std::vector<int> neighbour_;

	DistributedArray(MPI_Comm comm, unsigned int gw) :
			comm_(comm)
	{
	}

	~DistributedArray()
	{

	}
	void Decompose(int num_process, int process_num, size_t gw,

	nTuple<NDIMS, size_t> const &global_start, nTuple<NDIMS, size_t> const &global_count,

	nTuple<NDIMS, size_t> *outer_start, nTuple<NDIMS, size_t> *outer_count,

	nTuple<NDIMS, size_t> * inner_start, nTuple<NDIMS, size_t> * inner_count) const
	{

	}

	template<typename TV>
	void UpdateGhost(int num_process, int process_self, int gw, nTuple<NDIMS, size_t> const &global_start,
			nTuple<NDIMS, size_t> const &global_count, TV* data) const
	{
		MPI_Win win_;

		nTuple<NDIMS, size_t> local_outer_start;
		nTuple<NDIMS, size_t> local_outer_count;
		nTuple<NDIMS, size_t> local_inner_start;
		nTuple<NDIMS, size_t> local_inner_count;

		Decompose(num_process, process_self, gw, global_start, global_count, &local_outer_start, &local_outer_count,
				&local_inner_start, &local_inner_count);

		size_t local_size = NProduct(local_outer_count);

		MPI_Win_create(data, local_size, sizeof(TV), MPI_INFO_NULL, comm_, &win_);

		for (auto dest : neighbour_)
		{

			nTuple<NDIMS, size_t> remote_outer_start;
			nTuple<NDIMS, size_t> remote_outer_count;
			nTuple<NDIMS, size_t> remote_inner_start;
			nTuple<NDIMS, size_t> remote_inner_count;

			Decompose(num_process, dest, gw, global_start, global_count, &remote_outer_start, &remote_outer_count,
					&remote_inner_start, &remote_inner_count);

			MPI_Datatype local_data_type, remote_data_type;

			nTuple<NDIMS, size_t> local_range_start;
			nTuple<NDIMS, size_t> remote_range_start;
			nTuple<NDIMS, size_t> range_count;

			bool need_update = true;
			for (int i = 0; i < N; ++i)
			{

				local_range_start[i] =
						(remote_outer_start[i] > local_inner_start[i]) ? remote_outer_start[i] : local_inner_start[i];

				range_count[i] =
						(remote_outer_start[i] + remote_outer_count[i] < local_inner_start[i] + remote_inner_count[i]) ?
								remote_outer_start[i] + remote_outer_count[i] - local_range_start[i] :
								local_inner_start[i] + remote_inner_count[i] - local_range_start[i];

				remote_range_start[i] = local_range_start[i];

				if (range_count[i] < 0)
				{
					if ()
					{

					}
					else
					{
						need_update = false;
						break;
					}
				}

			}

			if (need_update)
			{
				MPI_Put(

				data, 1, MPIDataType<TV>(remote_outer_count, range_count, local_range_start - local_outer_start).type(),

				dest, 0, 1,
						MPIDataType<TV>(remote_outer_count, range_count, remote_range_start - remote_outer_start).type()

						, &win_);
			}
		}
		MPI_Win_fence(0, win_);
		MPI_Win_free(&win_);
	}

};
//template<typename TR, typename TF>
//void SyncGhost(std::map<int, std::pair<TR, TR>> const & comm_topology, TF *data)
//{
//
//	//IMCOMMPLETE, Need omptize
//
//	typedef decltype((*data)[comm_topology.begin().seond.first.begin()]) value_type;
//
//	auto & field = *data;
//
//	MPI_Datatype data_type = MPIDataType<value_type>().type();
//
//	unsigned int num = comm_topology.size();
//	MPI_Request request[2 * num];
//
//	std::shared_ptr<value_type> buff_in[num], buff_out[num];
//
//	int n = 0;
//
//	for (auto const & item : comm_topology)
//	{
//
//		auto dest = item.first;
//		auto const & in = item.second.first;
//		auto const & out = item.second.second;
//		int in_tag = dest;
//		int out_tag = GLOBAL_COMM.GetRank();
//
//		buff_in[n] = MEMPOOL.allocate_shared_ptr < value_type > (in.size());
//		buff_out[n] = MEMPOOL.allocate_shared_ptr < value_type > (out.size());
//
//		size_t count = 0;
//		for (auto it : out)
//		{
//			*(buff_out[n] + count) = field[it];
//		}
//
//		MPI_Isend(buff_out[2 * n].get(), out.size(), data_type, dest, out_tag,
//		GLOBAL_COMM.GetComm(),&request[2*n] );
//
//		MPI_Irecv(buff_in[2 * n + 1].get(), in.size(), data_type, dest, in_tag,
//		GLOBAL_COMM.GetComm(),&request[2*n+1]);
//
//		++n;
//	}
//
//	MPI_Waitall(2 * num, request, MPI_STATUS_IGNORE);
//
//	for (auto const & item : comm_topology)
//	{
//		auto const & in = item.second.first;
//		size_t count = 0;
//		for (auto it : in)
//		{
//			field[it] = *(buff_in + count);
//		}
//	}
//
//}

}
// namespace simpla

#endif /* SYNC_ARRAY_H_ */
