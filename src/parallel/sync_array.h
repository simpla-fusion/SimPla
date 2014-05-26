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
#include "../utilities/memory_pool.h"
#include "../utilities/singleton_holder.h"
#include "message_comm.h"
#include "mpi_datatype.h"

namespace simpla
{

template<typename TR, typename TF>
void SyncGhost(std::map<int, std::pair<TR, TR>> const & comm_topology, TF *data)
{

	//IMCOMMPLETE, Need omptize

	typedef decltype((*data)[comm_topology.begin().seond.first.begin()]) value_type;

	auto & field = *data;

	MPI_Datatype data_type = MPIDataType<value_type>().type();

	unsigned int num = comm_topology.size();
	MPI_Request request[2 * num];

	std::shared_ptr<value_type> buff_in[num], buff_out[num];

	int n = 0;

	for (auto const & item : comm_topology)
	{

		auto dest = item.first;
		auto const & in = item.second.first;
		auto const & out = item.second.second;
		int in_tag = dest;
		int out_tag = GLOBAL_COMM.GetRank();

		buff_in[n] = MEMPOOL.allocate_shared_ptr < value_type > (in.size());
		buff_out[n] = MEMPOOL.allocate_shared_ptr < value_type > (out.size());

		size_t count = 0;
		for (auto it : out)
		{
			*(buff_out[n] + count) = field[it];
		}

		MPI_Isend(buff_out[2 * n].get(), out.size(), data_type, dest, out_tag,
		GLOBAL_COMM.GetComm(),&request[2*n] );

		MPI_Irecv(buff_in[2 * n + 1].get(), in.size(), data_type, dest, in_tag,
		GLOBAL_COMM.GetComm(),&request[2*n+1]);

		++n;
	}

	MPI_Waitall(2 * num, request, MPI_STATUS_IGNORE);

	for (auto const & item : comm_topology)
	{
		auto const & in = item.second.first;
		size_t count = 0;
		for (auto it : in)
		{
			field[it] = *(buff_in + count);
		}
	}

}

}
// namespace simpla

#endif /* SYNC_ARRAY_H_ */
