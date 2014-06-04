/*
 * particle_parallel.h
 *
 *  Created on: 2014年6月4日
 *      Author: salmon
 */

#ifndef PARTICLE_PARALLEL_H_
#define PARTICLE_PARALLEL_H_

#include <vector>
#include <mpi.h>

#include "../parallel/mpi_datatype.h"

namespace simpla
{

template<typename TPool>
void UpdateGhosts(TPool *pool, MPI_Comm comm = MPI_COMM_NULL)
{
	auto const & g_array = pool->mesh.global_array_;

	if (g_array.send_recv_.size() == 0)
		return;

	if (comm == MPI_COMM_NULL)
	{
		comm = GLOBAL_COMM.GetComm();
	}

	typedef typename TPool::value_type value_type;

	MPIDataType<value_type> dtype;

	MPI_Request request[g_array.send_recv_.size() * 2];

	int count = 0;
	for (auto const & item : g_array.send_recv_)
	{
		auto range = pool->GetRange(item.send_start, item.send_count);
		std::vector<value_type> buffer;
		pool->MoveOut(pool->GetCellRange(item.send_start, item.send_count), &buffer);
		MPI_Isend(&buffer[0], buffer.size(), dtype, item.dest, item.send_tag, comm, &request[count * 2]);
		++count;
	}

	for (auto const & item : g_array.send_recv_)
	{
		auto range = pool->GetRange(item.send_start, item.send_count);

		std::vector<value_type> buffer;

		MPI_Irecv(data, 1, recv_type.type(), item.dest, item.recv_tag, comm, &request[count * 2 + 1]);
		++count;
	}

	MPI_Waitall(send_recv_.size() * 2, request, MPI_STATUSES_IGNORE);

}
}
// namespace simpla

#endif /* PARTICLE_PARALLEL_H_ */
