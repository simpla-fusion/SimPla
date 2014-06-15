/*
 * update_ghosts.h
 *
 *  Created on: 2014年6月4日
 *      Author: salmon
 */

#ifndef UPDATE_GHOSTS_H_
#define UPDATE_GHOSTS_H_

#include <mpi.h>
#include "message_comm.h"
#include "mpi_datatype.h"
#include "distributed_array.h"
namespace simpla
{
template<int N, typename TV>
void UpdateGhosts(TV* data, DistributedArray<N> const & global_array, MPI_Comm comm = MPI_COMM_NULL)
{
	if (global_array.send_recv_.size() == 0)
		return;

	if (comm == MPI_COMM_NULL)
	{
		comm = GLOBAL_COMM.GetComm();
	}

	MPI_Request request[global_array.send_recv_.size() * 2];

	int count = 0;
	for (auto const & item : global_array.send_recv_)
	{

		MPIDataType<TV> send_type(global_array.local_.outer_count, item.send_count,
		        item.send_start - global_array.local_.outer_start);
		MPIDataType<TV> recv_type(global_array.local_.outer_count, item.recv_count,
		        item.recv_start - global_array.local_.outer_start);

		MPI_Isend(data, 1, send_type.type(), item.dest, item.send_tag, comm, &request[count * 2]);
		MPI_Irecv(data, 1, recv_type.type(), item.dest, item.recv_tag, comm, &request[count * 2 + 1]);
		++count;
	}

	MPI_Waitall(global_array.send_recv_.size() * 2, request, MPI_STATUSES_IGNORE);

}

template<typename TG, int IFORM, typename TV> class Field;

template<typename TM, int IFORM, typename TV>
void UpdateGhosts(Field<TM, IFORM, TV>* field, MPI_Comm comm = MPI_COMM_NULL)
{

	auto const & global_array = field->mesh.global_array_;

	TV* data = &(*field->data());

	if (IFORM == VERTEX || IFORM == VOLUME)
	{
		UpdateGhosts(data, global_array, comm);
	}
	else
	{
		UpdateGhosts(reinterpret_cast<nTuple<3, TV>*>(data), global_array, comm);
	}
}

template<typename TM, typename TParticle> class ParticlePool;

template<typename TM, typename TParticle>
void UpdateGhosts(ParticlePool<TM, TParticle> *pool, MPI_Comm comm = MPI_COMM_NULL)
{

	typedef ParticlePool<TM, TParticle> pool_type;
	auto const & g_array = pool->mesh.global_array_;

	if (g_array.send_recv_.size() == 0)
		return;

	if (comm == MPI_COMM_NULL)
	{
		comm = GLOBAL_COMM.GetComm();
	}

	typedef typename pool_type::value_type value_type;

	MPIDataType<value_type> dtype;

	int num_of_neighbour = g_array.send_recv_.size();

	MPI_Request requests[num_of_neighbour * 2];

	std::vector<std::vector<value_type>> buffer(num_of_neighbour * 2);

	int count = 0;

	for (auto const & item : g_array.send_recv_)
	{

		auto t_cell = pool->GetCell();

		pool->Remove(pool->Select(item.send_start, item.send_count), &t_cell);

		CHECK(pool->Select(item.send_start, item.send_count).size());
		CHECK(t_cell.size());

		std::copy(t_cell.begin(), t_cell.end(), std::back_inserter(buffer[count]));

		MPI_Isend(&buffer[count][0], buffer[count].size(), dtype.type(), item.dest, item.send_tag, comm,
		        &requests[count]);

		++count;
	}

	for (auto const & item : g_array.send_recv_)
	{
		MPI_Status status;

		MPI_Probe(item.dest, item.recv_tag, comm, &status);

		// When probe returns, the status object has the size and other
		// attributes of the incoming message. Get the size of the message
		int number = 0;
		MPI_Get_count(&status, dtype.type(), &number);
		CHECK(number);
		buffer[count].resize(number);

		MPI_Irecv(&buffer[count][0], buffer[count].size(), dtype.type(), item.dest, item.recv_tag, comm,
		        &requests[count]);
		++count;
	}

	MPI_Waitall(num_of_neighbour, requests, MPI_STATUSES_IGNORE);

	auto cell_buffer = pool->GetCell();
	for (int i = 0; i < num_of_neighbour; ++i)
	{
		std::copy(buffer[num_of_neighbour + i].begin(), buffer[num_of_neighbour + i].end(),
		        std::back_inserter(cell_buffer));
	}

	pool->Add(&cell_buffer);
}

}  // namespace simpla

#endif /* UPDATE_GHOSTS_H_ */
