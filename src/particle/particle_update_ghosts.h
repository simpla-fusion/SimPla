/*
 * particle_update_ghosts.h
 *
 *  Created on: 2014年6月15日
 *      Author: salmon
 */

#ifndef PARTICLE_UPDATE_GHOSTS_H_
#define PARTICLE_UPDATE_GHOSTS_H_
#include "../parallel/update_ghosts.h"
#include "../utilities/log.h"

namespace simpla
{
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

		auto t_cell = pool->CreateBuffer();

		pool->Remove(pool->Select(item.send_start, item.send_count), &t_cell);

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

		buffer[count].resize(number);

		MPI_Irecv(&buffer[count][0], buffer[count].size(), dtype.type(), item.dest, item.recv_tag, comm,
		        &requests[count]);
		++count;
	}

	MPI_Waitall(num_of_neighbour, requests, MPI_STATUSES_IGNORE);

	auto cell_buffer = pool->CreateBuffer();
	for (int i = 0; i < num_of_neighbour; ++i)
	{
		std::copy(buffer[num_of_neighbour + i].begin(), buffer[num_of_neighbour + i].end(),
		        std::back_inserter(cell_buffer));
	}

	pool->Add(&cell_buffer);
}
}  // namespace simpla

#endif /* PARTICLE_UPDATE_GHOSTS_H_ */
