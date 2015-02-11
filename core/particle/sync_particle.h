/**
 * @file  sync_particle.h
 *
 *  created on: 2014-6-15
 *      Author: salmon
 */

#ifndef CORE_PARTICLE_SYNC_PARTICLE_H_
#define CORE_PARTICLE_SYNC_PARTICLE_H_

#include "../utilities/log.h"
#include "../parallel/mpi_comm.h"
#include <mpi.h>

namespace simpla
{
template<typename ...> class Particle;

template<typename ... Args>
void sync(Particle<Args...> *pool)
{
	typedef Particle<Args...> particle_type;
	typedef typename particle_type::mesh_type mesh_type;
	typedef typename particle_type::Point_s value_type;

	MPI_Comm global_comm = GLOBAL_COMM.comm();

	MPI_Barrier(global_comm);

	auto const & g_array = pool->mesh().global_array_;
	if (g_array.send_recv_.size() == 0)
	{
		return;
	}
	VERBOSE << "update ghosts (particle pool) ";

	int num_of_neighbour = g_array.send_recv_.size();

	MPI_Request requests[num_of_neighbour * 2];

	std::vector<std::vector<value_type>> buffer(num_of_neighbour * 2);

	int count = 0;

	for (auto const & item : g_array.send_recv_)
	{

		size_t num = 0;
		for (auto s : pool->mesh().select_inner(item.send_begin, item.send_end))
		{
			num += pool->get(s).size();
		}

		buffer[count].resize(num);

		num = 0;

		for (auto s : pool->mesh().select_inner(item.send_begin, item.send_end))
		{
			for (auto const & p : pool->get(s))
			{
				buffer[count][num] = p;
				++num;
			}
		}

		MPI_Isend(&buffer[count][0], buffer[count].size() * sizeof(value_type),
		MPI_BYTE, item.dest, item.send_tag, global_comm, &requests[count]);
		++count;

	}

	for (auto const & item : g_array.send_recv_)
	{
		pool->remove(pool->mesh().select_outer(item.recv_begin, item.recv_end));

		MPI_Status status;

		MPI_Probe(item.dest, item.recv_tag, global_comm, &status);

		// When probe returns, the status object has the size and other
		// attributes of the incoming message. Get the size of the message
		int mem_size = 0;
		MPI_Get_count(&status, MPI_BYTE, &mem_size);

		if (mem_size == MPI_UNDEFINED)
		{
			RUNTIME_ERROR("Update Ghosts Particle fail");
		}
		buffer[count].resize(mem_size / sizeof(value_type));

		MPI_Irecv(&buffer[count][0], buffer[count].size() * sizeof(value_type),
		MPI_BYTE, item.dest, item.recv_tag, global_comm, &requests[count]);
		++count;
	}

	MPI_Waitall(num_of_neighbour, requests, MPI_STATUSES_IGNORE);

	auto cell_buffer = pool->create_child();
	for (int i = 0; i < num_of_neighbour; ++i)
	{
		typename mesh_type::coordinates_type xmin, xmax, extents;

		std::tie(xmin, xmax) = pool->mesh().get_extents();

		bool flag = true;
		for (int n = 0; n < 3; ++n)
		{
			if (g_array.send_recv_[i].recv_begin[n]
					< pool->mesh().global_begin_[n])
			{
				extents[n] = xmin[n] - xmax[n];
			}
			else if (g_array.send_recv_[i].recv_begin[n]
					>= pool->mesh().global_end_[n])
			{
				extents[n] = xmax[n] - xmin[n];
			}
			else
			{
				extents[n] = 0;
			}

		}

		if (extents[0] != 0.0 || extents[1] != 0.0 || extents[2] != 0.0)
		{
			for (auto p : buffer[num_of_neighbour + i])
			{
				p.x += extents;
				cell_buffer.push_back(std::move(p));
			}
		}
		else
		{
			std::copy(buffer[num_of_neighbour + i].begin(),
					buffer[num_of_neighbour + i].end(),
					std::back_inserter(cell_buffer));
		}

	}

	MPI_Barrier(global_comm);

	pool->add(&cell_buffer);

}
}
// namespace simpla

#endif /* CORE_PARTICLE_SYNC_PARTICLE_H_ */
