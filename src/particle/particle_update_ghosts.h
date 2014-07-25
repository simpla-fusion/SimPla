/*
 * particle_update_ghosts.h
 *
 *  created on: 2014-6-15
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
void update_ghosts(ParticlePool<TM, TParticle> *pool)
{
#ifdef USE_MPI

	GLOBAL_COMM.barrier();
	auto const & g_array = pool->mesh.global_array_;

	if (g_array.send_recv_.size() == 0)
	{
		return;
	}

	VERBOSE << "update ghosts (particle pool) ";

	typedef ParticlePool<TM, TParticle> pool_type;

	typedef typename pool_type::particle_type value_type;

	MPI_Comm comm = GLOBAL_COMM.comm();

	MPIDataType<value_type> dtype;

	int num_of_neighbour = g_array.send_recv_.size();

	MPI_Request requests[num_of_neighbour * 2];

	std::vector<std::vector<value_type>> buffer(num_of_neighbour * 2);

	int count = 0;

	for (auto const & item : g_array.send_recv_)
	{
		buffer[count].clear();

		for (auto s : pool->mesh.SelectInner(ParticlePool<TM, TParticle>::IForm,item.send_begin , item.send_end))
		{
			for (auto const & p : pool->get(s))
			{
				buffer[count].push_back(p);
			}
		}

		MPI_Isend(&buffer[count][0], buffer[count].size(), dtype.type(), item.dest, item.send_tag, comm,
				&requests[count]);
		++count;

	}

	for (auto const & item : g_array.send_recv_)
	{
		pool->Remove(pool->mesh.SelectOuter(ParticlePool<TM, TParticle>::IForm, item.recv_begin, item.recv_end));

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

	auto cell_buffer = pool->create_child();
	for (int i = 0; i < num_of_neighbour; ++i)
	{
		typename TM::coordinates_type xmin,xmax,extents;

		std::tie(xmin,xmax)=pool->mesh.get_extents();

		bool flag=true;
		for(int n=0;n<3;++n)
		{
			if(g_array.send_recv_[i].recv_begin[n]<pool->mesh.global_begin_[n])
			{
				extents[n]=xmin[n]-xmax[n];
			}
			else if(g_array.send_recv_[i].recv_begin[n]>=pool->mesh.global_end_[n])
			{
				extents[n]=xmax[n]-xmin[n];
			}
			else
			{	extents[n]=0;}

		}

		if( extents[0]!=0.0 || extents[1]!=0.0|| extents[2]!=0.0)
		{
			for(auto p:buffer[num_of_neighbour + i])
			{
				p.x+=extents;
				cell_buffer.push_back(std::move(p));
			}
		}
		else
		{
			std::copy(buffer[num_of_neighbour + i].begin(), buffer[num_of_neighbour + i].end(),
					std::back_inserter(cell_buffer));
		}

	}
	GLOBAL_COMM.barrier();

	pool->Add(&cell_buffer);

#endif
}
}
		// namespace simpla

#endif /* PARTICLE_UPDATE_GHOSTS_H_ */
