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
#include "../utilities/memory_pool.h"

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

	int num_of_neighbour = g_array.send_recv_.size();

	// recv
	std::vector<MPI_data_pack_s> send_buffer(num_of_neighbour);
	std::vector<MPI_data_pack_s> recv_buffer(num_of_neighbour);

	for (int i=0,ie= g_array.send_recv_.size();i<ie;++i)
	{

		auto const & item=g_array.send_recv_[i];

		size_t num=0;

		auto range= pool->mesh.SelectInner(ParticlePool<TM, TParticle>::IForm,item.send_begin , item.send_end);

		for (auto s :range)
		{
			num+=pool->get(s).size();
		}

		int mem_size=num*sizeof(value_type);

		send_buffer[i].buffer = MEMPOOL.allocate_byte_shared_ptr(num*sizeof(value_type));
		recv_buffer[i].data_type;
	}

	MPI_Request requests[num_of_neighbour * 2];

	int req_count = 0;
	for (auto const & item : g_array.send_recv_)
	{
		pool->Remove(pool->mesh.SelectOuter(ParticlePool<TM, TParticle>::IForm, item.recv_begin, item.recv_end));

		MPI_Status status;

		MPI_Probe(item.dest, item.recv_tag, comm, &status);

		// When probe returns, the status object has the size and other
		// attributes of the incoming message. Get the size of the message
		int mem_size = 0;

		MPI_Get_count(&status, MPI_BYTE, &mem_size);

		if(mem_size==MPI_UNDEFINED)
		{
			RUNTIME_ERROR("Update Ghosts Particle fail");
		}

		recv_buffer[req_count]=MEMPOOL.allocate_byte_shared_ptr(mem_size);

		MPI_Irecv( recv_buffer[req_count].get(),mem_size, MPI_BYTE, item.dest, item.recv_tag, comm, &requests[req_count]);

		++req_count;
	}

	// send

	for (auto const & item : g_array.send_recv_)
	{

		size_t num=0;

		auto range= pool->mesh.SelectInner(ParticlePool<TM, TParticle>::IForm,item.send_begin , item.send_end);

		for (auto s :range)
		{
			num+=pool->get(s).size();
		}

		int mem_size=num*sizeof(value_type);

		send_buffer[req_count] = MEMPOOL.allocate_byte_shared_ptr(num*sizeof(value_type));

		num=0;

//		for (auto s :range)
//		{
//			for (auto const & p : pool->get(s))
//			{
//				send_buffer[req_count][num]=p;
//				++num;
//			}
//		}

		MPI_Isend( send_buffer[req_count].get(), mem_size,MPI_BYTE, item.dest, item.send_tag, comm, &requests[req_count]);

		++req_count;

	}

	MPI_Waitall(num_of_neighbour*2, requests, MPI_STATUSES_IGNORE);

//	auto cell_buffer = pool->create_child();
//	for (int i = 0; i < num_of_neighbour; ++i)
//	{
//		typename TM::coordinates_type xmin,xmax,extents;
//
//		std::tie(xmin,xmax)=pool->mesh.get_extents();
//
//		bool flag=true;
//		for(int n=0;n<3;++n)
//		{
//			if(g_array.send_recv_[i].recv_begin[n]<pool->mesh.global_begin_[n])
//			{
//				extents[n]=xmin[n]-xmax[n];
//			}
//			else if(g_array.send_recv_[i].recv_begin[n]>=pool->mesh.global_end_[n])
//			{
//				extents[n]=xmax[n]-xmin[n];
//			}
//			else
//			{	extents[n]=0;}
//
//		}
//
//		if( extents[0]!=0.0 || extents[1]!=0.0|| extents[2]!=0.0)
//		{
//			for(auto p:recv_buffer[num_of_neighbour + i])
//			{
//				p.x+=extents;
//				cell_buffer.push_back(std::move(p));
//			}
//		}
//		else
//		{
//			auto * ptr=reinterpret_cast<value_type const *>(recv_buffer[num_of_neighbour + i].get());
//
//			std::copy(ptr, ptr+recv_buffer_size[num_of_neighbour + i], std::back_inserter(cell_buffer));
//		}
//
//	}
	GLOBAL_COMM.barrier();

//	pool->Add(&cell_buffer);

#endif
}
}
		// namespace simpla

#endif /* PARTICLE_UPDATE_GHOSTS_H_ */
