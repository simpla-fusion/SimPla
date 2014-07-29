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

	auto const & g_array = pool->mesh.global_array_;

	if (g_array.send_recv_.size() == 0)
	{
		return;
	}

	VERBOSE << "update ghosts (particle pool) ";

	typedef ParticlePool<TM, TParticle> pool_type;

	typedef typename pool_type::particle_type value_type;

	int num_of_neighbour = g_array.send_recv_.size();

	// recv
	std::vector<std::tuple<int, // dest;
	        int, // send_tag;
	        int, // recv_tag;
	        int, // send buffer begin;
	        int  // send buffer size;
	        > > info;

	int pos = 0;

	for (int i = 0, ie = g_array.send_recv_.size(); i < ie; ++i)
	{

		auto const & item = g_array.send_recv_[i];

		size_t num = 0;

		auto range = pool->mesh.SelectInner(ParticlePool<TM, TParticle>::IForm, item.send_begin, item.send_end);

		for (auto s : range)
		{
			num += pool->get(s).size();
		}

		int mem_size = num * sizeof(value_type);

		info.emplace_back(std::make_tuple(item.dest, item.send_tag, item.recv_tag, pos, mem_size));

		pos += mem_size;
	}
	auto send_buffer = MEMPOOL.allocate_byte_shared_ptr(pos);

	std::shared_ptr<ByteType> recv_buffer;
	int buffer_size;

	std::tie(recv_buffer, buffer_size) = gather_unorder(send_buffer.get(), info);

	value_type const * data = reinterpret_cast<value_type*>(recv_buffer.get());

	int number = buffer_size / sizeof(value_type);

	auto cell_buffer = pool->create_child();

	for (int s = 0; s < number; ++s)
	{
		cell_buffer.push_back(data[s]);
	}

	pool->Add(&cell_buffer);

}
}
// namespace simpla

#endif /* PARTICLE_UPDATE_GHOSTS_H_ */
