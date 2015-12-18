/**
 * @file distributed_unordered_set.h
 * @author salmon
 * @date 2015-10-15.
 */

#ifndef SIMPLA_DISTRIBUTED_UNORDERED_SET_H
#define SIMPLA_DISTRIBUTED_UNORDERED_SET_H

#include "distributed.h"
#include "DistributedObject.h"

#include "../gtl/containers/unordered_set.h"

namespace simpla
{
template<typename ...> struct Distributed;

//namespace parallel
template<typename TV, typename ...Others, typename TRange>
struct Distributed<UnorderedSet<TV, Others...>, TRange>
		: public UnorderedSet<TV, Others...>, public DistributedObject
{
private:

	typedef UnorderedSet<TV, Others...> base_type;
	typedef Distributed<UnorderedSet<TV, Others...>, TRange> this_type;
	typedef TRange range_type;

public:

	Distributed() { };

	Distributed(this_type const &other) : base_type(other) { };

	virtual ~Distributed() { };

	void swap(this_type &other) { base_type::swap(other) };

	virtual void sync();

	virtual void wait();

	template<typename TConnection> void deploy(TConnection const &conns);

	template<typename Hash> void rehash(Hash const &hasher);

private:
	struct connection_node
	{
		nTuple<int, 3> coord_offset;
		range_type send_range;
		range_type recv_range;
		size_t send_size;
		size_t recv_size;
		std::shared_ptr<value_type> send_buffer;
		std::shared_ptr<value_type> recv_buffer;
	};

	std::vector<connection_node> m_connections_;

};

template<typename TV, typename ...Others>
void Distributed<UnorderedSet<TV, Others...>>::sync()
{
	auto d_type = traits::datatype::template create<TV>::create();
	for (auto &item : m_connections_)
	{

		item.send_size = base_type::size_all(item.send_range);

		item.send_buffer = sp_alloc_memory(item.send_size * sizeof(value_type));

		value_type *data = item.send_buffer.get();

		// FIXME need parallel optimize
		for (auto const &key : item.send_range)
		{
			for (auto const &p : base_type::operator[](key))
			{
				*data = p;
				++data;
			}
		}

		item.recv_size = 0;
		item.recv_buffer = nullptr;
		base_type::erase(item.recv_range);

		DistributedObject::add_link_send(&item.coord_offset[0], item.send_size, d_type, &item.send_buffer);
		DistributedObject::add_link_recv(&item.coord_offset[0], item.recv_size, d_type, &item.recv_buffer);

	}

	DistributedObject::sync();
}

template<typename TV, typename ...Others>
void Distributed<UnorderedSet<TV, Others...>>::wait()
{
	DistributedObject::wait();

	// FIXME need parallel optimize
	for (auto const &item : m_connections_)
	{
		TV const *data = item.recv_buffer.get();

		base_type::insert(data, data + item.recv_size);

		item.recv_buffer = nullptr;
		item.recv_size = 0;
	}


}

template<typename TV, typename ...Others>
template<typename TCoonection>
void Distributed<UnorderedSet<TV, Others...>>::deploy(TCoonection const &conns)
{
	for (auto const &item:conns)
	{
		m_connections_.emplace_back(item.coord_offset, item.send_range, item.recv_range,
				0, 0, std::shared_ptr<void>(), std::shared_ptr<void>());
	}
};


template<typename TV, typename ...Others>
template<typename Hash>
void Distributed<UnorderedSet<TV, Others...>>::rehash(Hash const &hasher)
{
	wait();
	base_type::rehash(hasher);
	sync();

};
}//namespace simpla
#endif //SIMPLA_DISTRIBUTED_UNORDERED_SET_H
