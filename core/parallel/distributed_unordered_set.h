/**
 * @file distributed_unordered_set.h
 * @author salmon
 * @date 2015-10-15.
 */

#ifndef SIMPLA_DISTRIBUTED_UNORDERED_SET_H
#define SIMPLA_DISTRIBUTED_UNORDERED_SET_H


#include "distributed_object.h"
#include "mpi_comm.h"
#include "mpi_aux_functions.h"
#include "mpi_update.h"
#include "../gtl/containers/unordered_set.h"

namespace simpla
{


//namespace parallel
template<typename TV, typename ...Others>
struct DistributedUnorderedSet
		: public UnorderedSet<TV, Others...>, public DistributedObject
{

	DistributedUnorderedSet() { };

	virtual ~DistributedUnorderedSet() { };


	virtual void sync();

	virtual void async();

	virtual void wait() const;

	virtual bool is_ready() const;

	DataType datatype;

	struct connection_s
	{
		nTuple<int, 3> coord_shift;

		range_type send_range;

		range_type recv_range;
	};

	std::vecotr<connection_s> m_connection_;

};

template<typename TV, typename ...Others>
void DistributedUnorderedSet<TV, Others...>::sync()
{
	for (auto const &item : m_connection_)
	{
		mpi_send_recv_buffer_s send_recv_buffer;


		std::tie(send_recv_buffer.dest, send_recv_buffer.send_tag,
				send_recv_buffer.recv_tag) = GLOBAL_COMM.make_send_recv_tag(global_id(),
				&item.coord_shift[0]);

		//   collect send data

		send_recv_buffer.send_size = container_type::size_all(item.send_range);

		send_recv_buffer.send_data = sp_alloc_memory(send_recv_buffer.send_size * send_recv_buffer.datatype.size());

		value_type *data = reinterpret_cast<value_type *>(send_recv_buffer.send_data.get());

		// FIXME need parallel optimize
		for (auto const &key : send_range)
		{
			for (auto const &p : container_type::operator[](key))
			{
				*data = p;
				++data;
			}
		}

		send_recv_buffer.recv_size = 0;
		send_recv_buffer.recv_data = nullptr;

		m_send_recv_buffer_.push_back(std::move(send_recv_buffer));

		container_type::erase(item.recv_range);
	}

	sync_update_varlength(&m_send_recv_buffer_, &(m_mpi_requests_));
}

template<typename TV, typename ...Others>
void DistributedUnorderedSet<TV, Others...>::wait() const
{
	// FIXME need parallel optimize
	for (auto const &item : m_send_recv_buffer_)
	{
		value_type *data = reinterpret_cast<value_type *>(item.recv_data.get());

		container_type::insert(data, data + item.recv_size);
	}
	m_send_recv_buffer_.clear();

}
}//namespace simpla
#endif //SIMPLA_DISTRIBUTED_UNORDERED_SET_H
