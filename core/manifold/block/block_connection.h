/**
 * @file block_connection.h
 * Created by salmon on 7/8/15.
 */
#ifndef SIMPLA_BLOCK_CONNECTION_H
#define SIMPLA_BLOCK_CONNECTION_H


#include <type_traits>
#include "block.h"
#include "block_layout.h"

namespace simpla
{
struct lock;


namespace mesh
{
/**
 *  Represents map relation between two block in same/different index space
 *
 *  @pre size(send_block)=size(recv_block)
 */
template<int NDIMS>
struct BlockConnection
{

	typedef Block<NDIMS> block_type;

	typedef BlockLayout<NDIMS> block_layout_type;

	typedef typename block_layout_type::id_tag_type id_tag_type;

private:
	id_tag_type m_id_send_;
	id_tag_type m_id_recv_;
	block_type m_send_block_;
	block_type m_recv_block_;

public:
	id_tag_type id_send() const
	{
		return m_id_send_;
	}

	id_tag_type id_recv() const
	{
		return m_id_recv_;
	}

	const block_type &send_block() const
	{
		return m_send_block_;
	}

	const block_type &recv_block() const
	{
		return m_recv_block_;
	}


};

} // namespace manifold
} // namespace simpla
#endif //SIMPLA_BLOCK_CONNECTION_H
