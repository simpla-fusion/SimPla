//
// Created by salmon on 15-7-2.
//

#ifndef SIMPLA_MESH_BLOCK_BLOCK_H_
#define SIMPLA_MESH_BLOCK_BLOCK_H_

#include "mesh_id.h"
#include "../../gtl/macro.h"
#include "../../gtl/primitives.h"
#include "../../gtl/ntuple.h"

#include "../../gtl/type_traits.h"

#include <vector>

namespace simpla
{




/**
 *  @brief Block represents a 'NDIMS'-dimensional 'LEVEL'th-level AMR mesh ;
 */

template<int NDIMS, size_t LEVEL>
struct Block
{


public:

	static constexpr int ndims = NDIMS;

	typedef MeshID<NDIMS, LEVEL> m;

	typedef Block<NDIMS, LEVEL> this_type;

	typedef typename m::id_type id_type;

	typedef typename m::id_tuple id_tuple;

	typedef typename m::index_type index_type;
	typedef typename m::index_tuple index_tuple;
	typedef typename m::range_type range_type;

/**
 *
 *   -----------------------------5
 *   |                            |
 *   |     ---------------4       |
 *   |     |              |       |
 *   |     |  ********3   |       |
 *   |     |  *       *   |       |
 *   |     |  *       *   |       |
 *   |     |  *       *   |       |
 *   |     |  2********   |       |
 *   |     1---------------       |
 *   0-----------------------------
 *
 *	5-0 = dimensions
 *	4-1 = e-d = ghosts
 *	2-1 = counts
 *
 *	0 = id_begin
 *	5 = id_end
 *
 *	1 = id_local_outer_begin
 *	4 = id_local_outer_end
 *
 *	2 = id_local_inner_begin
 *	3 = id_local_inner_end
 *
 *
 */

	id_type m_id_local_min_;

	id_type m_id_local_max_;

	id_type m_id_memory_min_;

	id_type m_id_memory_max_;


public:

	Block()
	{

	}


	Block(Block const &other) :

			m_id_local_min_(other.m_id_local_min_),

			m_id_local_max_(other.m_id_local_max_),

			m_id_memory_max_(other.m_id_memory_max_),

			m_id_memory_min_(other.m_id_memory_min_)
	{

	}

	~Block()
	{

	}

	Block &operator=(Block const &other)
	{
		Block(other).swap(*this);
		return *this;
	}

//	this_type operator&(this_type const &other) const
//	{
//		return *this;
//	}

	void swap(this_type &other)
	{
		std::swap(m_id_local_min_, other.m_id_local_min_);
		std::swap(m_id_local_max_, other.m_id_local_max_);
		std::swap(m_id_memory_max_, other.m_id_memory_max_);
		std::swap(m_id_memory_min_, other.m_id_memory_min_);
	}

	template<typename TI>
	void dimensions(TI const &d)
	{

		m_id_local_min_ = m::ID_ZERO;
		m_id_local_max_ = m_id_local_min_ + m::pack_index(d);

	}

	typename m::index_tuple dimensions() const
	{
		return m::unpack_index(m_id_local_max_ - m_id_local_min_);
	}

	template<size_t IFORM>
	size_t max_hash() const
	{
		return m::template max_hash<IFORM>(m_id_memory_min_,
				m_id_memory_max_);
	}

	size_t hash(id_type s) const
	{
		return m::hash(s, m_id_memory_min_, m_id_memory_max_);
	}

	template<size_t IFORM>
	id_type pack_relative_index(index_type i, index_type j,
			index_type k, index_type n = 0) const
	{
		return m::pack_index(nTuple<index_type, 3>({i, j, k}),
				m::template sub_index_to_id<IFORM>(n)) + m_id_local_min_;
	}

	nTuple<index_type, ndims + 1> unpack_relative_index(id_type s) const
	{
		nTuple<index_type, ndims + 1> res;
		res = m::unpack_index(s - m_id_local_min_);
		res[ndims] = m::sub_index(s);
		return std::move(res);
	}


	std::tuple<index_tuple, index_tuple> index_box() const
	{
		return std::make_tuple(m::unpack_index(m_id_local_min_),
				m::unpack_index(m_id_local_max_));
	}


//	auto box() const
//	DECL_RET_TYPE(std::forward_as_tuple(m_id_local_min_, m_id_local_max_))

	bool in_box(index_tuple const &x) const
	{
		auto b = m::unpack_index(m_id_local_min_);
		auto e = m::unpack_index(m_id_local_max_);
		return (b[1] <= x[1]) && (b[2] <= x[2]) && (b[0] <= x[0])  //
				&& (e[1] > x[1]) && (e[2] > x[2]) && (e[0] > x[0]);

	}

	bool in_box(id_type s) const
	{
		return in_box(m::unpack_index(s));
	}

	template<size_t IFORM>
	range_type range() const
	{
		return m::range_type(m_id_local_min_, m_id_local_max_);
	}

}; //struct Block



}// namespace simpla

#endif //SIMPLA_MESH_BLOCK_BLOCK_H_
