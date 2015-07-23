/**
 * @file block.h
 * @author salmon
 * @date 2015-07-22.
 */

#ifndef SIMPLA_BLOCK_H
#define SIMPLA_BLOCK_H

#include "../../gtl/primitives.h"
#include "../mesh_traits.h"
#include "mesh_id.h"

namespace simpla
{
namespace tags
{
struct split;
}
namespace mesh
{


/**
 * Class Block a NDIMS-dimensional box [min,max) in the index space
 *
 */
template<int NDIMS>
struct Block
{
	static constexpr int ndims = NDIMS;


	typedef MeshID<ndims, DEFAULT_MESH_LEVEL> m;

	typedef nTuple<Real, ndims> point_type;

	typedef typename m::id_type id_type;

	typedef typename m::id_tuple id_tuple;

	typedef typename m::index_type index_type;

	typedef typename m::difference_type difference_type;

	typedef typename m::index_tuple index_tuple;


private:

	id_type m_min_, m_max_;

public:

	Block(index_tuple i_min, index_tuple i_max) :
			m_min_(m::pack(i_min)), m_max_(m::pack(i_max))
	{
	}

	Block(Block const &other) :
			m_min_(other.m_min_), m_max_(other.m_max_)
	{
	}

	Block(Block &other, tags::split)
	{

	};

	virtual ~Block()
	{
	}

	void swap(Block &other)
	{
		std::swap(m_min_, other.m_min_);
		std::swap(m_max_, other.m_max_);

	}

public:

	/**
	 *   return index box [min,max)
	 */
	constexpr std::tuple<index_tuple, index_tuple> box() const
	{
		return std::make_tuple(m::unpack(m_min_), m::unpack(m_max_));
	};


	/**
	 * check whether a given index lies within the bounds of the block
	 */

	constexpr bool in_box(id_type s) const
	{
		return m::in_box(s, m_min_, m_max_);
	}

	/**
	 * check whether a given box  lies within the bounds of the block
	 */
	constexpr bool in_box(Block<ndims> const &b) const
	{
		return m::in_box(b.m_min_, m_min_, m_max_) &&
				m::in_box(b.m_max_, m_min_, m_max_);
	}


	/**
	 *  calculate the number of valid indices of IFORM represents by the block
	 */
	template<int IFORM, int LEVEL = DEFAULT_MESH_LEVEL>
	constexpr size_t size() const { MeshID<ndims, LEVEL>::max_hash<IFORM>(m_min_, m_max_); }


	/**
	 *  return a range used to traversing valid indices of IFORM
	 */
	template<int IFORM, int LEVEL = DEFAULT_MESH_LEVEL>
	typename MeshID<ndims, LEVEL>::range_type range() const
	{
		return typename MeshID<ndims, LEVEL>::range_type(m_min_, m_max_, IFORM);
	}

	/**
	 *  given an index, calculate the offset into the block
	 */
	template<int IFORM, int LEVEL = DEFAULT_MESH_LEVEL>
	constexpr size_t hash(id_type s) const { return MeshID<ndims, LEVEL>::hash(s, m_min_, m_max_); }


};


} //namespace mesh
} //namespace simpla

#endif //SIMPLA_BLOCK_H
