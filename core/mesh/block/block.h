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
	
	m::index_tuple dimensions() const
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
	

	auto box() const
	DECL_RET_TYPE(std::forward_as_tuple(m_id_local_min_, m_id_local_max_))
	
	template<typename T>
	bool in_box(T const &x) const
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
	range_type <IFORM> range() const
	{
		return range_type(*this);
	}

	template<size_t IFORM>
	struct range_type
	{

		struct const_iterator;

	private:

		Block m_block_;
	public:

		range_type(Block const &block) :
				m_block_(block)
		{
		}


		range_type(range_type const &other) :
				m_block_(other.m_block_)
		{

		}

		~ range_type()
		{

		}

		range_type &operator=(range_type const &other)
		{
			range_type(other).swap(*this);
			return *this;
		}

		range_type operator&(range_type const &other) const
		{
			return *this;
		}

		void swap(range_type &other)
		{
			std::swap(m_block_, other.m_block_);

		}

		const_iterator begin() const
		{
			return const_iterator(m_block_.m_id_local_min_ | m::m_iform_to_shift_[IFORM],
					m_block_.m_id_local_max_ | m::m_iform_to_shift_[IFORM],
					m_block_.m_id_local_min_ | m::m_iform_to_shift_[IFORM]);
		}

		const_iterator end() const
		{
			return ++const_iterator(m_block_.m_id_local_min_ | m::m_iform_to_shift_[IFORM],
					m_block_.m_id_local_max_ | m::m_iform_to_shift_[IFORM],
					m::inverse_rotate(m_block_.m_id_local_max_ | m::m_iform_to_shift_[IFORM] - (m::_DA << 1)));
		}


		constexpr size_t size() const
		{
			return NProduct(m::unpack_index(m_block_.m_id_local_max_ - m_block_.m_id_local_min_))
					* m::m_iform_to_num_of_ele_in_cell_[IFORM];
		}


		struct const_iterator : public std::iterator<typename std::bidirectional_iterator_tag,
				id_type, difference_type>
		{
		private:


			typedef MeshID<NDIMS, LEVEL> m;

			typedef typename m::id_type id_type;

			id_type m_min_, m_max_, m_self_;
		public:


			const_iterator(id_type const &min, id_type const &max,
					id_type const &self) :
					m_min_(min), m_max_(max), m_self_(self)
			{
			}

			const_iterator(id_type const &min, id_type const &max) :
					m_min_(min), m_max_(max), m_self_(min)
			{
			}

			const_iterator(const_iterator const &other) :
					m_min_(other.m_min_), m_max_(other.m_max_), m_self_(
					other.m_self_)
			{
			}

			~const_iterator()
			{

			}


			bool operator==(const_iterator const &other) const
			{
				return m_self_ == other.m_self_;
			}

			bool operator!=(const_iterator const &other) const
			{
				return m_self_ != other.m_self_;
			}

			id_type const &operator*() const
			{
				return m_self_;
			}

		private:

			index_type carray_(id_type *self, id_type min, id_type max,
					index_type flag = 0)
			{

				auto div = std::div(
						static_cast<long>(*self + flag * (m::_D << 1) + max
								- min * 2), static_cast<long>(max - min));

				*self = static_cast<id_type>(div.rem + min);

				return div.quot - 1L;
			}

			m::index_type carray(id_type *self, id_type xmin, id_type xmax,
					m::index_type flag = 0)
			{
				m::id_tuple idx, min, max;

				idx = m::unpack(*self);
				min = m::unpack(xmin);
				max = m::unpack(xmax);

				flag = carray_(&idx[0], min[0], max[0], flag);
				flag = carray_(&idx[1], min[1], max[1], flag);
				flag = carray_(&idx[2], min[2], max[2], flag);

				*self = m::pack(idx) | (std::abs(flag) << (m::FULL_DIGITS - 1));
				return flag;
			}

		public:
			void next()
			{
				m_self_ = m::rotate(m_self_);
				if (m::sub_index(m_self_) == 0)
				{
					carray(&m_self_, m_min_, m_max_, 1);
				}

			}

			void prev()
			{
				m_self_ = m::inverse_rotate(m_self_);
				if (m::sub_index(m_self_) == 0)
				{
					carray(&m_self_, m_min_, m_max_, -1);
				}
			}

			const_iterator &operator++()
			{
				next();
				return *this;
			}

			const_iterator &operator--()
			{
				prev();

				return *this;
			}

			const_iterator operator++(int)
			{
				const_iterator res(*this);
				++(*this);
				return std::move(res);
			}

			const_iterator operator--(int)
			{
				const_iterator res(*this);
				--(*this);
				return std::move(res);
			}

		}; //struct const_iterator

	}; // struct range_type
}; //struct Block



}// namespace simpla

#endif //SIMPLA_MESH_BLOCK_BLOCK_H_
