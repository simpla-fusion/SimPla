/**
 * \file block_range.h
 *
 * \date    2014年9月5日  下午2:57:34 
 * \author salmon
 */

#ifndef BLOCK_RANGE_H_
#define BLOCK_RANGE_H_

#include <cstdbool>
#include <cstddef>

#include "../utilities/log.h"

namespace simpla
{

struct split_tag
{
};

/**
 *   is compatible with TBB block_range  1D,2D,3D
 */

template<typename T>
class BlockRange
{
public:
	typedef T value_type;
	typedef BlockRange<value_type> this_type;

	//! Type for size of a range
	typedef std::size_t size_type;

	/// \ingroup conecpt_range
	BlockRange(this_type const &)  //! Copy constructor
	:
			i_e_(), i_b_(),

			o_e_(i_e_),

			o_b_(i_b_),

			g_e_(i_e_),

			g_b_(i_b_),

			grainsize_(1),

			ghostwidth_(0)
	{
	}

	BlockRange(this_type & r, split_tag) //! Split range r into two subranges.
	:
			i_e_(r.i_e_), i_b_(do_split(r, split_tag())),

			o_e_(r.i_e_),

			o_b_(i_b_ - r.ghostwidth_),

			g_e_(r.g_e_),

			g_b_(r.g_e_),

//			grainsize_(grainsize),

			ghostwidth_(r.ghostwidth_)

	{
	}

	BlockRange(value_type b, value_type e, size_type grainsize = 1,
			size_type ghost_width = 2) //! Split range r into two subranges.
	:
			i_e_(e), i_b_(b),

			o_e_(e + ghost_width),

			o_b_(b - ghost_width),

			g_e_(e),

			g_b_(b),

			grainsize_(grainsize), ghostwidth_(ghost_width)
	{
	}
	~BlockRange()  //! Destructor
	{

	}

	bool empty() const //! True if range is empty
	{
		return i_e_ <= i_b_;
	}

	bool is_divisible() const //!True if range can be partitioned into two subranges
	{
		return grainsize_ > size();
	}

	struct iterator
	{
		value_type v_;
		iterator(value_type const &v) :
				v_(v)
		{

		}
		~iterator()
		{

		}
		value_type operator*() const
		{
			return v_;
		}

		iterator & operator++()
		{
			++v_;
			return *this;
		}
		iterator operator++(int) const
		{
			this_type res(v_);
			++res;
			return std::move(res);
		}

		bool operator==(iterator const & that) const
		{
			return v_ == that.v_;
		}

		bool operator!=(iterator const & that) const
		{
			return v_ != that.v_;
		}

	};
/// \ingroup container_range

	iterator begin() const
	{
		return std::move(iterator(i_b_));
	}

	iterator end() const
	{
		return std::move(iterator(i_e_));
	}
	size_type size() const
	{
		return size_type(end() - begin());
	}

	iterator outter_begin() const
	{
		return o_b_;
	}

	iterator outter_end() const
	{
		return o_e_;
	}

	size_type memory_size() const
	{
		return size_type(end() - begin());
	}

	iterator global_begin() const
	{
		return g_b_;
	}

	iterator global_end() const
	{
		return g_e_;
	}
	size_type global_size() const
	{
		return size_type(global_end() - global_begin());
	}

	size_type hash(value_type const &i) const
	{
		return i - i_b_;
	}

	size_type max_hash() const
	{
		return i_e_ - i_b_;
	}
private:
	value_type i_e_, i_b_;
	value_type o_e_, o_b_;
	value_type g_e_, g_b_;
	size_type grainsize_;
	size_type ghostwidth_;

	static value_type do_split(this_type & r, split_tag)
	{
		ASSERT(r.is_divisible());

		value_type m = r.i_b_ + (r.i_e_ - r.i_b_) / 2u;
		r.i_e_ = m;
		r.o_e_ = m + r.ghostwidth_;
		return m;
	}

}
;
//template<typename T, unsigned int N>
//class BlockRange<nTuple<N, T>>
//{
//public:
//
//	typedef T const_iterator;
//
//	typedef T value_type;
//	//! Type for size of a range
//	typedef std::size_t size_type;
//	static constexpr unsigned ndims = N;
//
//	typedef BlockRange<T, N> this_type;
//
//	typedef nTuple<N, BlockRange<T, 1>> base_type;
//
//	/// \ingroup conecpt_range
//	BlockRange(this_type const & r)  //! Copy constructor
//	:
//			base_type(r)
//	{
//
//	}
//
//	BlockRange(this_type & r, split_tag)  //! Split range r into two subranges.
//	{
//
//	}
//
//	BlockRange(nTuple<N, value_type> const & b, nTuple<N, value_type> const & e,
//			size_type grainsize = 1, size_type ghost_width = 2) //! Split range r into two subranges.
//	{
//
//	}
//	BlockRange(nTuple<N, value_type> const & b, nTuple<N, value_type> const & e,
//			nTuple<N, size_type> const & grainsize,
//			nTuple<N, size_type> ghost_width) //! Split range r into two subranges.
//
//	{
//	}
//	~BlockRange() //! Destructor
//	{
//
//	}
//
//	bool empty() const; //! True if range is empty
//
//	bool is_divisible() const //!True if range can be partitioned into two subranges
//
//	/// \ingroup BlockRange
//private:
//	unsigned int ndims = 3;
//
//};

}// namespace simpla

#endif /* BLOCK_RANGE_H_ */
