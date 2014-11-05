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
#include "../utilities/range.h"
namespace simpla
{

/**
 *   is compatible with TBB block_range  1D,2D,3D
 */

template<typename T>
class BlockRange
{
public:
	typedef T index_type;
	typedef BlockRange<index_type> this_type;

	//! Type for size of a range
	typedef size_t size_type;

	BlockRange(index_type b = 0, index_type e = 1, size_type grainsize = 1) :
			i_e_(e), i_b_(b), grainsize_(grainsize)
	{
	}
	/// \ingroup conecpt_range

	//! Copy constructor
	BlockRange(this_type const & that) :
			i_e_(that.i_e_), i_b_(that.i_b_), grainsize_(that.grainsize_)
	{
	}
	//! Split range r into two subranges.
	BlockRange(this_type & r, split_tag) :
			i_e_(r.i_e_), i_b_(do_split(r, split_tag())), grainsize_(
					r.grainsize_)
	{
	}

	//! Split range r into two subranges.

	~BlockRange()  //! Destructor
	{

	}

	void swap(this_type & r)
	{
		std::swap(i_e_, r.i_e_);
		std::swap(i_b_, r.i_b_);

		std::swap(grainsize_, r.grainsize_);

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
		index_type v_;

		iterator(iterator const& that) :
				v_(that.v_)
		{

		}
		iterator(index_type const &v) :
				v_(v)
		{

		}
		~iterator()
		{

		}
		size_type operator*() const
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

	size_type hash(index_type const &i) const
	{
		return i - i_b_;
	}

	size_type max_hash() const
	{
		return i_e_ - i_b_;
	}

private:
	index_type i_e_, i_b_;

	size_type grainsize_;

	static index_type do_split(this_type & r, split_tag)
	{
		ASSERT(r.is_divisible());

		index_type m = r.i_b_ + (r.i_e_ - r.i_b_) / 2u;
		r.i_e_ = m;
		return m;
	}

}
;

}
// namespace simpla

#endif /* BLOCK_RANGE_H_ */
