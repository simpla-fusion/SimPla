/**
 * \file sp_range.h
 *
 * \date    2014年8月27日  上午9:49:48 
 * \author salmon
 */

#ifndef SP_RANGE_H_
#define SP_RANGE_H_
#include <functional>

#include "iterator.h"
namespace simpla
{
/**
 *  @addtogroup range Range
 *  @{
 *  @brief  Range
 *  "
 *    A Range can be recursively subdivided into two parts
 *  - Summary Requirements for type representing a recursively divisible set of values.
 *  - Requirements The following table lists the requirements for a Range type R.
 *  "  --TBB
 */
template<typename TI>
class Range
{
public:

	// concept Range
	typedef TI iterator; //! Iterator type for range
	typedef Range<iterator> this_type;
	typedef typename std::result_of<std::minus<iterator>(iterator, iterator)>::type diff_type;

	Range(iterator const & b, iterator const & e, size_t grainsize = 0) :
			b_(b), e_(e), grainsize_(grainsize == 0 ? (e - b) : grainsize)
	{
	}

	//! Copy constructor
	Range(this_type const & r) :
			b_(r.b_), e_(r.e_), grainsize_(r.grainsize_)
	{

	}

	~Range()
	{
	}

	//! True if range is empty
	bool empty() const
	{
		return e_ == b_;
	}
	//!True if range can be partitioned into two subranges
	bool is_divisible() const
	{
		return e_ - b_ > grainsize_ * 2;
	}

	//**************************************************************************************************

	// concept block range
	// Additional Requirements on a Container Range R
	//! First item 	in range
	iterator const & begin() const
	{
		return b_;
	}
	//! One past last item	in range
	iterator const & end() const
	{
		return e_;
	}
	//!	Grain size
	size_t grainsize() const
	{
		return grainsize_;
	}

	diff_type size() const
	{
		return e_ - b_;
	}

private:
	iterator b_, e_;

	diff_type grainsize_;

	std::tuple<this_type, this_type> do_split(this_type & r) const
	{
		iterator m = b_ + (e_ - b_) / 2u;

		return std::forward_as_tuple(Range(b_, m, grainsize_),
				Range(m, e_, grainsize_));
	}

};

template<typename TI>
typename Range<TI>::const_iterator begin(Range<TI> const & range)
{
	return range.begin();
}

template<typename TI>
typename Range<TI>::const_iterator end(Range<TI> const & range)
{
	return range.end();
}

template<typename TI>
typename Range<TI>::const_iterator const_begin(Range<TI> const& range)
{
	return range.begin();
}

template<typename TI>
typename Range<TI>::const_iterator const_end(Range<TI> const& range)
{
	return range.end();
}
template<typename TI>
typename Range<TI>::size_type size(Range<TI> const & range)
{
	return range.size();
}

template<typename TI>
Range<TI> make_range(TI const & b, TI const &e)
{
	return std::move(Range<TI>(b, e));
}
/** @}*/
}  // namespace simpla

#endif /* SP_RANGE_H_ */
