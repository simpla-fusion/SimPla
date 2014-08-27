/**
 * \file sp_range.h
 *
 * \date    2014年8月27日  上午9:49:48 
 * \author salmon
 */

#ifndef SP_RANGE_H_
#define SP_RANGE_H_
#include "sp_iterator.h"
namespace simpla
{
template<typename TI>
class Range
{
public:

	typedef TI iterator;
	typedef Range<iterator> this_type;

	iterator b_, e_;

	Range(iterator const & b, iterator const & e) :
			b_(b), e_(e)
	{
	}
	~Range()
	{
	}

	std::tuple<this_type, this_type> split() const
	{
		iterator m = (e_ - b_) / 2u + b_;

		return std::forward_as_tuple(Range(b_, m), Range(m, e_));
	}

	size_t size() const
	{
		return e_ - b_;
	}
	TI const & begin() const
	{
		return b_;
	}
	TI const & end() const
	{
		return e_;
	}

	TI & begin()
	{
		return b_;
	}
	TI & end()
	{
		return e_;
	}

};

template<typename TI>
Range<TI> make_range(TI const & b, TI const &e)
{
	return std::move(Range<TI>(b, e));
}

template<typename TI>
TI & begin(Range<TI> & range)
{
	return range.begin();
}

template<typename TI>
TI & end(Range<TI> & range)
{
	return range.end();
}

template<typename TI>
TI const& const_begin(Range<TI> const& range)
{
	return range.begin();
}

template<typename TI>
TI const& const_end(Range<TI> const& range)
{
	return range.end();
}
template<typename TI>
size_t size(Range<TI> const & range)
{
	return range.size();
}
template<typename TI>
std::tuple<Range<TI>, Range<TI>> split(Range<TI> const & range)
{

	return std::move(range.split());
}
}  // namespace simpla

#endif /* SP_RANGE_H_ */
