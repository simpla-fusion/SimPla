/**
 * \file range.h
 *
 * \date    2014年8月27日  上午9:49:48 
 * \author salmon
 */

#ifndef RANGE_H_
#define RANGE_H_
namespace simpla
{

class split
{
};

template<typename TI>
class Range
{
public:

	// Requirements on range concept
	Range(Range const &); //! Copy constructor

	Range(Range & r, split); //! Split range r into two subranges.

	~Range(); //! Destructor

	bool empty() const; //! True if range is empty

	bool is_divisible() const //!True if range can be partitioned into two subranges

	// Additional Requirements on a Container Range R
	typedef TI iterator; //! Iterator type for range

	typedef decltype(*std::declval<iterator>()) value_type;

	typedef value_type & reference; //! 	Item  reference type

	typedef value_type const& const_reference; //! 	Item 	const reference type

	typedef decltype( std::declval<value_type>()- std::declval<value_type>()) difference_type; //! Type for difference of two iterators

	iterator begin(); //! First item 	in range

	iterator end(); //! One past last item	in range

	size_t grainsize() const; //!	Grain size

	//**************************************************************************************************
	typedef Range<iterator> this_type;
	Range(iterator const & b, iterator const & e, size_t grainsize = 0)
			: b_(b), e_(e), grainsize_(grainsize == 0 ? (e - b) : grainsize)
	{
	}
private:
	iterator b_, e_;

	size_t grainsize_;

	std::tuple<this_type, this_type> split() const
	{
		iterator m = b_ + (e_ - b_) / 2u;

		return std::forward_as_tuple(Range(b_, m, grainsize_), Range(m, e_, grainsize_));
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
Range<TI> make_range(TI const & b, TI const &e, size_t grainsize = 0)
{
	return std::move(Range<TI>(b, e, grainsize));
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
//template<typename TI>
//std::tuple<Range<TI>, Range<TI>> split(Range<TI> const & range)
//{
//	return std::move(range.split());
//}
//template<typename TI>
//bool is_divisible(Range<TI> const& range)
//{
//	return range.is_divisible();
//}
}  // namespace simpla

#endif /* RANGE_H_ */
