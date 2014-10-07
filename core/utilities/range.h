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

class split_tag
{
};

/**
 *  \page concept_range Range
 *  "
 *    A Range can be recursively subdivided into two parts
 *  - Summary Requirements for type representing a recursively divisible set of values.
 *  - Requirements The following table lists the requirements for a Range type R.
 *  "  --TBB
 *  \brief Range
 *
 *
 */
template<typename TI>
class Range
{
public:

	// concept Range
	typedef TI const_iterator; //! Iterator type for range
	typedef Range<const_iterator> this_type;
	typedef decltype(std::declval<const_iterator>()-std::declval<const_iterator>()) size_type;

	Range(this_type const &); //! Copy constructor

	Range(this_type & r, split_tag); //! Split range r into two subranges.

	~Range(); //! Destructor

	bool empty() const; //! True if range is empty

	bool is_divisible() const //!True if range can be partitioned into two subranges

	//**************************************************************************************************

	// concept block range
	// Additional Requirements on a Container Range R

	const_iterator begin(); //! First item 	in range

	const_iterator end(); //! One past last item	in range

	size_t grainsize() const; //!	Grain size

	Range(const_iterator const & b, const_iterator const & e, size_t grainsize = 0)
			: b_(b), e_(e), grainsize_(grainsize == 0 ? (e - b) : grainsize)
	{
	}

	size_type size() const
	{
		return e_ - b_;
	}
	const_iterator const & begin() const
	{
		return b_;
	}
	const_iterator const & end() const
	{
		return e_;
	}

private:
	const_iterator b_, e_;

	size_type grainsize_;

	std::tuple<this_type, this_type> do_split(this_type & r, split_tag) const
	{
		const_iterator m = b_ + (e_ - b_) / 2u;

		return std::forward_as_tuple(Range(b_, m, grainsize_), Range(m, e_, grainsize_));
	}

};

template<typename TI>
Range<TI> make_range(TI const & b, TI const &e, size_t grainsize = 0)
{
	return std::move(Range<TI>(b, e, grainsize));
}

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

} // namespace simpla

#endif /* RANGE_H_ */
