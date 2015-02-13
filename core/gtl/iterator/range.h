/**
 * @file range.h
 *
 * \date    2014年8月27日  上午9:49:48 
 * \author salmon
 */

#ifndef SP_RANGE_H_
#define SP_RANGE_H_
#include <functional>
#include <iterator>

namespace simpla
{

//#ifdef USE_TBB
//#include <tbb/tbb.h>
//typedef tbb::split op_split;
//#else
class op_split
{
};

/**
 * @ingroup gtl
 *  @addtogroup range Range
 *  @{
 *  @brief  Range
 *
 *  >   A Range can be recursively subdivided into two parts
 *  > - Summary Requirements for type representing a recursively divisible set of values.
 *  > - Requirements The following table lists the requirements for a Range type R.
 *  >                                        -- from TBB
 */
template<typename TI>
class Range
{
public:

	typedef TI iterator_type; //! Iterator type for range

	typedef Range<iterator_type> this_type;

	typedef typename std::iterator_traits<iterator_type>::difference_type difference_type;

	Range(iterator_type first, iterator_type last, size_t grainsize = 0)
			: m_first_(first), m_last_(last), grainsize_(
					grainsize == 0 ? std::distance(first, last) : grainsize)
	{
	}

	//! Copy constructor
	Range(this_type const & other)
			: m_first_(other.m_first_), m_last_(other.m_last_), grainsize_(
					other.grainsize_)
	{
	}
	Range(this_type & other, op_split)
			: m_first_(other.m_first_), m_last_(other.m_first_), grainsize_(
					other.grainsize_)
	{
		std::advance(m_last_, std::distance(other.m_first_, other.m_last_) / 2);

		other.m_first_ = m_last_;
	}
	~Range()
	{
	}
	difference_type size() const
	{
		return std::distance(m_first_, m_last_);
	}
	//! True if range is empty
	bool empty() const
	{
		return m_last_ == m_first_;
	}

	//!True if range can be partitioned into two subranges
	bool is_divisible() const
	{
		return size() > grainsize_ * 2;
	}

	//!	Grain size
	size_t grainsize() const
	{
		return grainsize_;
	}

	//**************************************************************************************************

	/// concept block range
	/// Additional Requirements on a Container Range R
	/// First item 	in range
	iterator_type begin() const
	{
		return m_first_;
	}
	/// One past last item	in range
	iterator_type end() const
	{
		return m_last_;
	}

private:
	iterator_type m_first_, m_last_;

	difference_type grainsize_;

};

template<typename TI>
Range<TI> make_range(TI const & b, TI const &e)
{
	return std::move(Range<TI>(b, e));
}
/** @}*/
}  // namespace simpla

#endif /* SP_RANGE_H_ */
