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

template<typename ...> class Range;
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
class Range<TI>
{
public:

	typedef TI iterator; //! Iterator type for range

	typedef Range<iterator> this_type;

	typedef typename std::iterator_traits<iterator>::difference_type difference_type;

	Range()
	{
	}

	Range(iterator const &first, iterator const & last)
			: m_first_(first), m_last_(last)
	{
		m_grainsize_ = 0;
	}

	Range(iterator const &first, iterator const &last,
			difference_type const &grainsize)
			: m_first_(first), m_last_(last), m_grainsize_(
					grainsize == 0 ? std::distance(first, last) : grainsize)
	{
	}

	//! Copy constructor
	Range(this_type const & other)
			: m_first_(other.m_first_), m_last_(other.m_last_), m_grainsize_(
					other.m_grainsize_)
	{
	}

	Range(this_type & other, op_split)
			: m_first_(other.m_first_), m_last_(other.m_first_), m_grainsize_(
					other.m_grainsize_)
	{
		std::advance(m_last_, std::distance(other.m_first_, other.m_last_) / 2);
		other.m_first_ = m_last_;
	}

	virtual ~Range()
	{
	}

	void swap(Range & other)
	{
		std::swap(m_first_, other.m_first_);
		std::swap(m_last_, other.m_last_);
		std::swap(m_grainsize_, other.m_grainsize_);
	}
	//! True if range is empty
	virtual bool empty() const
	{
		return m_last_ == m_first_;
	}

	//!True if range can be partitioned into two subranges
	virtual bool is_divisible() const
	{
		return false;
	}
	//!	Grain size
	difference_type size() const
	{
		return m_last_ - m_first_;
	}
	//!	Grain size
	difference_type grainsize() const
	{
		return m_grainsize_;
	}

	//**************************************************************************************************

	/// concept block range
	/// Additional Requirements on a Container Range R
	/// First item 	in range
	iterator begin() const
	{
		return m_first_;
	}
	/// One past last item	in range
	iterator end() const
	{
		return m_last_;
	}

private:
	iterator m_first_, m_last_;

	difference_type m_grainsize_;

};

template<typename TI>
Range<TI> make_range(TI const & b, TI const &e)
{
	return std::move(Range<TI>(b, e));
}
/** @}*/
}  // namespace simpla

#endif /* SP_RANGE_H_ */
