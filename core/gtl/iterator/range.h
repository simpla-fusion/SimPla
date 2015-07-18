/**
 * @file range.h
 *
 * @date    2014-8-27  AM9:49:48
 * @author salmon
 */

#ifndef SP_RANGE_H_
#define SP_RANGE_H_
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
 *  @brief Common  Range class
 *
 *  This class does nothing but define nested typedefs.  %Range classes
 *  can inherit from this class to save some work.  The typedefs are then
 *  used in specializations and overloading.
 *
 *  In particular, there are no default implementations of requirements
 *  such as @c operator++ and the like.  (How could there be?)
 *
 *  >   A Range can be recursively subdivided into two parts
 *  > - Summary Requirements for type representing a recursively divisible set of values.
 *  > - Requirements The following table lists the requirements for a Range type R.
 *  >                                        -- from TBB
 */

template<typename TIterator>
class Range<TIterator>
{
	typedef TIterator iterator_type;

	typedef typename iterator_type::iterator_category iterator_category;
	/// The type "pointed to" by the iterator.
	typedef typename iterator_type::value_type value_type;
	/// Distance between iterators is represented as this type.
	typedef typename iterator_type::value_type difference_type;
	/// This type represents a pointer-to-value_type.
	typedef typename iterator_type::pointer pointer;
	/// This type represents a reference-to-value_type.
	typedef typename iterator_type::reference reference;

};

/** @}*/
}  // namespace simpla

#endif /* SP_RANGE_H_ */
