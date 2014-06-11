/*
 * range.h
 *
 *  Created on: 2014年6月11日
 *      Author: salmon
 */

#ifndef RANGE_H_
#define RANGE_H_

#include <iterator>
#include <type_traits>
#include <utility>
#include "filter_iterator.h"
s
namespace std
{
template<typename TI> struct iterator_traits;
}  // namespace std

namespace simpla
{
template<typename TIterator>
std::pair<TIterator, TIterator> Split(TIterator ib, TIterator ie, unsigned int num_process, unsigned int process_num)
{
	return std::make_pair(ib + (ie - ib) * process_num / num_process, ib + (ie - ib) * (process_num + 1) / num_process);
}

template<typename TIterator>
class Range
{
public:

	typedef TIterator iterator;
	typedef Range<iterator> this_type;
	typedef typename std::iterator_traits<iterator>::iterator_category iterator_category;

	Range()
	{
	}
	Range(std::pair<iterator, iterator> const & r)
			: ib_(r.first), ie_(r.second)
	{

	}
	Range(this_type const & other)
			: ib_(other.ib_), ie_(other.ie_)
	{

	}
	Range(iterator ib, iterator ie)
			: ib_(ib), ie_(ie)
	{

	}

	iterator begin() const
	{
		return ib_;
	}
	iterator end() const
	{
		return ie_;
	}
//	typename std::enable_if<std::is_same<iterator_category, std::bidirectional_iterator_tag>::value, iterator>::type rbegin() const
//	{
//		return --ie_;
//	}
//	typename std::enable_if<std::is_same<iterator_category, std::bidirectional_iterator_tag>::value, iterator>::type rend() const
//	{
//		return --ib_;
//	}

	template<typename ...Args>
	this_type Split(Args const & ... args) const
	{
		return this_type(simpla::Split(ib_, ie_, std::forward<Args const &>(args)...));
	}
private:
	iterator ib_, ie_;
};

template<typename TIterator>
auto create_range(TIterator const & ib, TIterator const & ie)
DECL_RET_TYPE (Range<TIterator>(ib, ie))

template< typename TMapOrPred,typename TIterator>
auto make_range(TMapOrPred & m, TIterator const & ib, TIterator const & ie)
DECL_RET_TYPE (create_range(make_filter_iterator(m, ib, ie), make_filter_iterator(m, ie, ie)))

template<typename TMapOrPred,typename TIterator>
auto make_range(TMapOrPred & m, std::pair<TIterator, TIterator> const & r)
DECL_RET_TYPE (create_range(make_filter_iterator(m, r.first, r.second), make_filter_iterator(m, r.second, r.second)))

template<typename TMapOrPred,typename TRange>
auto make_range(TMapOrPred & m, TRange const & r)
DECL_RET_TYPE (create_range(make_filter_iterator(m, r.begin(), r.end()), make_filter_iterator(m, r.end(), r.end())))

//DECL_RET_TYPE( make_range( m, r.begin(), r.end()) )

}
  // namespace simpla

#endif /* RANGE_H_ */
