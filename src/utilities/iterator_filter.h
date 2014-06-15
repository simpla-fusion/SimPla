/*
 * iterator_filter.h
 *
 *  Created on: 2014年6月5日
 *      Author: salmon
 */

#ifndef ITERATOR_FILTER_H_
#define ITERATOR_FILTER_H_

#include <iterator>
#include <map>
#include "../utilities/sp_type_traits.h"

namespace std
{
template<typename TI> struct iterator_traits;
}  // namespace std
namespace simpla
{
/**
 *  @concept Range is a container
 *   class Range
 *   {
 *     typedef iterator;
 *     iterator begin();
 *     iterator end();
 *     const_iterator begin()const;
 *     const_iterator end()const;
 *
 *     Range split(int num_process,int process_num); //optional
 *   }
 */

/**
 *  @ref boost::filter_iterator
 *
 */

template<typename TPred, typename TIterator>
struct FilterIterator: public TIterator
{

	typedef TIterator base_iterator;

	typedef TPred predicate_fun;

	typedef FilterIterator<predicate_fun, base_iterator> this_type;

	typedef typename std::iterator_traits<base_iterator>::iterator_category iterator_category;
	typedef typename std::iterator_traits<base_iterator>::value_type value_type;
	typedef typename std::iterator_traits<base_iterator>::difference_type difference_type;
	typedef typename std::iterator_traits<base_iterator>::pointer pointer;
	typedef typename std::iterator_traits<base_iterator>::reference reference;

	predicate_fun predicate_;
	base_iterator it_end_;

	FilterIterator()
	{
	}
	FilterIterator(predicate_fun const &p, base_iterator const & ib, base_iterator ie = base_iterator())
			: base_iterator(ib), predicate_(p), it_end_(ie)
	{
		satisfy_predicate();
	}

	~FilterIterator()
	{
	}

	this_type & operator ++()
	{
		base_iterator::operator++();
		satisfy_predicate();

		return *this;
	}
	this_type operator ++(int)
	{
		this_type res(*this);
		++res;
		return std::move(res);
	}
	this_type & operator --()
	{
		while (base_iterator::operator !=(it_end_) && !(predicate_(*(base_iterator::operator--()))))
		{
		}

		return *this;
	}
	this_type operator --(int)
	{
		this_type res(*this);
		--res;
		return std::move(res);
	}

private:
	void satisfy_predicate()
	{
		while (base_iterator::operator !=(it_end_) && !(predicate_(**this)))
		{
			base_iterator::operator++();
		}
	}
};

template<typename TPred, typename TIterator>
FilterIterator<TPred, TIterator> make_filter_iterator(TPred const & pred, TIterator k_ib, TIterator k_ie = TIterator())
{
	return ((FilterIterator<TPred, TIterator>(pred, k_ib, k_ie)));
}

template<typename > struct Range;

template<typename TPred, typename TIterator>
auto make_filter_range(TPred & m, TIterator const & ib, TIterator const & ie)
DECL_RET_TYPE ((Range<FilterIterator<TPred, TIterator>>(make_filter_iterator(m, ib, ie),
						make_filter_iterator(m, ie, ie))))

template<typename TPred, typename TIterator>
auto make_filter_range(TPred & m, std::pair<TIterator, TIterator> const & r)
DECL_RET_TYPE ((Range<FilterIterator<TPred, TIterator>>(
						make_filter_iterator(m, r.first, r.second),
						make_filter_iterator(m, r.second, r.second))))

template<typename TPred, typename TRange>
auto make_filter_range(TPred & m, TRange const & r)
DECL_RET_TYPE ((Range<FilterIterator<TPred, typename TRange::iterator>> (
						make_filter_iterator(m, r.begin(), r.end()),
						make_filter_iterator(m, r.end(), r.end()))))

}
// namespace simpla
namespace std
{

template<typename TM, typename TI>
struct iterator_traits<::simpla::FilterIterator<TM, TI>>
{
	typedef simpla::FilterIterator<TM, TI> iterator;
	typedef typename iterator::iterator_category iterator_category;
	typedef typename iterator::value_type value_type;
	typedef typename iterator::difference_type difference_type;
	typedef typename iterator::pointer pointer;
	typedef typename iterator::reference reference;

};
}  // namespace std
#endif /* ITERATOR_FILTER_H_ */
