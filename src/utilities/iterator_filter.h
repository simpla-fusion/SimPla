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
 *  @defgroup iterator
 */

/**
 *  @ingroup iterator
 *
 *  @ref boost::filter_iterator
 *
 */
template<typename TIterator, typename TPred>
struct FilterIterator: public TIterator
{

	typedef TIterator base_iterator;

	typedef TPred predicate_fun;

	typedef FilterIterator<base_iterator, predicate_fun> this_type;

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

	FilterIterator(this_type const & other)
			: base_iterator(other), predicate_(other.predicate_), it_end_(other.it_end_)
	{
	}

	FilterIterator(this_type && other)
			: base_iterator(other), predicate_(other.predicate_), it_end_(other.it_end_)
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
auto make_filter_iterator(TPred pred, TIterator k_ib, TIterator k_ie = TIterator())
DECL_RET_TYPE((FilterIterator< TIterator,TPred>(pred, k_ib, k_ie)))

template<typename > struct Range;

template<typename TPred, typename TIterator>
auto make_filter_range(TPred pred, TIterator ib, TIterator ie)
DECL_RET_TYPE ((std::make_pair( make_filter_iterator(pred, ib, ie),
						make_filter_iterator(pred, ie, ie))))

template<typename TPred, typename TIterator>
auto make_filter_range(TPred pred, std::pair<TIterator, TIterator> const & r)
DECL_RET_TYPE ((std::make_pair( make_filter_iterator(pred, r.first, r.second),
						make_filter_iterator(pred, r.second, r.second))))

template<typename TPred, typename TRange>
auto make_filter_range(TPred pred, TRange const & r)
DECL_RET_TYPE ((std::make_pair( make_filter_iterator(pred, r.begin(), r.end()),
						make_filter_iterator(pred, r.end(), r.end()))))

}
// namespace simpla
namespace std
{

template<typename TPred, typename TI>
struct iterator_traits<::simpla::FilterIterator<TI, TPred>>
{
	typedef simpla::FilterIterator<TI, TPred> iterator;
	typedef typename iterator::iterator_category iterator_category;
	typedef typename iterator::value_type value_type;
	typedef typename iterator::difference_type difference_type;
	typedef typename iterator::pointer pointer;
	typedef typename iterator::reference reference;

};
}  // namespace std
#endif /* ITERATOR_FILTER_H_ */
