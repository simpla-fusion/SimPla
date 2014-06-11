/*
 * filter_iterator.h
 *
 *  Created on: 2014年6月5日
 *      Author: salmon
 */

#ifndef FILTER_ITERATOR_H_
#define FILTER_ITERATOR_H_

#include <iterator>
#include <map>

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
//
//template<typename TContainer, typename TRange>
//RangeWrapper<typename TRange::iterator, TContainer> make_range(TContainer & data, TRange const& r)
//{
//	return RangeWrapper<typename TRange::iterator, TContainer>(data, r);
//}
//
//template<typename TIterator>
//struct IteratorWrapper<TIterator, std::function<bool(TIterator const&)> >
//{
//public:
//
//	typedef TIterator base_iterator;
//	typedef std::function<bool(base_iterator const&)> filter_fun;
//	typedef IteratorWrapper<base_iterator, std::function<bool(TIterator const&)>> this_type;
//
///// One of the @link iterator_tags tag types@endlink.
//	typedef std::forward_iterator_tag iterator_category;
//
///// The type "pointed to" by the iterator.
//	typedef typename base_iterator::value_type value_type;
//
///// This type represents a pointer-to-value_type.
//	typedef value_type * pointer;
//
///// This type represents a reference-to-value_type.
//	typedef value_type & reference;
//
//	filter_fun filter_;
//
//	base_iterator k_it_, k_ie_;
//
//	template<typename ...Args>
//	IteratorWrapper(filter_fun & data, base_iterator k_it, base_iterator k_ib, base_iterator k_ie) :
//			filter_(data), k_it_(k_it), k_ie_(k_ie)
//	{
//	}
//
//	~IteratorWrapper()
//	{
//	}
//
//	void FindValue_()
//	{
//		while (k_it_ != k_ie_ && !filter_(k_it_))
//		{
//			++k_it_;
//		}
//	}
//
//	reference operator*()
//	{
//		while (k_it_ != k_ie_ && !filter_(k_it_))
//		{
//			++k_it_;
//		}
//		return *k_it_;
//	}
//	pointer operator ->()
//	{
//		return &(operator*());
//	}
//	const reference operator*() const
//	{
//		return *k_it_;
//	}
//	const pointer operator ->() const
//	{
//		return &(operator*());
//	}
//
//	bool operator==(this_type const & rhs) const
//	{
//		return k_it_ == rhs.k_it_;
//	}
//
//	bool operator!=(this_type const & rhs) const
//	{
//		return !(this->operator==(rhs));
//	}
//
//	this_type & operator ++()
//	{
//		++k_it_;
//		return *this;
//	}
//	this_type operator ++(int)
//	{
//		this_type res(*this);
//		++res;
//		return std::move(res);
//	}
//
//}
//;
//template<typename TRange>
//struct RangeWrapper<TRange, std::function<bool(typename TRange::iterator const&)> > : public TRange
//{
//
//	typedef RangeWrapper<TRange, std::function<bool(typename TRange::iterator const&)> > this_type;
//
//	typedef TRange base_range;
//
//	typedef typename base_range::iterator base_iterator;
//
//	typedef typename base_iterator::value_type value_type;
//
//	typedef std::function<bool(base_iterator)> filter_type;
//
//	typedef IteratorWrapper<base_iterator, filter_type> iterator;
//
//private:
//	base_range range_;
//	filter_type filter_;
//public:
//
//	RangeWrapper(base_range range, filter_type filter) :
//			range_(range), filter_(filter)
//	{
//	}
//
//	RangeWrapper() :
//			range_(base_range())
//	{
//	}
//	~RangeWrapper()
//	{
//	}
//
//	bool empty()
//	{
//		return begin() == end();
//	}
//
//	iterator begin() const
//	{
//		return iterator(range_.begin(), range_.end(), filter_);
//	}
//	iterator end() const
//	{
//		return iterator(range_.end());
//	}
//
//	this_type Split(size_t num, size_t id)
//	{
//		return this_type(TRange::Split(num, id), filter_);
//	}
//
//};

}// namespace simpla
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
#endif /* FILTER_ITERATOR_H_ */
