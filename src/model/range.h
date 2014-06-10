/*
 * range.h
 *
 *  Created on: 2014年6月5日
 *      Author: salmon
 */

#ifndef RANGE_H_
#define RANGE_H_

#include <utility>

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
namespace _impl
{

template<typename Key, typename Mapped, typename key_iterator>
Mapped* NextValue_(std::map<Key, Mapped> & data_, key_iterator * k_it, key_iterator const & k_ie_)
{
	boost::filter_iterator
	auto res = data_.end();

	while (res == data_.end() && *k_it != k_ie_)
	{
		++(*k_it);
		res = data_.find(**k_it);
	}

	return (res == data_.end()) ? nullptr : (&(res->seond));
}
template<typename Key, typename Mapped, typename key_iterator>
Mapped* PrevValue_(std::map<Key, Mapped> & data_, key_iterator * k_it, key_iterator const & k_ie_)
{

	auto res = data_.end();

	while (res == data_.end() && *k_it != k_ie_)
	{
		--(*k_it);
		res = data_.find(**k_it);
	}

	return (res == data_.end()) ? nullptr : (&(res->seond));
}
template<typename TC, typename key_iterator>
auto PrevValue_(TC & data_, key_iterator * k_it, key_iterator const & k_ie)
DECL_RET_TYPE ((&(data_[*(--(*k_it))])))

}
// namespace _impl

/**
 *  @ref boost::filter_iterator
 *
 */
template<typename TIterator>
class FilterIterator: public TIterator
{
	typedef FilterIterator<TIterator> this_type;

	typedef TIterator base_iterator;

	typedef std::function<bool(value_type const&)> predicate_fun;

	typedef typename base_iterator::value_type value_type;

	typedef value_type * pointer;

	typedef value_type & reference;

	predicate_fun predicate_;
	base_iterator it_, ie_;

	FilterIterator(predicate_fun const &p, base_iterator const & ib, base_iterator ie = base_iterator()) :
			predicate_(), it_(ib), ie_(ie)
	{
	}
	~FilterIterator()
	{
	}

	this_type & operator ++()
	{
		++it_;
		while (it_ != ie_ && !(predicate_(*it_)))
		{
			++it_;
		}
		return *this;
	}
	this_type operator ++(int)
	{
		this_type res(*this);
		++res;
		return std::move(res);
	}

	value_type & operator*()
	{
		return *it_;
	}

	value_type const& operator*() const
	{
		return *it_;
	}

	pointer operator ->()
	{
		return &(operator*());
	}

	const pointer operator ->() const
	{
		return &(operator*());
	}

	bool operator==(this_type const & rhs) const
	{
		return it_ == rhs.it_;
	}

	bool operator!=(this_type const & rhs) const
	{
		return !(this->operator==(rhs));
	}

};

template<typename MapConatiner, typename KeyIterator>
class MapIterator
{
public:
	typedef MapIterator<MapConatiner, KeyIterator> this_type;
	typedef MapConatiner container_type;
	typedef typename MapConatiner::key_type key_type;
	typedef typename MapConatiner::mapped_type mapped_type;
	typedef KeyIterator key_iterator;

	typedef mapped_type value_type;

	typedef value_type * pointer;

	typedef value_type & reference;

	container_type & container_;
	key_iterator it_;
	const key_iterator ie_;

	MapIterator(container_type & container, key_iterator k_ib, key_iterator k_ie = key_iterator()) :
			container_(container), it_(k_ib), ie_(k_ie)
	{
	}
	~MapIterator()
	{

	}

	this_type & operator ++()
	{
		++it_;
		while (it_ != ie_ && (container_.find(*it_) == container_.end()))
		{
			++it_;
		}
		return *this;
	}
	this_type operator ++(int)
	{
		this_type res(*this);
		++res;
		return std::move(res);
	}

	value_type & operator*()
	{
		return container_[*it_];
	}

	value_type const& operator*() const
	{
		return container_[*it_];
	}

	pointer operator ->()
	{
		return &(operator*());
	}

	const pointer operator ->() const
	{
		return &(operator*());
	}

	bool operator==(this_type const & rhs) const
	{
		return it_ == rhs.it_;
	}

	bool operator!=(this_type const & rhs) const
	{
		return !(this->operator==(rhs));
	}

};

template<typename MapConatiner, typename KeyIterator>
MapIterator<MapConatiner, KeyIterator> make_map_iterator(MapConatiner & container, KeyIterator k_ib, KeyIterator k_ie =
		KeyIterator())
{
	return MapIterator<MapConatiner, KeyIterator>(container, k_ib, k_ie);
}

//template<typename TRange, typename TContainer>
//struct RangeWrapper: public TRange
//{
//public:
//	typedef TRange base_range_type;
//	typedef typename base_range_type::iterator base_iterator;
//	typedef TContainer container_type;
//	typedef RangeWrapper<base_range_type, container_type> this_type;
//
//	typedef IteratorWrapper<base_iterator, container_type> iterator;
//
//private:
//	container_type & data_;
//public:
//	RangeWrapper(container_type & data, base_range_type const & r) :
//			base_range_type(r), data_(data)
//	{
//	}
//
//	~RangeWrapper()
//	{
//	}
//
//	iterator begin() const
//	{
//		return iterator(data_, base_range_type::begin(), base_range_type::begin(), base_range_type::end());
//	}
//	iterator end() const
//	{
//		return iterator(data_, base_range_type::end(), base_range_type::begin(), base_range_type::end());
//	}
//
//	typename std::enable_if<
//			std::is_same<typename base_iterator::iterator_category, std::bidirectional_iterator_tag>::value, iterator>::type rbegin() const
//	{
//		return iterator(data_, base_range_type::rbegin(), base_range_type::rbegin(), base_range_type::rend());
//	}
//	typename std::enable_if<
//			std::is_same<typename base_iterator::iterator_category, std::bidirectional_iterator_tag>::value, iterator>::type rend() const
//	{
//		return iterator(data_, base_range_type::rend(), base_range_type::rbegin(), base_range_type::rend());
//	}
//	template<typename ...Args>
//	this_type Split(Args const & ... args) const
//	{
//		return this_type(data_, base_range_type::Split(std::forward<Args const &>(args)...));
//	}
//
//}
//;
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

}  // namespace simpla

#endif /* RANGE_H_ */
