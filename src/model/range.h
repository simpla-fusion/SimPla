/*
 * range.h
 *
 *  Created on: 2014年6月5日
 *      Author: salmon
 */

#ifndef RANGE_H_
#define RANGE_H_

namespace simpla
{

namespace _impl
{

template<typename Key, typename Mapped, typename key_iterator>
auto FindValue_(std::map<Key, Mapped> & data_, key_iterator const & k_ie_,
		key_iterator * k_it) ->decltype(&(data_.find(*k_it)->second))
{

	auto res = data_.find(*k_it);

	while (res == data_.end() && *k_it != k_ie_)
	{
		++(*k_it);
		res = data_.find(*k_it);
	}

	return (*k_it == k_ie_) ? nullptr : &res->second;
}

template<typename TC, typename key_iterator>
auto FindValue_(TC & data_, key_iterator const & k_ie_, key_iterator * k_it)
DECL_RET_TYPE ((&(data_[(*k_it)])))

}  // namespace _impl

template<typename TContainer, typename TKeyIterator>
class IteratorWrapper
{
public:

	typedef TContainer container_type;

	typedef TKeyIterator key_iterator;

	typedef IteratorWrapper<container_type, key_iterator> this_type;

	/// One of the @link iterator_tags tag types@endlink.
	typedef std::forward_iterator_tag iterator_category;

	typedef decltype(*std::declval<key_iterator>()) key_type;
	/// The type "pointed to" by the iterator.
	typedef typename std::remove_reference<decltype(std::declval<container_type>()[*std::declval<key_iterator>()])>::type value_type;

	/// This type represents a pointer-to-value_type.
	typedef value_type * pointer;

	/// This type represents a reference-to-value_type.
	typedef value_type & reference;

	container_type & data_;

	key_iterator k_it_, k_ie_;

	template<typename ...Args>
	IteratorWrapper(container_type & data, key_iterator k_it, key_iterator k_ib, key_iterator k_ie) :
			data_(data), k_it_(k_it), k_ie_(k_ie)
	{
	}

	~IteratorWrapper()
	{
	}

	reference operator*()
	{
		return *_impl::FindValue_(data_, k_ie_, &k_it_);
	}
	const reference operator*() const
	{
		return *_impl::FindValue_(data_, k_ie_, &k_it_);
	}
	pointer operator ->()
	{
		return (_impl::FindValue_(data_, k_ie_, &k_it_));
	}
	const pointer operator ->() const
	{
		return (_impl::FindValue_(data_, k_ie_, &k_it_));
	}
	bool operator==(this_type const & rhs) const
	{
		return k_it_ == rhs.k_it_;
	}

	bool operator!=(this_type const & rhs) const
	{
		return !(this->operator==(rhs));
	}

	this_type & operator ++()
	{
		++k_it_;
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
		--k_it_;
		return *this;
	}
	this_type operator --(int)
	{
		this_type res(*this);
		--res;
		return std::move(res);
	}
};

template<typename TContainer, typename TIterator>
struct RangeWrapper
{
public:
	typedef TIterator orig_iterator;
	typedef TContainer container_type;

	typedef RangeWrapper<container_type, orig_iterator> this_type;
	typedef IteratorWrapper<container_type, orig_iterator> iterator;
private:
	container_type & data_;
	orig_iterator ib_, ie_;
public:
	template<typename TR>
	RangeWrapper(container_type & data, TR const & r) :
			data_(data), ib_(r.begin()), ie_(r.end())
	{
	}
	RangeWrapper(container_type & data, orig_iterator ib, orig_iterator ie) :
			data_(data), ib_(ib), ie_(ie)
	{
	}

	~RangeWrapper()
	{
	}

	iterator begin() const
	{
		return iterator(data_, ib_, ib_, ie_);
	}
	iterator end() const
	{
		return iterator(data_, ie_, ib_, ie_);
	}

	typename std::enable_if<
			std::is_same<typename orig_iterator::iterator_category, std::bidirectional_iterator_tag>::value, iterator>::type rbegin() const
	{
		orig_iterator rb = ie_;
		--rb;
		orig_iterator re = ib_;
		--re;
		return iterator(data_, rb, rb, re);
	}
	typename std::enable_if<
			std::is_same<typename orig_iterator::iterator_category, std::bidirectional_iterator_tag>::value, iterator>::type rend() const
	{
		orig_iterator rb = ie_;
		--rb;
		orig_iterator re = ib_;
		--re;
		return iterator(data_, re, rb, re);
	}

}
;

template<typename TContainer, typename TRange>
RangeWrapper<TContainer, typename TRange::iterator> make_range_wrapper(TContainer & data, TRange const& r)
{
	return RangeWrapper<TContainer, typename TRange::iterator>(data, r);
}

template<typename TContainer, typename TIterator>
RangeWrapper<TContainer, TIterator> make_range_wrapper(TContainer & data, TIterator const& ib, TIterator const &ie)
{
	return RangeWrapper<TContainer, TIterator>(data, ib, ie);
}

template<typename TRange, typename ...Args>
TRange Split(TRange const & r, Args const & ... args)
{
	return TRange(Split(r.begin(), r.end(), std::forward<Args const &>(args) ...));
}

}  // namespace simpla

#endif /* RANGE_H_ */
