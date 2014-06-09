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
template<typename TIterator>
struct Range
{
public:
	typedef TIterator iterator;
	typedef Range<iterator> this_type;

	iterator ib_, ie_;

	Range()
	{
	}

	Range(Range const & r)
			: ib_(r.ib_), ie_(r.ie_)
	{
	}
	Range(iterator ib, iterator ie)
			: ib_(ib), ie_(ie)
	{
	}

	~Range()
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

	typename std::enable_if<std::is_same<typename iterator::iterator_category, std::bidirectional_iterator_tag>::value,
	        iterator>::type rbegin() const
	{
		iterator res = ie_;
		--res;
		return res;
	}
	typename std::enable_if<std::is_same<typename iterator::iterator_category, std::bidirectional_iterator_tag>::value,
	        iterator>::type rend() const
	{
		iterator res = ib_;
		--res;
		return res;
	}

	template<typename ...Args>
	this_type Split(Args const &...args) const
	{
		return Split(ib_, ie_, std::forward<Args const &> (args)...);
	}

}
;
template<typename TIterator, typename ...Args>
Range<TIterator> Split(Range<TIterator> const & r, Args const & ... args)
{
	return Range<TIterator>(Split(r.begin(), r.end(), std::forward<Args const &>(args) ...));
}

template<typename TContainer, typename TKeyIterator> struct IteratorWrap;

template<typename TKeyITerator, typename TMaped>
class IteratorWrap<std::map<typename TKeyITerator::value_type, TMaped>, TKeyITerator>
{
	typedef TMaped mapped_type;

	typedef TKeyITerator key_iterator;

	typedef typename key_iterator::value_type key_type;

	typedef std::map<key_type, mapped_type> container_type;

	typedef IteratorWrap<std::map<key_type, mapped_type>, key_iterator> this_type;

	typedef typename container_type::iterator container_iterator;

	typedef typename container_type::iterator value_iterator;

	/// One of the @link iterator_tags tag types@endlink.
	typedef std::forward_iterator_tag iterator_category;

	/// The type "pointed to" by the iterator.
	typedef mapped_type value_type;

	/// This type represents a pointer-to-value_type.
	typedef value_type * pointer;

	/// This type represents a reference-to-value_type.
	typedef value_type & reference;

	typedef container_type& container_reference;

	container_reference data_;

	key_iterator k_it_, k_ie_;

	template<typename ...Args>
	IteratorWrap(container_reference data, key_iterator k_ib, key_iterator k_ie)
			: data_(data), k_it_(k_ib), k_ie_(k_ie)
	{
	}

	~IteratorWrap()
	{
	}

	reference operator*()
	{
		return FindValue_(data_, &k_it_);
	}
	const reference operator*() const
	{
		return FindValue_(data_, &k_it_)->second;
	}
	pointer operator ->()
	{
		return &(FindValue_(data_, &k_it_)->second);
	}
	const pointer operator ->() const
	{
		return &(FindValue_(data_, &k_it_)->second);
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

	this_type operator --(int)
	{
		this_type res(*this);
		--res;
		return std::move(res);
	}

private:
	value_iterator FindNextValue_(container_type data, key_iterator * k_it) const
	{
		value_iterator res = data_.find(*k_it);
		while (res == data_.end() && *k_it != k_ie_)
		{
			++(*k_it);
			res = data_.find(*k_it);

		}
		return res;
	}

};

}  // namespace simpla

#endif /* RANGE_H_ */
