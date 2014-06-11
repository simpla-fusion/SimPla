/*
 * mapped_iterator.h
 *
 *  Created on: 2014年6月11日
 *      Author: salmon
 */

#ifndef MAPPED_ITERATOR_H_
#define MAPPED_ITERATOR_H_

#include <iterator>
#include <map>
namespace std
{
template<typename TI> struct iterator_traits;
}  // namespace std
namespace simpla
{
template<typename TContainer, typename TIterator>
struct MappedIterator
{

	typedef TContainer conatiner_type;
	typedef TIterator key_iterator;

	typedef MappedIterator<conatiner_type, key_iterator> this_type;

	typedef typename std::iterator_traits<key_iterator>::iterator_category iterator_category;
	typedef typename std::iterator_traits<key_iterator>::difference_type difference_type;

	typedef typename TContainer::value_type value_type;
	typedef value_type* pointer;
	typedef value_type& reference;

	conatiner_type * data_;
	key_iterator k_it_;

	MappedIterator()
			: data_(nullptr)
	{
	}
	MappedIterator(this_type const & other)
			: data_(other.data_), k_it_(other.k_it_)
	{
	}

	MappedIterator(conatiner_type &d, key_iterator const & ib, key_iterator const&)
			: data_(&d), k_it_(ib)
	{
	}

	~MappedIterator()
	{
	}
	bool operator==(this_type const & other) const
	{
		return data_ == other.data_ && k_it_ == other.k_it_;
	}
	bool operator!=(this_type const & other) const
	{
		return !(operator==(other));
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
	reference operator*()
	{
		return (*data_)[*k_it_];
	}
	const reference operator*() const
	{
		return (*data_)[*k_it_];
	}
	pointer operator->()
	{
		return &(*data_)[*k_it_];
	}
	const pointer operator->() const
	{
		return &(*data_)[*k_it_];
	}
};

template<typename TMapped, typename TIterator>
struct MappedIterator<std::map<typename TIterator::value_type, TMapped>, TIterator> : public std::map<
        typename TIterator::value_type, TMapped>::iterator
{

	typedef std::map<typename std::iterator_traits<TIterator>::value_type, TMapped> conatiner_type;

	typedef TIterator key_iterator;

	typedef MappedIterator<conatiner_type, key_iterator> this_type;

	typedef typename conatiner_type::iterator base_iterator;

	typedef typename std::iterator_traits<key_iterator>::iterator_category iterator_category;
	typedef typename std::iterator_traits<key_iterator>::difference_type difference_type;

	typedef typename std::iterator_traits<base_iterator>::value_type value_type;
	typedef typename std::iterator_traits<base_iterator>::pointer pointer;
	typedef typename std::iterator_traits<base_iterator>::reference reference;

	conatiner_type * data_;
	key_iterator k_it_, k_it_end_;

	MappedIterator()
			: data_(nullptr)
	{

	}
	MappedIterator(this_type const & other)
			: base_iterator(other), data_(other.data_), k_it_(other.k_it_), k_it_end_(other.k_it_end_)
	{

	}

	MappedIterator(conatiner_type &d, key_iterator const & ib, key_iterator ie = key_iterator())
			: base_iterator(d.find(*ib)), data_(&d), k_it_(ib), k_it_end_(ie)
	{
		find_next_value_in_container();
	}

	~MappedIterator()
	{
	}
	bool operator==(this_type const & other) const
	{
		return data_ == other.data_ && k_it_ == other.k_it_;
	}
	this_type & operator ++()
	{
		++k_it_;
		*this = data_->find(*k_it_);
		find_next_value_in_container();

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
		*this = data_->find(*k_it_);

		find_prev_value_in_container();

		return *this;
	}
	this_type operator --(int)
	{
		this_type res(*this);
		--res;
		return std::move(res);
	}

private:
	void find_next_value_in_container()
	{

		while (k_it_ != k_it_end_ && (*this == data_->end()))
		{
			++k_it_;
			*this = data_->find(*k_it_);
		}
	}

	void find_prev_value_in_container()
	{

		while (k_it_ != k_it_end_ && (*this == data_->end()))
		{
			--k_it_;
			*this = data_->find(*k_it_);
		}
	}
};
template<typename TContainer, typename TIterator>
MappedIterator<TContainer, TIterator> make_mapped_iterator(TContainer & container, TIterator k_ib, TIterator k_ie =
        TIterator())
{
	return ((MappedIterator<TContainer, TIterator>(container, k_ib, k_ie)));
}
}  // namespace simpla

#endif /* MAPPED_ITERATOR_H_ */
