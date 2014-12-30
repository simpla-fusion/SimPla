/*
 * sp_iterator_mapped.h
 *
 *  created on: 2014-6-11
 *      Author: salmon
 */

#ifndef SP_ITERATOR_MAPPED_H_
#define SP_ITERATOR_MAPPED_H_

#include <map>
#include <memory>

#include "../containers/sp_iterator.h"
#include "../utilities/sp_type_traits.h"

namespace simpla
{
class _iterator_policy_mapped
{
};

/**
 *  @ingroup iterator
 *  @brief mapped iterator
 */
template<typename TIterator, typename TContainer, bool IsReference>
struct Iterator<TIterator, TContainer, _iterator_policy_mapped, IsReference>
{
	HAS_MEMBER_FUNCTION(at);

public:
	typedef TContainer conatiner_type;
	typedef TIterator key_iterator;

	typedef Iterator< key_iterator, conatiner_type,_iterator_policy_mapped, IsReference> this_type;

	typedef typename std::conditional<std::is_pointer<conatiner_type>::value,
	typename std::remove_pointer<conatiner_type>::type, typename conatiner_type::value_type>::type value_type;

	typedef typename key_iterator::iterator_category iterator_category;
	typedef value_type* pointer;
	typedef value_type& reference;
	typedef value_type const* const_pointer;
	typedef value_type const& const_reference;

	typedef typename std::conditional<IsReference, conatiner_type&, conatiner_type >::type storage_type;

	storage_type data_;
	key_iterator k_it_;

	Iterator()
	: data_(nullptr)
	{
	}
	Iterator(this_type const & other)
	: data_(other.data_), k_it_(other.k_it_)
	{
	}

	Iterator( key_iterator const & ib, key_iterator const& ,storage_type d)
	: data_( d), k_it_(ib)
	{
	}

	~Iterator()
	{
	}
	bool operator==(this_type const & other) const
	{
		return &data_ == &other.data_ && k_it_ == other.k_it_;
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
		return get_value( data_,*k_it_);
	}
	pointer operator->()
	{
		return &get_value( data_,*k_it_);
	}

	const_reference operator*() const
	{
		return get(typename std::integral_constant<bool, has_member_function_at<conatiner_type>::value>());
	}
	const_pointer operator->() const
	{
		return &get(typename std::integral_constant<bool, has_member_function_at<conatiner_type>::value>());

	}
	reference get(std::false_type)
	{
		return get_value( data_,*k_it_);
	}
	reference get(std::true_type)
	{
		return get_value( data_,*k_it_);
	}
	const_reference get(std::false_type) const
	{
		return get_value( data_,*k_it_);
	}
	const_reference get(std::true_type) const
	{
		return get_value( data_,*k_it_);
	}
};

template<typename TKey, typename TMapped, typename TIterator, bool IsReference>
struct Iterator<TIterator, std::map<TKey, TMapped>, _iterator_policy_mapped,
		IsReference>
{

	typedef std::map<TKey, TMapped> conatiner_type;

	typedef TIterator key_iterator;

	typedef Iterator<key_iterator, conatiner_type, _iterator_policy_mapped,
			IsReference> this_type;

	typedef typename conatiner_type::iterator base_iterator;

	typedef typename key_iterator::iterator_category iterator_category;

	typedef typename conatiner_type::mapped_type value_type;
	typedef value_type* pointer;
	typedef value_type& reference;
	typedef value_type const* const_pointer;
	typedef value_type const& const_reference;

	typedef typename std::conditional<IsReference, conatiner_type&,
			conatiner_type>::type storage_type;

	storage_type data_;
	key_iterator k_it_, k_it_end_;
	base_iterator b_it_;

	Iterator() :
			data_(nullptr)
	{

	}
	Iterator(this_type const & other) :
			data_(other.data_), k_it_(other.k_it_), k_it_end_(other.k_it_end_), b_it_(
					other.b_it_)
	{

	}

	Iterator(key_iterator const & ib, key_iterator ie, storage_type d) :
			data_(d), k_it_(ib), k_it_end_(ie), b_it_(d->end())
	{
		next();
	}

	~Iterator()
	{
	}
	bool operator==(this_type const & other) const
	{
		return k_it_ == other.k_it_ && b_it_ == other.b_it_;

	}
	this_type & operator ++()
	{
		next();

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
		prev();

		return *this;
	}
	this_type operator --(int)
	{
		this_type res(*this);
		--res;
		return std::move(res);
	}
	reference operator *()
	{
		return b_it_->second;
	}
	const_reference operator *() const
	{
		return b_it_->second;
	}
	pointer operator ->()
	{
		return &(b_it_->second);
	}
	const_pointer operator ->() const
	{
		return &(b_it_->second);
	}

	void next()
	{
		do
		{
			++k_it_;
			b_it_ = data_->find(*k_it_);

		} while (k_it_ != k_it_end_ && (b_it_ == data_->end()));
	}

	void prev()
	{
		do
		{
			--k_it_;
			b_it_ = data_->find(*k_it_);

		} while (k_it_ != k_it_end_ && (b_it_ == data_->end()));
	}
}
;

template<typename ...Args>
auto make_iterator_mapped(
		Args&& ... args)
				DECL_RET_TYPE((make_iterator<_iterator_policy_mapped>( std::forward<Args>(args)...)))

template<typename ...Args>
auto make_range_mapped(
		Args&& ... args)
				DECL_RET_TYPE((make_range<_iterator_policy_mapped>( std::forward<Args>(args)...)))

}
// namespace simpla

#endif /* SP_ITERATOR_MAPPED_H_ */
