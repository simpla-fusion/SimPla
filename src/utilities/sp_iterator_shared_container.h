/*
 * iterator_shared_container.h
 *
 *  Created on: 2014年6月20日
 *      Author: salmon
 */

#ifndef ITERATOR_SHARED_CONTAINER_H_
#define ITERATOR_SHARED_CONTAINER_H_
namespace std
{
template<typename TI> struct iterator_traits;

}  // namespace std
namespace simpla
{

template<typename > class ContainerTraits;
namespace _impl
{
template<typename in, typename out> struct IteratorConvert;

template<typename T>
struct IteratorConvert<T, T>
{
	static T const &convert(T const &it)
	{
		return it;
	}
};

template<typename T>
struct IteratorConvert<T**, T*>
{
	static T* convert(T** it)
	{
		return *it;
	}
};

template<typename T> struct StorageSharedContainer
{
	typedef T container_type;

	typedef StorageSharedContainer<container_type> this_type;

	typedef typename ContainerTraits<T>::value_type value_type;

	std::shared_ptr<T> data_;

	typedef value_type* pointer;
	typedef value_type& reference;

	template<typename TI>
	const reference get(TI const& it) const
	{
		return (*data_)(it);
	}

	template<typename TI>
	reference get(TI const& it)
	{
		return (*data_)(it);
	}

	template<typename TI>
	static reference get(this_type * self, TI const& it)
	{
		return self->get(it);
	}

	template<typename TI>
	static reference get(this_type const* self, TI const& it)
	{
		return self->get(it);
	}

};

template<typename T>
struct StorageSharedContainer<T*>
{
	typedef T* container_type;

	typedef StorageSharedContainer<container_type> this_type;

	typedef T value_type;

	value_type* data_;

	typedef value_type* pointer;
	typedef value_type& reference;

	const reference get(size_t it) const
	{
		return data_[it];
	}

	template<typename TI>
	reference get(TI const& it)
	{
		return data_[it];
	}

	template<typename TI>
	static reference get(this_type * self, TI const& it)
	{
		return self->get(it);
	}

	template<typename TI>
	static const reference get(this_type const* self, TI const& it)
	{
		return self->get(it);
	}

};

}  // namespace _impl

/**
 *   an iterator over the elements of some random-access conatiner,
 *   rearranged according to some hash/conversion of input iterator.
 *
 *   shared container
 *   filter
 *   map
 *   cache
 *
 */
template<typename TContainer, typename TKey,

template<typename, typename > class IteratorConvertPolicy = _impl::IteratorConvert,

template<typename > class StorageContainerPolicy = _impl::StorageSharedContainer

>
class TransformIterator: public StorageContainerPolicy<TContainer>
{
public:
	typedef TContainer conatiner_type;

	typedef TKey key_iterator_type;

	typedef IteratorConvertPolicy<key_iterator_type, typename ContainerTraits<container_type>::index_type> convert_type;

	typedef StorageContainerPolicy<container_type> storage_type;

	typedef typename std::iterator_traits<key_iterator_type>::iterator_category iterator_category;
	typedef typename std::iterator_traits<key_iterator_type>::difference_type difference_type;

	typedef typename ContainerTraits<conatiner_type>::value_type value_type;

	typedef value_type* pointer;
	typedef value_type& reference;

	key_iterator_type k_it_;

	TransformIterator(container_type & container, key_iterator_type const & ib)
			: storage_type(container), k_it_(ib)
	{
	}
	TransformIterator(this_type const & other)
			: storage_type(other), k_it_(other.k_it_)
	{
	}

	TransformIterator(conatiner_type &d, key_iterator const & ib, key_iterator const&)
			: data_(&d), k_it_(ib)
	{
	}

	~TransformIterator()
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
		return get();
	}
	pointer operator->()
	{
		return &get();
	}

	reference operator*() const
	{
		return get();
	}
	pointer operator->() const
	{
		return &get();

	}
	const reference get() const
	{
		return storage_type::get(*this, convert_type::convert(k_it_));
	}

	reference get()
	{
		return storage_type::get(*this, convert_type::convert(k_it_));
	}

};
}  // namespace simpla

#endif /* ITERATOR_SHARED_CONTAINER_H_ */
