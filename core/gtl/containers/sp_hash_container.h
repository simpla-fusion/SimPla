/**
 * @file sp_hash_container.h
 *
 * @date 2015-3-20
 * @author salmon
 */

#ifndef CORE_GTL_CONTAINERS_SP_HASH_CONTAINER_H_
#define CORE_GTL_CONTAINERS_SP_HASH_CONTAINER_H_

namespace simpla
{namespace gtl
{

template<typename Key, typename Value, typename Hash>
struct SpHashContainer
{

	typedef Key key_type;
	typedef Value value_type;
	typedef Hash hash_function;

	typedef SpHashContainer<key_type, value_type, hash_function> this_type;

	std::shared_ptr<value_type> m_data_;
	size_t m_max_hash_ = 0;
	hash_function m_hash_;

	SpHashContainer()
			: m_data_(nullptr)
	{
	}
	SpHashContainer(hash_function const & fun, size_t max_hash = 0)
			: m_data_(nullptr), m_max_hash_(max_hash), m_hash_(fun)
	{
		deploy();
	}
	SpHashContainer(this_type const & other)
			: m_data_(other.m_data_), m_max_hash_(other.m_max_hash_), m_hash_(
					other.m_hash_)
	{
		deploy();
	}
	SpHashContainer(this_type && other)
			: m_data_(other.m_data_), m_max_hash_(other.m_max_hash_), m_hash_(
					other.m_hash_)
	{
		deploy();
	}
	~SpHashContainer()
	{
	}

	void swap(this_type & other)
	{
		std::swap(m_data_, other.m_data_);
		std::swap(m_max_hash_, other.m_max_hash_);
		std::swap(m_hash_, other.m_hash_);
	}
	std::shared_ptr<value_type> data()
	{
		return m_data_;
	}
	void max_hash(size_t m) const
	{
		m_max_hash_ = m;
	}
	size_t max_hash() const
	{
		return m_max_hash_;
	}
	size_t hash(key_type const & s) const
	{
		return m_hash_(s);
	}
	hash_function const & hasher() const
	{
		return m_hash_;
	}

	template<typename ...Args>
	void hasher(Args && ...args) const
	{
		m_hash_ = hash_function(std::forward<Args>(args)...);
	}

	void hasher(hash_function const & fun) const
	{
		m_hash_ = fun;
	}

	this_type & operator=(this_type const & other)
	{
		this_type(other).swap(*this);
		return *this;
	}

	constexpr size_t size() const
	{
		return m_max_hash_;
	}
	void fill(value_type const & v)
	{
		std::fill(m_data_.get(), m_data_.get() + size(), v);
	}

	bool empty() const
	{
		return size() == 0 || m_data_ == nullptr;
	}

	void deploy()
	{
		if (m_data_ == nullptr)
		{
			m_data_ = sp_make_shared_array<value_type>(size());
		}

	}

	value_type & operator[](key_type const& s)
	{
		return m_data_.get()[m_hash_(s)];
	}
	value_type const& operator[](key_type const& s) const
	{
		return m_data_.get()[m_hash_(s)];
	}
};

}  }//  namespace simpla::gtl

#endif /* CORE_GTL_CONTAINERS_SP_HASH_CONTAINER_H_ */
