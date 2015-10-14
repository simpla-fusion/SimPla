/**
 * @file field_dense.h
 *
 *  Created on: @date{ 2015-1-30}
 *      @author: salmon
 */

#ifndef COREFieldField_DENSE_H_
#define COREFieldField_DENSE_H_

#include <algorithm>
#include <cstdbool>
#include <memory>
#include <string>

#include "../application/sp_object.h"
#include "../gtl/properties.h"
#include "../gtl/type_traits.h"

namespace simpla
{

/**
 * @ingroup field
 * @{
 */

/**
 *  Simple Field
 */
template<typename TD, typename TV, typename ...TAGS>
struct Field<TD, TV, TAGS...> : public SpObject
{
public:

	typedef TV value_type;

	typedef TD domain_type;

	typedef typename domain_type::id_type id_type;

	typedef typename domain_type::point_type point_type;

	typedef Field<domain_type, value_type, TAGS...> this_type;

	typedef traits::field_value_t<this_type> field_value_type;

private:

	domain_type m_domain_;
	std::shared_ptr<value_type> m_data_;

public:

	Field(domain_type const &d)
			: m_domain_(d), m_data_(nullptr)
	{
	}

	Field(this_type const &other)
			: m_domain_(other.m_domain_), m_data_(other.m_data_)
	{
	}

	Field(this_type &&other)
			: m_domain_(other.m_domain_), m_data_(other.m_data_)
	{
	}

	~Field()
	{
	}


	void swap(this_type &other)
	{
		std::swap(m_domain_, other.m_domain_);
		std::swap(m_data_, other.m_data_);
	}

	domain_type const &domain() const
	{
		return m_domain_;
	}

	domain_type &domain()
	{
		return m_domain_;
	}

	std::shared_ptr<value_type> &data() { return m_data_; }

	std::shared_ptr<const value_type> data() const { return m_data_; }

	void clear()
	{
		wait();
		value_type t;
		t = 0;

//		std::fill(m_data_.get(), m_data_.get() + m_domain_.max_hash(), t);

		sync();

	}

	template<typename T>
	void fill(T const &v)
	{
		wait();

//		std::fill(m_data_.get(), m_data_.get() + m_domain_.max_hash(), v);

		sync();

	}

	/** @name range concept
	 * @{
	 */

	bool empty() const
	{
		return m_data_ == nullptr;
	}


	bool is_valid() const
	{
		return m_data_ != nullptr;
	}

	std::string get_type_as_string() const { return "Field"; }


	void deploy()
	{
		if (m_data_ == nullptr)
		{
			m_data_ = sp_make_shared_array<value_type>(m_domain_.max_hash());
		}

		SpObject::prepare_sync(m_domain_.ghost_shape());
	}

	DataSet dataset() const
	{
		//ASSERT(is_ready());

		DataSet res;

		res.data = m_data_;

		res.datatype = traits::datatype<value_type>::create();

		res.dataspace = m_domain_.dataspace();

		res.properties = SpObject::properties;

		return std::move(res);
	}

	/**@}*/

	/**
	 * @name assignment
	 * @{
	 */

	inline this_type &operator=(this_type const &other)
	{
		assign(other, _impl::_assign());
		return *this;
	}

	template<typename Other>
	inline this_type &operator=(Other const &other)
	{
		assign(other, _impl::_assign());
		return *this;
	}

	template<typename Other>
	inline this_type &operator+=(Other const &other)
	{
		assign(other, _impl::plus_assign());
		return *this;
	}

	template<typename Other>
	inline this_type &operator-=(Other const &other)
	{
		assign(other, _impl::minus_assign());
		return *this;
	}

	template<typename Other>
	inline this_type &operator*=(Other const &other)
	{
		assign(other, _impl::multiplies_assign());
		return *this;
	}

	template<typename Other>
	inline this_type &operator/=(Other const &other)
	{
		assign(other, _impl::divides_assign());
		return *this;
	}

	template<typename TOther, typename TOP>
	void assign(TOther const &other, TOP const &op)
	{

		wait();

		m_domain_.for_each([&](id_type const &s)
		{
			op(at(s), m_domain_.calculate(other, s));
		});

		sync();
	}

	template<typename ...T, typename TOP>
	void assign(Field<domain_type, T...> const &other, TOP const &op)
	{

		wait();

		if (!other.domain().is_null())
		{
			m_domain_.for_each(other.domain(), [&](id_type const &s)
			{
				op(at(s), other[s]);
			});
		}
		sync();
	}

	template<typename TFun>
	void self_assign(TFun const &other)
	{

		wait();

		if (!other.domain().is_null())
		{
			m_domain_.for_each(other.domain(),
					[&](id_type const &s)
					{
						auto x = m_domain_.point(s);
						at(s) += m_domain_.sample(s, other(x, m_domain_.time(), gather(x)));
					});
		}

		sync();
	}

	template<typename ...T>
	void assign(
			Field<domain_type, value_type, tags::function, T...> const &other)
	{

		wait();

		if (!other.domain().is_null())
		{
			other.domain().for_each(
					[&](id_type const &s)
					{
						auto x = m_domain_.point(s);
						at(s) = m_domain_.sample(s, other(x, m_domain_.time(), gather(x)));
					});
		}

		sync();
	}

	/** @} */



	template<typename TFun>
	void for_each(TFun const &fun)
	{
		wait();

		for (auto s : m_domain_)
		{
			fun(m_data_.get()[m_domain_.hash(s)]);
		}

	}

	template<typename TFun>
	void for_each(TFun const &fun) const
	{
//		ASSERT(is_ready());

		for (auto s : m_domain_)
		{
			fun(m_data_.get()[m_domain_.hash(s)]);
		}
	}

public:
	/**
	 *  @name as_array
	 *  @{
	 */
	value_type &operator[](id_type const &s)
	{
		return m_data_.get()[m_domain_.hash(s)];
	}

	value_type const &operator[](id_type const &s) const
	{
		return m_data_.get()[m_domain_.hash(s)];
	}

	template<typename ...Args>
	value_type &at(Args &&... args)
	{
		return (m_data_.get()[m_domain_.hash(std::forward<Args>(args)...)]);
	}

	template<typename ...Args>
	value_type const &at(Args &&... args) const
	{
		return (m_data_.get()[m_domain_.hash(std::forward<Args>(args)...)]);
	}
	/**
	 * @}
	 */

	/** @name as_function
	 *  @{*/
	field_value_type gather(point_type const &x) const
	{
		return std::move(m_domain_.gather(*this, x));
	}

	template<typename ...Args>
	void scatter(Args &&... args)
	{
		m_domain_.scatter(*this, std::forward<Args>(args)...);
	}

	template<typename ...Args>
	field_value_type operator()(Args &&...args) const
	{
		return m_domain_.gather(*this, std::forward<Args>(args)...);
	}

	/**@}*/

};

namespace traits
{

template<typename ... TM, typename ...Others>
struct type_id<Field<Domain<TM ...>, Others...>>
{
	static const std::string name()
	{
		return Field<Domain<TM ...>, Others...>::get_type_as_string();
	}
};

template<typename OS, typename ... TM, typename ...Others>
OS print(OS &os, Field<Domain<TM ...>, Others...> const &f)
{
	return f.dataset().print(os);
}

template<typename ... TM, typename TV, typename ...Policies>
struct value_type<Field<Domain<TM ...>, TV, Policies...>>
{
	typedef TV type;
};

template<typename ... TM, typename TV, typename ...Policies>
struct domain_type<Field<Domain<TM ...>, TV, Policies...>>
{
	typedef Domain<TM ...> type;
};

}
// namespace traits

}// namespace simpla

#endif /* COREFieldField_DENSE_H_ */
