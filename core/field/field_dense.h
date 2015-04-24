/**
 * @file field_dense.h
 *
 *  Created on: @date{ 2015-1-30}
 *      @author: salmon
 */

#ifndef CORE_FIELD_FIELD_DENSE_H_
#define CORE_FIELD_FIELD_DENSE_H_

#include <algorithm>
#include <cstdbool>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../application/sp_object.h"
#include "../dataset/dataset.h"
#include "../gtl/ntuple.h"
#include "../gtl/primitives.h"
#include "../gtl/properties.h"
#include "../gtl/type_traits.h"
#include "../parallel/mpi_update.h"

#include "field_expression.h"

namespace simpla
{

template<typename, size_t> struct Domain;

/**
 * @ingroup field
 * @{
 */

/**
 *  Simple Field
 */
template<typename TM, size_t IFORM, typename TV>
struct _Field<Domain<TM, IFORM>, TV, _impl::is_sequence_container> : public SpObject
{
public:
	typedef Domain<TM, IFORM> domain_type;
	typedef typename domain_type::mesh_type mesh_type;
	static constexpr size_t iform = domain_type::iform;
	typedef typename mesh_type::id_type id_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef TV value_type;
	typedef typename domain_type::template field_value_type<value_type> field_value_type;

	typedef _Field<domain_type, value_type, _impl::is_sequence_container> this_type;

private:

	domain_type m_domain_;
	std::shared_ptr<TV> m_data_;

public:

	_Field(domain_type const & d)
			: m_domain_(d), m_data_(nullptr)
	{
	}
	_Field(this_type const & other)
			: m_domain_(other.m_domain_), m_data_(other.m_data_)
	{
	}
	_Field(this_type && other)
			: m_domain_(other.m_domain_), m_data_(other.m_data_)
	{
	}
	~_Field()
	{
	}
	void swap(this_type & other)
	{
		std::swap(m_domain_, other.m_domain_);
		std::swap(m_data_, other.m_data_);
	}
	std::string get_type_as_string() const
	{
		return "Field<" + m_domain_.mesh().get_type_as_string() + ">";
	}
	mesh_type const & mesh() const
	{
		return m_domain_.mesh();
	}
	domain_type const & domain() const
	{
		return m_domain_;
	}

	std::shared_ptr<value_type> data()
	{
		return m_data_;
	}

	void clear()
	{
		wait();
		value_type t;
		t = 0;

		std::fill(m_data_.get(), m_data_.get() + m_domain_.max_hash(), t);

		sync();

	}
	template<typename T>
	void fill(T const &v)
	{
		wait();

		std::fill(m_data_.get(), m_data_.get() + m_domain_.max_hash(), v);

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
	/**@}*/

	/**
	 * @name assignment
	 * @{
	 */

	inline this_type & operator =(this_type const &other)
	{
		assign(other, _impl::_assign());
		return *this;
	}

	template<typename Other>
	inline this_type & operator =(Other const &other)
	{
		assign(other, _impl::_assign());
		return *this;
	}

	template<typename Other>
	inline this_type & operator+=(Other const &other)
	{
		assign(other, _impl::plus_assign());
		return *this;
	}
	template<typename Other>
	inline this_type & operator-=(Other const &other)
	{
		assign(other, _impl::minus_assign());
		return *this;
	}
	template<typename Other>
	inline this_type & operator*=(Other const &other)
	{
		assign(other, _impl::multiplies_assign());
		return *this;
	}
	template<typename Other>
	inline this_type & operator/=(Other const &other)
	{
		assign(other, _impl::divides_assign());
		return *this;
	}

	template<typename TOther, typename TOP>
	void assign(TOther const & other, TOP const &op)
	{

		wait();

		mesh_type const & m = mesh();
		m_domain_.for_each([&](id_type const &s)
		{
			op(at(s), m.calculate(other, s));
		});

		sync();
	}

	template<typename ...T, typename TOP>
	void assign(_Field<domain_type, T...> const & other, TOP const &op)
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
	void self_assign(TFun const & other)
	{

		wait();

		if (!other.domain().is_null())
		{
			m_domain_.for_each(other.domain(),
					[&](id_type const &s)
					{
						auto x=m_domain_.mesh().coordinates(s);
						auto t=m_domain_.mesh().time();
						at(s)+=m_domain_.mesh().template sample<iform>(s,other(x,t,gather(x)));
					});
		}

		sync();
	}

public:

	/** @} */

	/** @name access
	 *  @{*/

	field_value_type gather(coordinates_type const& x) const
	{
		return std::move(m_domain_.mesh().gather(*this, x));
	}

	template<typename ...Args>
	void scatter(Args && ... args)
	{
		m_domain_.mesh().scatter(*this, std::forward<Args>(args)...);
	}

	/**@}*/

	void deploy()
	{
		if (m_data_ == nullptr)
		{
			m_data_ = sp_make_shared_array<value_type>(m_domain_.max_hash());
		}

		SpObject::prepare_sync(m_domain_.mesh().template ghost_shape<iform>());
	}

	DataSet dataset() const
	{
		//ASSERT(is_ready());

		DataSet res;

		res.data = m_data_;

		res.datatype = DataType::create<value_type>();

		res.dataspace = m_domain_.dataspace();

		res.properties = SpObject::properties;

		return std::move(res);
	}

	template<typename TFun>
	void for_each(TFun const& fun)
	{
		wait();

		for (auto s : m_domain_)
		{
			fun(m_data_.get()[m_domain_.hash(s)]);
		}

	}

	template<typename TFun>
	void for_each(TFun const& fun) const
	{
//		ASSERT(is_ready());

		for (auto s : m_domain_)
		{
			fun(m_data_.get()[m_domain_.hash(s)]);
		}
	}

public:
	template<typename IndexType>
	value_type & operator[](IndexType const & s)
	{
		return m_data_.get()[m_domain_.hash(s)];
	}
	template<typename IndexType>
	value_type const & operator[](IndexType const & s) const
	{
		return m_data_.get()[m_domain_.hash(s)];
	}

	template<typename ...Args>
	value_type & at(Args && ... args)
	{
		return (m_data_.get()[m_domain_.hash(std::forward<Args>(args)...)]);
	}

	template<typename ...Args>
	value_type const & at(Args && ... args) const
	{
		return (m_data_.get()[m_domain_.hash(std::forward<Args>(args)...)]);
	}

	template<typename ...Args>
	field_value_type operator()(Args && ...args) const
	{
		return m_domain_.mesh().gather(*this, std::forward<Args>(args)...);
	}

	template<typename OS>
	OS & print(OS & os) const
	{
		return dataset().print(os);
	}
}
;
template<typename TM, size_t IFORM, typename ...Others>
struct field_traits<_Field<Domain<TM, IFORM>, Others...>>
{
	static constexpr bool is_field = true;

	typedef Domain<TM, IFORM> domain_type;

	typedef typename _Field<Domain<TM, IFORM>, Others...>::value_type value_type;

	static constexpr size_t iform = domain_type::iform;

	static constexpr size_t ndims = domain_type::ndims;

};

/**@} */

}
// namespace simpla

#endif /* CORE_FIELD_FIELD_DENSE_H_ */
