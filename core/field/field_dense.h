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

	}
	template<typename T>
	void fill(T const &v)
	{
		wait();

		std::fill(m_data_.get(), m_data_.get() + m_domain_.max_hash(), v);
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
		wait();

		m_domain_.for_each([&](id_type const &s)
		{
			at(s) =other.at(s);
		});

		return *this;
	}

	template<typename ...Others>
	inline this_type & operator =(
			_Field<domain_type, value_type, Others...> const &other)
	{
		if (other.is_valid())
		{
			other.domain().for_each([&](id_type const &s)
			{
				at(s) = other[s];
			});
		}
		return *this;
	}
	template<typename ...Others>
	inline this_type & operator +=(
			_Field<domain_type, value_type, Others...> const &other)
	{
		wait();
		if (other.is_valid())
		{
			other.domain().for_each([&](id_type const &s)
			{
				at(s)+= other[s];
			});
		}
		return *this;
	}

	template<typename TR>
	inline this_type & operator =(TR const &other)
	{
		wait();
		if (other.domain().is_valid())
		{
			m_domain_.for_each([&](id_type const &s)
			{
				at(s) = m_domain_.mesh().calculate(other, s);
			});
		}
		return *this;
	}
private:
	template<typename T, typename TOther>
	void assign(T const & d, TOther const & other)
	{
		wait();

		d.for_each([&](id_type const &s)
		{
			at(s) = m_domain_.mesh().calculate(other, s);
		});

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

		SpObject::prepare_sync(m_domain_.ghost_shape());
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
		return this->m_domain_.mesh().gather(*this, std::forward<Args>(args)...);
	}

	template<typename TDomain, typename TFun>
	void pull_back(TDomain const & domain, TFun const & fun)
	{
		wait();

		Real t = m_domain_.mesh().time();
		for (auto s : domain)
		{
			this->at(s) = m_domain_.mesh().sample(
					fun(m_domain_.mesh().coordinates(s), t), s);
		}
	}

	template<typename TFun> void pull_back(TFun const &fun)
	{
		pull_back(m_domain_.template range<iform>(), fun);
	}

	template<typename TDomain, typename TFun>
	void pull_back(std::tuple<TDomain, TFun> const & fun)
	{
		pull_back(std::get<0>(fun), std::get<1>(fun));
	}

	template<typename TDomain, typename TFun>
	void constraint(TDomain const & domain, TFun const & fun)
	{
		Real t = m_domain_.mesh().time();
		for (auto s : domain)
		{
			auto x = m_domain_.mesh().coordinates(s);
			this->at(s) = m_domain_.mesh().sample(
					fun(x, t, this->operator()(x)), s);
		}
	}

	template<typename TDomain, typename TFun>
	void constraint(std::tuple<TDomain, TFun> const & fun)
	{
		constraint(std::get<0>(fun), std::get<1>(fun));
	}

}
;

/**@} */

}
// namespace simpla

#endif /* CORE_FIELD_FIELD_DENSE_H_ */
