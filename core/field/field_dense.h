/**
 * @file field_dense.h
 *
 *  Created on: @date{ 2015-1-30}
 *      @author: salmon
 */

#ifndef FIELD_DENSE_H_
#define FIELD_DENSE_H_

#include "field_comm.h"


#include <algorithm>
#include <cstdbool>
#include <memory>
#include <string>

#include "../gtl/type_traits.h"
#include "../parallel/distributed_array.h"
#include "../manifold/domain_traits.h"
#include "../manifold/manifold_traits.h"
#include "../dataset/dataset_traits.h"

namespace simpla
{
template<typename ...> struct Field;

/**
 * @ingroup field
 * @{
 */

/**
 *  Simple Field
 */
template<typename TG, int IFORM, typename TV>
struct Field<Domain<TG, std::integral_constant<int, IFORM>>, TV>
		: public DistributedArray
{
public:

	typedef TV value_type;
	typedef TG mesh_type;
	static constexpr int iform = IFORM;
	typedef DistributedArray storage_policy;

	typedef Domain<mesh_type, std::integral_constant<int, IFORM>> domain_type;

	typedef typename mesh_type::id_type id_type;

	typedef typename mesh_type::point_type point_type;

	typedef Field<domain_type, value_type> this_type;


private:

	domain_type m_domain_;

	mesh_type const &m_mesh_;

public:

	Field(domain_type const &d)
			: storage_policy(traits::datatype<value_type>::create(), d.m_mesh_.template dataspace<iform>()),
			m_domain_(d), m_mesh_(d.mesh())
	{
	}

	Field(this_type const &other)
			: storage_policy(other), m_domain_(other.m_domain_), m_mesh_(other.m_mesh_)
	{
	}

	Field(this_type &&other)
			: storage_policy(other), m_domain_(other.m_domain_), m_mesh_(other.m_mesh_)
	{
	}

	~Field()
	{
	}


	void swap(this_type &other)
	{
		std::swap(m_domain_, other.m_domain_);
		std::swap(m_mesh_, other.m_mesh_);
		storage_policy::swap(other);
	}


	domain_type const &domain() const
	{
		return m_domain_;
	}

	domain_type &domain()
	{
		return m_domain_;
	}


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
		storage_policy::deploy();

		parallel::wait(this);

		m_domain_.for_each([&](id_type const &s)
		{
			op(at(s), m_mesh_.calculate(other, s));
		});

		parallel::sync(this);
	}

	template<typename ...T, typename TOP>
	void assign(Field<domain_type, T...> const &other, TOP const &op)
	{
		storage_policy::deploy();

		parallel::wait(this);

		if (!other.domain().is_null())
		{
			m_domain_.for_each(other.domain(), [&](id_type const &s)
			{
				op(at(s), other[s]);
			});
		}
		parallel::sync(this);
	}

	template<typename TFun>
	void self_assign(TFun const &other)
	{
		storage_policy::deploy();

		parallel::wait(this);

		if (!other.domain().is_null())
		{
			m_domain_.for_each(other.domain(),
					[&](id_type const &s)
					{
						auto x = m_mesh_.point(s);
						at(s) += m_mesh_.template sample<iform>(s, other(x, m_mesh_.time(), gather(x)));
					});
		}

		parallel::sync(this);
	}

	template<typename ...T> void assign(
			Field<domain_type, value_type, tags::function, T...> const &other)
	{
		storage_policy::deploy();

		parallel::wait(this);

		if (!other.domain().is_null())
		{
			other.domain().for_each(
					[&](id_type const &s)
					{
						auto x = m_mesh_.point(s);
						at(s) = m_mesh_.template sample<iform>(s, other(x, m_mesh_.time(), gather(x)));
					});
		}

		parallel::sync(this);
	}

	/** @} */



	template<typename TFun>
	void for_each(TFun const &fun)
	{
		storage_policy::deploy();

		parallel::wait(this);

		for (auto s : m_domain_)
		{
			fun(at(s));
		}

	}

	template<typename TFun>
	void for_each(TFun const &fun) const
	{
//		ASSERT(is_ready());
		parallel::wait(this);
		for (auto s : m_domain_)
		{
			fun(at(s));
		}
	}

public:
	/**
	 *  @name as_array
	 *  @{
	 */
	value_type &operator[](id_type const &s)
	{
		return storage_policy::get_value<value_type>(m_mesh_.hash(s));
	}

	value_type const &operator[](id_type const &s) const
	{
		return storage_policy::get_value<value_type>(m_mesh_.hash(s));
	}

	template<typename ...Args>
	value_type &at(Args &&... args)
	{
		return storage_policy::get_value<value_type>(m_mesh_.hash(std::forward<Args>(args)...));
	}

	template<typename ...Args>
	value_type const &at(Args &&... args) const
	{
		return storage_policy::get_value<value_type>(m_mesh_.hash(std::forward<Args>(args)...));
	}
	/**
	 * @}
	 */

	/** @name as_function
	 *  @{*/


	template<typename ...Args>
	void scatter(Args &&... args)
	{
		m_mesh_.scatter(*this, std::forward<Args>(args)...);
	}

	auto gather(point_type const &x) const
	DECL_RET_TYPE((m_mesh_.gather(*this, x)))

	template<typename ...Args>
	auto operator()(Args &&...args) const
	DECL_RET_TYPE((m_mesh_.gather(*this, std::forward<Args>(args)...)))


	/**@}*/

}; // struct Field

namespace traits
{

template<typename ... TM, typename ...Others>
struct type_id<Field<Domain<TM ...>, Others...>>
{
	static const std::string name()
	{
		return "Feild<" + type_id<Domain<TM...> >::name() + "," + type_id<Others...>::name() + ">";
	}
};


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

}// namespace traits

}// namespace simpla

#endif /* FIELD_DENSE_H_ */
