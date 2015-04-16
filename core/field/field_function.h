/*
 * @file field_function.h
 *
 *  Created on: 2015年3月10日
 *      Author: salmon
 */

#ifndef CORE_FIELD_FIELD_FUNCTION_H_
#define CORE_FIELD_FIELD_FUNCTION_H_
#include "../gtl/type_traits.h"
namespace simpla
{
template<typename ...>class _Field;

template<typename TM, typename TV, typename TDomain, typename TFun> class _Field<
		TM, TV, _impl::is_function, TDomain, TFun>
{
	typedef TM mesh_type;
	typedef TV value_type;
	typedef TFun function_type;
	typedef TDomain domain_type;

	typedef _Field<mesh_type, value_type, _impl::is_function, domain_type,
			function_type> this_type;

	typedef typename mesh_type::id_type id_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::template field_value_type<TV> field_value_type;

private:
	function_type m_fun_;
	domain_type m_domain_;

	mesh_type m_mesh_;
public:
	template<typename TD, typename TF> _Field(mesh_type const & mesh,
			TD const& domain, TF const& fun)
			: m_mesh_(mesh), m_domain_(domain), m_fun_(fun)
	{
	}

	_Field(this_type && other)
			: m_mesh_(other.m_mesh_), m_domain_(other.m_domain_), m_fun_(
					other.m_fun_)
	{
	}
	~_Field()
	{
	}

	bool is_valid() const
	{
		return !(!m_fun_) && m_domain_.size() > 0;
	}

	domain_type const & domain() const
	{
		return m_domain_;
	}

	field_value_type operator()(coordinates_type const& x, Real t) const
	{
		return static_cast<field_value_type>(m_fun_(x, t));
	}

	template<typename ...Others> field_value_type operator()(
			coordinates_type const& x, Real t, Others const&... others) const
	{
		return static_cast<field_value_type>(m_fun_(x, t,
				try_invoke(others,x, t)...));
	}

	value_type operator[](id_type s) const
	{
		Real t = m_mesh_.time();
		return m_mesh_.sample(operator()(m_mesh_.coordinates(s), t), s);
	}

};

template<typename TV, typename TM, typename TDomain, typename TFun> _Field<TM,
		TV, _impl::is_function, TDomain, TFun> make_field_function(
		TM const & mesh, TDomain && domain, TFun && fun)
{
	return _Field<TM, TV, _impl::is_function, TDomain, TFun>(mesh, domain, fun);
}

}
// namespace simpla

#endif /* CORE_FIELD_FIELD_FUNCTION_H_ */
