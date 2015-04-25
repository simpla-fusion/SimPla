/*
 * @file field_function.h
 *
 *  Created on: 2015年3月10日
 *      Author: salmon
 */

#ifndef CORE_FIELD_FIELD_FUNCTION_H_
#define CORE_FIELD_FIELD_FUNCTION_H_
#include "../gtl/type_traits.h"
#include "../model/select.h"
namespace simpla
{
template<typename ...>class _Field;

template<typename _T, typename ... _Args>
struct is_callable
{
private:
	typedef std::true_type yes;
	typedef std::false_type no;

	template<typename _U>
	static auto test(int) ->
	decltype(std::declval<_U>() (std::declval<_Args>() ...));

	template<typename > static no test(...);

public:

	static constexpr bool value =
			(!std::is_same<decltype(test<_T>()), no>::value);

};

template<typename TFun, typename ... Args>
auto try_invoke(TFun const & fun, Args &&...args)->
typename std::result_of<TFun( Args &&...)>::type
{
	return (fun(std::forward<Args>(args)...));
}

//template<typename TFun, typename ... Args>
//auto try_invoke(TFun const & fun, Args &&...args)->
//typename std::enable_if<!is_callable<TFun,Args&&...>::value,TFun>::type
//{
//	return fun;
//}
template<typename TDomain, typename TV, typename TFun>
class _Field<TDomain, TV, _impl::is_function, TFun>
{
public:
	typedef TDomain domain_type;
	typedef typename domain_type::mesh_type mesh_type;
	typedef TV value_type;
	typedef TFun function_type;

	typedef _Field<domain_type, value_type, _impl::is_function, function_type> this_type;

	typedef typename mesh_type::id_type id_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename domain_type::template field_value_type<value_type> field_value_type;

	static constexpr size_t iform = domain_type::iform;
	static constexpr size_t ndims = domain_type::ndims;

private:
	function_type m_fun_;
	domain_type m_domain_;
public:
	_Field()
	{
	}
	_Field(domain_type const& domain) :
			m_domain_(domain)
	{
	}
	template<typename TF>
	_Field(domain_type const& domain, TF const& fun) :
			m_domain_(domain), m_fun_(fun)
	{
	}
	_Field(this_type const& other) :
			m_domain_(other.m_domain_), m_fun_(other.m_fun_)
	{
	}
	_Field(this_type && other) :
			m_domain_(other.m_domain_), m_fun_(other.m_fun_)
	{
	}
	~_Field()
	{
	}

	bool is_valid() const
	{
		return (!!m_fun_) && m_domain_.is_valid();
	}
	operator bool() const
	{
		return !!m_fun_;
	}
	domain_type const & domain() const
	{
		return m_domain_;
	}

	value_type operator[](id_type s) const
	{
		Real t = m_domain_.mesh().time();

		return m_domain_.mesh().template sample<iform>(s,
				static_cast<field_value_type>(m_fun_(
						m_domain_.mesh().coordinates(s), t)));
	}

	field_value_type operator()(coordinates_type const& x, Real t) const
	{
		return static_cast<field_value_type>(m_fun_(x, t));
	}

	template<typename ...Others>
	field_value_type operator()(coordinates_type const& x, Real t,
			Others &&... others) const
	{
		return static_cast<field_value_type>(m_fun_(x, t,
				std::forward<Others>(others )...));
	}

	/**
	 *
	 * @param args
	 * @return (x,t) -> m_fun_(x,t,args(x,t))
	 */
	template<typename ...Args>
	_Field<domain_type, value_type, _impl::is_function,
			std::function<field_value_type(coordinates_type const&, Real)>> op_on(
			Args const& ...args) const
	{
		typedef std::function<field_value_type(coordinates_type const&, Real)> res_function_type;
		typedef _Field<domain_type, value_type, _impl::is_function,
				res_function_type> res_type;

		res_function_type fun = [ &](coordinates_type const& x, Real t)
		{
			return static_cast<field_value_type>(m_fun_(x, t,
							static_cast<field_value_type>( (args)( x ))...));
		};

		return res_type(m_domain_, fun);

	}

};

template<typename TV, typename TDomain, typename TFun>
_Field<TDomain, TV, _impl::is_function, TFun> make_field_function(
		TDomain const& domain, TFun const& fun)
{
	return std::move(_Field<TDomain, TV, _impl::is_function, TFun>(domain, fun));
}

template<size_t IFORM, typename TV, typename TM, typename TDict>
_Field<Domain<TM, IFORM>, TV, _impl::is_function, TDict> //
make_field_function_by_config(TM const & mesh, TDict const & dict)
{
	typedef TV value_type;

	typedef Domain<TM, IFORM> domain_type;

	typedef _Field<domain_type, value_type, _impl::is_function, TDict> field_type;

	// TODO create null filed

	domain_type domain = mesh.template domain<field_type::iform>();

	if (dict["Domain"])
	{

		filter_domain_by_config(dict["Domain"], &domain);

		return field_type(domain, dict["Value"]);
	}
	else
	{
		domain.clear();
		return field_type(domain);

	}

}

}
// namespace simpla

#endif /* CORE_FIELD_FIELD_FUNCTION_H_ */
