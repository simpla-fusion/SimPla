/**
 * @file implicit_function.h
 *
 * @date 2015-6-2
 * @author salmon
 */

#ifndef CORE_GEOMETRY_IMPLICIT_FUNCTION_H_
#define CORE_GEOMETRY_IMPLICIT_FUNCTION_H_
#include "simpla/algebra/Expression.h"
namespace simpla
{

template<typename ...> class ImplicitFunction;
template<typename TFun>
ImplicitFunction<TFun> make_implicit_function(TFun const & fun)
{
	return ImplicitFunction<TFun>(fun);
}

/**
 *  Implicit function of BaseManifold object
 *
 *  ImplicitFunction<TF>(x,y,z) return distance from (x,y,z)to object
 *
 *  negative mean inside,positive means outside
 *
 */
template<typename TF>
class ImplicitFunction<TF>
{
	typedef TF function_type;
	function_type m_fun_;

	ImplicitFunction(function_type const &fun) :
			m_fun_(fun)
	{
	}

	~ImplicitFunction()
	{
	}

	template<typename ...Args>
	Real operator()(Args && ...args) const
	{
		return static_cast<Real>(m_fun_(std::forward<Args>(args)...));
	}
};

template<typename ...>class Expression;

template<typename TL>
struct ImplicitFunction<Expression<_impl::logical_not, TL, std::nullptr_t>> : public Expression<
		_impl::logical_not, TL, std::nullptr_t>
{

	typedef ImplicitFunction<Expression<_impl::logical_not, TL, std::nullptr_t>> this_type;
	typedef Expression<_impl::logical_not, TL, std::nullptr_t> base_type;
	using base_type::Expression;

	template<typename ...Args>
	Real operator()(Args && ...args) const
	{
		return -base_type::lhs(std::forward<Args>(args)...);
	}
};

template<typename ...T>
ImplicitFunction<
		Expression<_impl::logical_not, ImplicitFunction<T...>, std::nullptr_t>> operator !(
		ImplicitFunction<T...> const &l)
{
	return (ImplicitFunction<
			Expression<_impl::logical_not, ImplicitFunction<T...>,
					std::nullptr_t>>(l));
}

template<typename TL, typename TR>
struct ImplicitFunction<Expression<_impl::logical_and, TL, TR>> : public Expression<
		_impl::logical_and, TL, TR>
{
public:

	typedef ImplicitFunction<Expression<TOP, TL, TR>> this_type;

	typedef Expression<_impl::logical_and, TL, std::nullptr_t> base_type;

	using base_type::Expression;

	template<typename ...Args>
	Real operator()(Args && ...args) const
	{
		return std::max(base_type::lhs(std::forward<Args>(args)...),
				base_type::rhs(std::forward<Args>(args)...));
	}
};

template<typename ... T1, typename ...T2>
ImplicitFunction<
		Expression<_impl::logical_and, ImplicitFunction<T1...>,
				ImplicitFunction<T2...>>> operator &&(
		ImplicitFunction<T1...> const & l, ImplicitFunction<T2...> const &r)
{
	return (ImplicitFunction<
			Expression<_impl::logical_and, ImplicitFunction<T1...>,
			ImplicitFunction<T2...>>>(l,r));
}
template<typename TL, typename TR>
struct ImplicitFunction<Expression<_impl::logical_or, TL, TR>> : public Expression<
		_impl::logical_and, TL, TR>
{
public:

	typedef ImplicitFunction<Expression<_impl::logical_or, TL, TR>> this_type;

	typedef Expression<_impl::logical_and, TL, std::nullptr_t> base_type;

	using base_type::Expression;

	template<typename ...Args>
	Real operator()(Args && ...args) const
	{
		return std::min(base_type::lhs(std::forward<Args>(args)...),
				base_type::rhs(std::forward<Args>(args)...));
	}
};

template<typename ... T1, typename ...T2>
ImplicitFunction<
		Expression<_impl::logical_or, ImplicitFunction<T1...>,
				ImplicitFunction<T2...>>> operator ||(
		ImplicitFunction<T1...> const & l, ImplicitFunction<T2...> const &r)
{
	return (ImplicitFunction<
			Expression<_impl::logical_or, ImplicitFunction<T1...>,
					ImplicitFunction<T2...>>>(l,r));
}

template<typename TL>
struct ImplicitFunction<Expression<_impl::negate, TL, std::nullptr_t>> : public Expression<
		_impl::negate, TL, std::nullptr_t>
{

	typedef ImplicitFunction<Expression<_impl::negate, TL, std::nullptr_t>> this_type;
	typedef Expression<_impl::negate, TL, std::nullptr_t> base_type;
	using Expression<_impl::negate, TL, std::nullptr_t>::Expression;

	template<typename ...Args>
	Real operator()(Args && ...args) const
	{
		return -base_type::lhs(std::forward<Args>(args)...);
	}
};
template<typename ...T>
ImplicitFunction<
		Expression<_impl::negate, ImplicitFunction<T...>, std::nullptr_t>> operator -(
		ImplicitFunction<T...> const &l)
{
	return (ImplicitFunction<
			Expression<_impl::negate, ImplicitFunction<T...>, std::nullptr_t>>(
			l));
}
template<typename TL, typename TR>
struct ImplicitFunction<Expression<_impl::plus, TL, TR>> : public Expression<
		_impl::plus, TL, TR>
{
public:

	typedef ImplicitFunction<Expression<_impl::plus, TL, TR>> this_type;

	typedef Expression<_impl::plus, TL, std::nullptr_t> base_type;

	using base_type::Expression;

	template<typename ...Args>
	Real operator()(Args && ...args) const
	{
		return std::min(base_type::lhs(std::forward<Args>(args)...),
				base_type::rhs(std::forward<Args>(args)...));
	}
};

template<typename ... T1, typename ...T2>
ImplicitFunction<
		Expression<_impl::plus, ImplicitFunction<T1...>, ImplicitFunction<T2...>>>\
 operator +(ImplicitFunction< T1...> const & l,ImplicitFunction< T2...> const &r)
{
	return (ImplicitFunction<Expression< _impl::plus,ImplicitFunction< T1...>,
			ImplicitFunction< T2...>>>(l,r));
}
template<typename TL, typename TR>
struct ImplicitFunction<Expression<_impl::minus, TL, TR>> : public Expression<
		_impl::logical_and, TL, TR>
{
public:

	typedef ImplicitFunction<Expression<_impl::minus, TL, TR>> this_type;

	typedef Expression<_impl::logical_and, TL, std::nullptr_t> base_type;

	using base_type::Expression;

	template<typename ...Args>
	Real operator()(Args && ...args) const
	{
		return std::max(base_type::lhs(std::forward<Args>(args)...),
				-base_type::rhs(std::forward<Args>(args)...));
	}
};

template<typename ... T1, typename ...T2>
ImplicitFunction<
		Expression<_impl::minus, ImplicitFunction<T1...>,
				ImplicitFunction<T2...>>> operator-(
		ImplicitFunction<T1...> const & l, ImplicitFunction<T2...> const &r)
{
	return (ImplicitFunction<
			Expression<_impl::minus, ImplicitFunction<T1...>,
					ImplicitFunction<T2...>>>(l, r));
}

}
// namespace simpla

#endif /* CORE_GEOMETRY_IMPLICIT_FUNCTION_H_ */
