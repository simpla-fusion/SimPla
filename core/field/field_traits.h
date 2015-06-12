/**
 * @file field_traits.h
 *
 * @date 2015年6月12日
 * @author salmon
 */

#ifndef CORE_FIELD_FIELD_TRAITS_H_
#define CORE_FIELD_FIELD_TRAITS_H_

#include <stddef.h>
#include <type_traits>

#include "../gtl/integer_sequence.h"
#include "../gtl/type_traits.h"

namespace simpla
{

template<typename, size_t> struct Domain;
template<typename ... >struct _Field;

namespace tags
{

class sequence_container;

class associative_container;

class function;

}  // namespace tags
namespace traits
{

template<typename > struct is_field: public std::integral_constant<bool, false>
{
};

template<typename ...T> struct is_field<_Field<T...>> : public std::integral_constant<
		bool, true>
{
};
template<typename T>
struct is_domain: public std::integral_constant<bool, false>
{
};
template<typename TM, size_t IFORM>
struct is_domain<Domain<TM, IFORM>> : public std::integral_constant<bool, true>
{
};
template<typename TM, typename TV, typename ...Others>
struct reference<_Field<TM, TV, Others...> >
{
	typedef _Field<TM, TV, Others...> const & type;
};

template<typename ...T, size_t M>
struct extent<_Field<T ...>, M> : public std::integral_constant<size_t,
		simpla::_impl::seq_get<M, extents_t<_Field<T ...> >>::value>
{
};

template<typename ...T>
struct key_type<_Field<T ...> >
{
	typedef size_t type;
};

namespace _impl
{

template<typename ...> struct field_traits;

template<typename T> struct field_traits<T>
{

	typedef std::integral_constant<size_t, 1> domain_type;

	typedef T value_type;

	static constexpr bool is_field = false;

	static constexpr size_t iform = 0;

	static constexpr size_t ndims = 3;

};

template<typename ...T>
struct field_traits<_Field<T ...>>
{
	static constexpr bool is_field = true;

	typedef typename _Field<T ...>::domain_type domain_type;

	typedef typename _Field<T ...>::value_type value_type;

	static constexpr size_t iform = domain_type::iform;

	static constexpr size_t ndims = domain_type::ndims;

};

}  // namespace _impl
template<typename ...T> struct domain;
template<typename ...T> using domain_t= typename domain<T...>::type;

template<typename T>
struct domain<T>
{
	typedef typename _impl::field_traits<T>::domain_type type;
};

template<typename ... T>
struct value_type<_Field<T ...>>
{
	typedef typename _impl::field_traits<_Field<T ...> >::value_type type;
};
template<typename ...T> struct iform;

template<typename ...T>
struct iform<_Field<T...>> : public std::integral_constant<size_t,
		_impl::field_traits<_Field<T...> >::iform>
{
};

template<typename ...T>
struct rank<_Field<T...>> : public std::integral_constant<size_t,
		_impl::field_traits<_Field<T...> >::ndims>
{
};
}  // namespace traits
}  // namespace simpla

#endif /* CORE_FIELD_FIELD_TRAITS_H_ */
