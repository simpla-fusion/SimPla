/**
 * @file field_traits.h
 *
 * @date 2015-6-12
 * @author salmon
 */

#ifndef COREFieldField_TRAITS_H_
#define COREFieldField_TRAITS_H_

#include <stddef.h>
#include <cstdbool>
#include <memory>
#include <type_traits>

#include "../gtl/integer_sequence.h"
#include "../gtl/type_traits.h"
#include "../manifold/manifold_traits.h"
#include "../manifold/domain_traits.h"

namespace simpla
{

template<typename ...> struct Domain;
template<typename ...> struct Field;

namespace tags
{

class sequence_container;

class associative_container;

class function;

}  // namespace tags
namespace traits
{

template<typename TM, int IFORM, typename ValueType, typename ...Policies>
struct field_type
{
	typedef Field<Domain<TM, std::integral_constant<int, IFORM>, Policies...>,
			ValueType, tags::sequence_container> type;
};

template<typename TM, int IFORM, typename ValueType, typename ...Policies>
using field_t= typename field_type<TM, IFORM, ValueType, Policies...>::type;

template<typename> struct isField : public std::integral_constant<bool, false>
{
};

template<typename ...T> struct isField<Field<T...>> : public std::integral_constant<
		bool, true>
{
};

template<typename TM, typename TV, typename ...Others>
struct reference<Field<TM, TV, Others...> >
{
	typedef Field<TM, TV, Others...> const &type;
};

template<typename ...T, int M>
struct extent<Field<T ...>, M> : public std::integral_constant<int,
		simpla::mpl::seq_get<M, extents_t<Field<T ...> >>::value>
{
};

template<typename ...T>
struct key_type<Field<T ...> >
{
	typedef size_t type;
};

template<typename ...T>
struct mesh_type<Field<T...> >
{
	typedef mesh_type_t<domain_t<Field<T...> >> type;
};

template<typename ...T>
struct iform<Field<T...> > : public iform<domain_t<Field<T...> > >::type
{
};

template<typename ...T>
struct rank<Field<T...>> : public rank<domain_t<Field<T...> > >::type
{
};

template<typename> struct field_value_type;

template<typename T>
struct field_value_type
{
	typedef typename std::conditional<
			(iform<T>::value == VERTEX || iform<T>::value == VOLUME),
			value_type_t<T>, nTuple<value_type_t<T>, 3> >::type type;
};

template<typename T> using field_value_t = typename field_value_type<T>::type;

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

template<typename T>
struct container_tag
{
	typedef tags::sequence_container type;
};

namespace _impl
{

template<typename TV, typename TAG> struct container_type_helper;

template<typename TV>
struct container_type_helper<TV, tags::sequence_container>
{
	typedef std::shared_ptr<TV> type;
};
}  // namespace _impl
template<typename> struct container_type;

template<typename T> struct container_type
{

	typedef typename _impl::container_type_helper<traits::value_type_t<T>,
			typename container_tag<T>::type>::type type;
};

template<typename T> using container_t=typename container_type<T>::type;


template<int I, typename ...U, typename TM>
Field<Domain<TM, std::integral_constant<int, I>>, U...>
make_field(TM const &mesh)
{
	return Field<Domain<TM, std::integral_constant<int, I>>, U...>(make_domain<I>(mesh));
};

}  // namespace traits
}  // namespace simpla

#endif /* COREFieldField_TRAITS_H_ */
