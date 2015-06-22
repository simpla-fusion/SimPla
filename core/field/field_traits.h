/**
 * @file field_traits.h
 *
 * @date 2015年6月12日
 * @author salmon
 */

#ifndef CORE_FIELD_FIELD_TRAITS_H_
#define CORE_FIELD_FIELD_TRAITS_H_

#include <stddef.h>
#include <cstdbool>
#include <memory>
#include <type_traits>

#include "../gtl/integer_sequence.h"
#include "../gtl/type_traits.h"
#include "../mesh/domain.h"
#include "../mesh/mesh_ids.h"

namespace simpla
{

template<typename ...> struct Domain;
template<typename ... >struct _Field;

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
	typedef _Field<Domain<TM, std::integral_constant<int, IFORM>, Policies...>,
			ValueType, tags::sequence_container> type;
};

template<typename TM, int IFORM, typename ValueType, typename ...Policies>
using field_t= typename field_type<TM,IFORM,ValueType,Policies...>::type;

template<typename > struct is_field: public std::integral_constant<bool, false>
{
};

template<typename ...T> struct is_field<_Field<T...>> : public std::integral_constant<
bool, true>
{
};

template<typename TM, typename TV, typename ...Others>
struct reference<_Field<TM, TV, Others...> >
{
	typedef _Field<TM, TV, Others...> const & type;
};

template<typename ...T, int M>
struct extent<_Field<T ...>, M> : public std::integral_constant<int,
		simpla::mpl::seq_get<M, extents_t<_Field<T ...> >>::value>
{
};

template<typename ...T>
struct key_type<_Field<T ...> >
{
	typedef size_t type;
};

template<typename ...T>
struct mesh_type<_Field<T...> >
{
	typedef mesh_t<domain_t<_Field<T...> > > type;
};

template<typename ...T>
struct iform<_Field<T...> > : public iform<domain_t<_Field<T...> > >::type
{
};

template<typename ...T>
struct rank<_Field<T...>> : public rank<domain_t<_Field<T...> > >::type
{
};

template<typename > struct field_value_type;

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
template<typename > struct container_type;

template<typename T> struct container_type
{

	typedef typename _impl::container_type_helper<traits::value_type_t<T>,
			typename container_tag<T>::type>::type type;
};

template<typename T> using container_t=typename container_type<T>::type;

}  // namespace traits
}  // namespace simpla

#endif /* CORE_FIELD_FIELD_TRAITS_H_ */
