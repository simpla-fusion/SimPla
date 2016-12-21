/**
 * @file field_traits.h
 *
 * @date 2015-6-12
 * @author salmon
 */

#ifndef FIELD_TRAITS_H_
#define FIELD_TRAITS_H_

#include <stddef.h>
#include <cstdbool>
#include <memory>
#include <type_traits>
#include "simpla/toolbox/type_traits.h"
#include "simpla/toolbox/mpl.h"
#include "simpla/toolbox/integer_sequence.h"
#include "simpla/manifold/ManifoldTraits.h"
#include "simpla/mesh/EntityIdRange.h"
#include "FieldExpression.h"

namespace simpla
{

template<typename ...> struct Field;

namespace traits
{

template<typename> struct mesh_type;


//template<typename ValueType, typename TM, int IFORM, typename ...Policies>
//struct field_type
//{
//    typedef field<ValueType, TM, std::integral_constant<int, IFORM>, Policies...> type;
//};

template<typename> struct is_field : public std::integral_constant<bool, false> {};

template<typename ...T> struct is_field<Field<T...>> : public std::integral_constant<bool, true> {};

template<typename T> using is_field_t=  std::enable_if_t<is_field<T>::value>;
template<typename T> using is_primary_field_t=   std::enable_if_t<is_field<T>::value && !(is_expression<T>::value)>;
template<typename T> using is_expression_field_t=  std::enable_if_t<is_field<T>::value && (is_expression<T>::value)>;

template<typename TM, typename TV, typename ...Others>
struct reference<Field<TM, TV, Others...> > { typedef Field<TM, TV, Others...> const &type; };

template<typename ...T, int M>
struct extent<Field<T ...>, M> : public index_const<traits::seq_get<M, extents<Field<T ...> > >::value> {};

template<typename ...T>
struct key_type<Field<T ...> > { typedef size_t type; };


template<typename> struct field_value_type;

template<typename T>
struct field_value_type
{
    typedef typename std::conditional<
            (iform<T>::value == mesh::VERTEX || iform<T>::value == mesh::VOLUME),
            value_type_t<T>, nTuple<value_type_t<T>, 3> >::type type;
};

template<typename T> using field_value_t = typename field_value_type<T>::type;


}  // namespace traits
}  // namespace simpla

#endif /* FIELD_TRAITS_H_ */
