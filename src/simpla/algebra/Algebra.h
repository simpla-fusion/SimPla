//
// Created by salmon on 16-12-22.
//

#ifndef SIMPLA_ALGEBRACOMMON_H
#define SIMPLA_ALGEBRACOMMON_H

#include <simpla/SIMPLA_config.h>
#include <type_traits>
#include <utility>
#include <simpla/mpl/type_traits.h>
#include "../mpl/integer_sequence.h"

namespace simpla
{
enum { VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3, FIBER = 6 };
}
namespace simpla { namespace algebra
{


template<typename ...> class Expression;

template<typename ...> struct BooleanExpression;

template<typename ...> struct AssignmentExpression;

template<typename, size_type ...> struct nTuple_;
template<typename, typename, size_type ...I> struct Field_;

namespace traits
{

template<typename> struct iform : public index_const<1> {};

template<typename> struct dof : public index_const<1> {};

template<typename> struct rank : public index_const<3> {};

template<typename ...T> struct iform_list { typedef index_sequence<iform<T>::value...> type; };

template<typename ...T> using iform_list_t= typename iform_list<T...>::type;

template<typename TOP, typename T0, typename ... T> struct iform<Expression<TOP, T0, T...> > : public iform<T0> {};

template<typename T> struct value_type { typedef T type; };

template<typename T> using value_type_t=typename value_type<T>::type;

template<typename T> struct scalar_type { typedef Real type; };

template<typename T> using scalar_type_t=typename scalar_type<T>::type;


template<typename TOP, typename ...Others>
struct value_type<Expression<TOP, Others...> > { typedef std::result_of_t<TOP(value_type_t<Others> ...)> type; };

template<typename ...>
struct is_nTuple : public std::integral_constant<bool, false> {};

template<typename ...>
struct is_field : public std::integral_constant<bool, false> {};

template<typename V, size_type ...I>
struct is_nTuple<nTuple_<V, I...> > : public std::integral_constant<bool, true> {};

template<typename U, typename M, size_type...I>
struct is_field<Field_<U, M, I...> > : public std::integral_constant<bool, true> {};

template<typename First, typename  ...Others>
struct is_nTuple<First, Others...>
        : public std::integral_constant<bool,
                (is_nTuple<First>::value || is_nTuple<Others...>::value) && !is_field<First, Others...>::value

        >
{
};

template<typename First, typename  ...Others>
struct is_field<First, Others...>
        : public std::integral_constant<bool, is_field<First>::value || is_field<Others...>::value>
{
};

template<typename TOP, typename ...Others>
struct is_nTuple<Expression<TOP, Others...> > : public is_nTuple<Others...>
{
};

template<typename TOP, typename ...Others>
struct is_field<Expression<TOP, Others...> > : public is_field<Others...>
{
};

template<typename T>
struct field_value_type
{
    typedef std::conditional_t<
            iform<T>::value == VERTEX || iform<T>::value == VOLUME, value_type_t<T>,
            nTuple_<value_type_t<T>, rank<T>::value> > type;
};

template<typename T> using field_value_t=typename field_value_type<T>::type;


template<typename TV, size_type N0, size_type ...N>
struct value_type<nTuple_<TV, N0, N...> > { typedef typename value_type<TV>::type type; };

template<typename TV> struct reference { typedef TV type; };

template<typename TV, typename TM, size_type I, size_type DOF>
struct reference<Field_<TV, TM, I, DOF> > { typedef Field_<TV, TM, I, DOF> const &type; };

template<typename TV, typename TM, size_type I, size_type DOF>
struct iform<Field_<TV, TM, I, DOF> > : public index_const<I> {};

template<typename TV, typename TM, size_type I, size_type DOF>
struct dof<Field_<TV, TM, I, DOF> > : public index_const<DOF> {};

template<typename TV, typename TM, size_type I, size_type DOF>
struct rank<Field_<TV, TM, I, DOF> > : public rank<TM> { typedef TV type; };

template<typename TV, typename TM, size_type I, size_type DOF>
struct value_type<Field_<TV, TM, I, DOF> > { typedef TV type; };


}//namespace traits

}}//namespace simpla{namespace algebra{
#endif //SIMPLA_ALGEBRACOMMON_H
