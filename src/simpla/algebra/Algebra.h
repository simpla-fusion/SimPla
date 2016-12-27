//
// Created by salmon on 16-12-22.
//

#ifndef SIMPLA_ALGEBRACOMMON_H
#define SIMPLA_ALGEBRACOMMON_H

#include <simpla/SIMPLA_config.h>
#include <type_traits>
#include <utility>
#include <simpla/mpl/type_traits.h>
#include <simpla/mpl/integer_sequence.h>

namespace simpla
{
enum { VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3, FIBER = 6 };

namespace algebra
{
namespace declare
{
template<typename ...> struct Expression;

template<typename, size_type ...> struct nTuple_;

template<typename, typename, size_type ...> struct Field_;
}

namespace traits
{

template<typename> struct iform : public index_const<0> {};

template<typename> struct dof : public index_const<1> {};

template<typename> struct rank : public index_const<3> {};

template<typename ...> struct extents : public index_sequence<> {};


template<typename ...T> struct iform_list { typedef index_sequence<iform<T>::value...> type; };

template<typename ...T> using iform_list_t= typename iform_list<T...>::type;

template<typename ...> struct value_type;

template<typename ...T> using value_type_t=typename value_type<T...>::type;

template<typename T> struct value_type<T> { typedef T type; };
template<typename T> struct value_type<T &> { typedef T &type; };
template<typename T> struct value_type<T const &> { typedef T const &type; };

template<typename T> struct value_type<T *> { typedef T type; };
template<typename T, size_type N> struct value_type<T[N]> { typedef T type; };
template<typename T> struct value_type<T const *> { typedef T type; };
template<typename T> struct value_type<T const[]> { typedef T type; };


template<typename T> struct scalar_type { typedef Real type; };

template<typename T> using scalar_type_t=typename scalar_type<T>::type;

template<typename ...> struct is_complex : public std::integral_constant<bool, false> {};

template<typename T> struct is_complex<std::complex<T>> : public std::integral_constant<bool, true> {};


template<typename ...> struct is_scalar : public std::integral_constant<bool, false> {};

template<typename T>
struct is_scalar<T> : public std::integral_constant<bool,
        std::is_arithmetic<std::decay_t<T>>::value || is_complex<std::decay_t<T>>::value>
{
};

template<typename ...> struct is_nTuple;

template<typename T> struct is_nTuple<T> : public std::integral_constant<bool, false> {};

template<typename ...> struct is_field;

template<typename T> struct is_field<T> : public std::integral_constant<bool, false> {};


template<typename First, typename  ...Others>
struct is_nTuple<First, Others...> : public std::integral_constant<bool,
        (is_nTuple<First>::value && (!is_field<First>::value)) || is_nTuple<Others...>::value>
{
};

template<typename First, typename  ...Others>
struct is_field<First, Others...> : public std::integral_constant<bool,
        is_field<First>::value || is_field<Others...>::value>
{
};

template<typename T> struct reference { typedef T type; };
template<typename T> using reference_t=typename reference<T>::type;
template<typename T, int N> struct reference<T[N]> { typedef T *type; };
template<typename T, int N> struct reference<const T[N]> { typedef T const *type; };

//***********************************************************************************************************************

template<typename T> struct field_value_type { typedef T type; };

template<typename T> using field_value_t=typename field_value_type<T>::type;

template<typename> struct mesh_type { typedef void type; };

template<typename TV, typename TM, size_type ...I>
struct mesh_type<declare::Field_<TV, TM, I...> > { typedef TM type; };


template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct is_field<declare::Field_<TV, TM, IFORM, DOF> > : public std::integral_constant<bool, true> {};

template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct reference<declare::Field_<TV, TM, IFORM, DOF> > { typedef declare::Field_<TV, TM, IFORM, DOF> const &type; };

template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct reference<const declare::Field_<TV, TM, IFORM, DOF> > { typedef declare::Field_<TV, TM, IFORM, DOF> const &type; };

template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct value_type<declare::Field_<TV, TM, IFORM, DOF>> { typedef TV type; };

template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct rank<declare::Field_<TV, TM, IFORM, DOF> > : public index_const<3> {};

template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct iform<declare::Field_<TV, TM, IFORM, DOF> > : public index_const<IFORM> {};

template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct dof<declare::Field_<TV, TM, IFORM, DOF> > : public index_const<DOF> {};


template<typename TV, typename TM, size_type IFORM, size_type DOF>
struct field_value_type<declare::Field_<TV, TM, IFORM, DOF>>
{
    typedef std::conditional_t<DOF == 1, TV, declare::nTuple_<TV, DOF> > cell_tuple;
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME),
            cell_tuple, declare::nTuple_<cell_tuple, 3> > type;
};

//***********************************************************************************************************************
template<typename ...> struct make_nTuple { typedef void type; };

template<typename TV, size_type ...I>
struct make_nTuple<TV, integer_sequence<size_type, I...> >
{
    typedef declare::nTuple_<TV, I...> type;
};
template<typename T, class Enable=void> struct primary_type { typedef T type; };
template<typename T> using primary_type_t=typename primary_type<T>::type;

template<typename T> struct primary_type<T, typename std::enable_if<is_nTuple<T>::value>::type>
{
    typedef typename make_nTuple<value_type_t<T>, extents<T>>::type type;
};
template<typename T> struct primary_type<T, typename std::enable_if<is_field<T>::value>::type>
{
    typedef typename declare::Field_<value_type_t<T>, typename mesh_type<T>::type, iform<T>::value, dof<T>::value> type;
};


//template<typename TOP, typename ...Others> struct is_nTuple<Expression < TOP, Others...> > : public is_nTuple<Others...> {};
//template<typename TOP, typename ...Others> struct is_field<Expression < TOP, Others...> > : public is_field<Others...>{};
//template<typename V, size_type ...I> struct is_nTuple<nTuple_ < V, I...> > : public std::integral_constant<bool, true>{};
//template<typename U, typename M, size_type...I> struct is_field<Field_ < U, M, I...> > : public std::integral_constant<bool, true> {};
//template<typename TV, size_type N0, size_type ...N> struct value_type<nTuple_<TV, N0, N...> > { typedef typename value_type<TV>::type type; };
//template<typename TV, typename TM, size_type I, size_type DOF>
//struct reference<Field_<TV, TM, I, DOF> > { typedef Field_<TV, TM, I, DOF> const &type; };
//
//template<typename TV, typename TM, size_type I, size_type DOF> struct iform<Field_<TV, TM, I, DOF> > : public index_const<I> {};
//
//template<typename TV, typename TM, size_type I, size_type DOF> struct dof<Field_<TV, TM, I, DOF> > : public index_const<DOF> {};
//
//template<typename TV, typename TM, size_type I, size_type DOF> struct rank<Field_<TV, TM, I, DOF> > : public rank<TM> { typedef TV type; };
//
//template<typename TV, typename TM, size_type I, size_type DOF> struct value_type<Field_<TV, TM, I, DOF> > { typedef TV type; };
template<typename T> T
calculate(T const &expr, ENABLE_IF(is_scalar<T>::value)) { return expr; }

template<typename T> primary_type_t<T>
calculate(T const &expr, ENABLE_IF(!is_scalar<T>::value))
{
    primary_type_t<T> res = expr;
    return std::move(res);
}
} //namespace traits



} //namespace algebra
} //namespace simpla
#endif //SIMPLA_ALGEBRACOMMON_H
