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

namespace simpla { namespace algebra
{
enum { VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3 };


template<typename ...> class Expression;

template<typename ...> struct BooleanExpression;

template<typename ...> struct AssignmentExpression;

namespace traits
{

template<typename> struct iform : public index_const<1> {};

template<typename> struct dof : public index_const<1> {};

template<typename> struct rank : public index_const<3> {};

template<typename TOP, typename T0, typename ... T> struct iform<Expression<TOP, T0, T...> > : public iform<T0> {};

template<typename T> struct value_type { typedef T type; };

template<typename T> using value_type_t=typename value_type<T>::type;

template<typename T> struct scalar_type { typedef Real type; };

template<typename T> using scalar_type_t=typename scalar_type<T>::type;


template<typename TOP, typename ...Others>
struct value_type<Expression<TOP, Others...> >
{
    typedef std::result_of_t<TOP(value_type_t<Others> ...)> type;
};

}//namespace traits

}}//namespace simpla{namespace algebra{
#endif //SIMPLA_ALGEBRACOMMON_H
