//
// Created by salmon on 16-5-28.
//

#ifndef SIMPLA_NTUPLEEXPR_H
#define SIMPLA_NTUPLEEXPR_H

#include "Algebra.h"
#include "Expression.h"
#include "Arithmetic.h"

namespace simpla
{
namespace algebra
{


}//namespace algebra{


#define _SP_DEFINE_nTuple_EXPR_BINARY_RIGHT_OPERATOR(_OP_, _NAME_)                                                  \
    template<typename T1,size_type ...N1,typename  T2> \
    constexpr  algebra::Expression<algebra::tags::_NAME_,nTuple<T1,N1...>,T2>  \
    operator _OP_(nTuple<T1,N1...> const & l,T2 const &r)  \
    {return ( algebra::Expression<algebra::tags::_NAME_,nTuple<T1,N1...>,T2> (l,r)) ;}                 \


#define _SP_DEFINE_nTuple_EXPR_BINARY_OPERATOR(_OP_, _NAME_)                                                  \
    template<typename T1,size_type ...N1,typename  T2> \
    constexpr  algebra::Expression<algebra::tags::_NAME_,nTuple<T1,N1...>,T2>  \
    operator _OP_(nTuple<T1, N1...> const & l,T2 const&r)  \
    {return ( algebra::Expression<algebra::tags::_NAME_,nTuple<T1,N1...>,T2> (l,r));}                    \
    \
    template< typename T1,typename T2 ,size_type ...N2> \
    constexpr algebra::Expression< algebra::tags::_NAME_,T1,nTuple< T2,N2...>>  \
    operator _OP_(T1 const & l, nTuple< T2,N2...>const &r)                    \
    {return ( algebra::Expression< algebra::tags::_NAME_,T1,nTuple< T2,N2...>> (l,r))  ;}                \
    \
    template< typename T1,size_type ... N1,typename T2 ,size_type ...N2>  \
    constexpr  algebra::Expression< algebra::tags::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>> \
    operator _OP_(nTuple< T1,N1...> const & l,nTuple< T2,N2...>  const &r)                    \
    {return ( algebra::Expression< algebra::tags::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>> (l,r));}                    \


#define _SP_DEFINE_nTuple_EXPR_UNARY_OPERATOR(_OP_, _NAME_)                           \
        template<typename T,size_type ...N> \
        constexpr  algebra::Expression<algebra::tags::_NAME_,nTuple<T,N...> >  \
        operator _OP_(nTuple<T,N...> const &l)  \
        {return ( algebra::Expression<algebra::tags::_NAME_,nTuple<T,N...> > (l)) ;}    \


#define _SP_DEFINE_nTuple_EXPR_BINARY_FUNCTION(_NAME_)                                                  \
            template<typename T1,size_type ...N1,typename  T2> \
            constexpr  algebra::BooleanExpression<algebra::tags::_NAME_,nTuple<T1,N1...>,T2>  \
            _NAME_(nTuple<T1,N1...> const & l,T2 const &r)  \
            {return ( algebra::BooleanExpression<algebra::tags::_NAME_,nTuple<T1,N1...>,T2> (l,r));}       \
            \
            template< typename T1,typename T2,size_type ...N2> \
            constexpr   algebra::Expression< algebra::tags::_NAME_,T1,nTuple< T2,N2...>> \
            _NAME_(T1 const & l, nTuple< T2,N2...>const &r)                    \
            {return ( algebra::Expression< algebra::tags::_NAME_,T1,nTuple< T2,N2...>> (l,r)) ;}       \
            \
            template< typename T1,size_type ... N1,typename T2,size_type  ...N2> \
            constexpr    algebra::Expression< algebra::tags::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>> \
            _NAME_(nTuple< T1,N1...> const & l,nTuple< T2,N2...>  const &r)                    \
            {return ( algebra::Expression< algebra::tags::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>> (l,r))  ;}   \


#define _SP_DEFINE_nTuple_EXPR_UNARY_FUNCTION(_NAME_)                           \
        template<typename T,size_type ...N> \
        constexpr   algebra::Expression<algebra::tags::_NAME_,nTuple<T,N...>>  \
        _NAME_(nTuple<T,N ...> const &r)  \
        {return ( algebra::Expression<algebra::tags::_NAME_,nTuple<T,N...>> (r));}     \


DEFINE_EXPRESSION_TEMPLATE_BASIC_ALGEBRA2(nTuple)


}


//namespace simpla
/** @}*/
//----------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------
//
//namespace traits
//{
//
////template<typename TV, size_type I0, size_type ...I>
////auto get_value(nTuple<TV, I0, I...> const &v, int i) -> decltype(v[i]) { return (v[i]); }
//
//template<typename TV, size_type I0, size_type ...I, typename TR>
//void assign(std::nullptr_t, nTuple<TV, I0, I...> &lhs, TR const &rhs)
//{
//    for (int i = 0; i < I0; ++i) { assign(nullptr, lhs[i], get_value(rhs, i)); }
//};
//
//template<typename TOP, typename TV, size_type I0, size_type ...I, typename TR>
//void assign(TOP const *op, nTuple<TV, I0, I...> &lhs, TR const &rhs)
//{
//    for (int i = 0; i < I0; ++i) { assign(op, lhs[i], get_value(rhs, i)); }
//};
//
//template<typename TOP, typename TV, size_type I0, size_type ...I, typename TR>
//void evaluate(TOP const *op, nTuple<TV, I0, I...> const &lhs, TR const &rhs)
//{
//    for (int i = 0; i < I0; ++i) { evaluate(op, lhs[i], get_value(rhs, i)); }
//};
//
//template<typename TOP, typename U, size_type...I, typename ...Others>
//struct result_type<TOP, nTuple<U, I...>, Others...>
//{
//    typedef decltype(evaluate(std::declval<TOP const *>(), std::declval<nTuple<U, I...> >(),
//                              std::declval<Others>()...)) type;
//};
//}

//
//template<typename T>
//inline auto determinant(nTuple<T, 3> const &m) DECL_RET_TYPE(((m[0] * m[1] * m[2])))
//
//template<typename T>
//inline auto determinant(nTuple<T, 4> const &m) DECL_RET_TYPE((m[0] * m[1] * m[2] * m[3]))
//
//template<typename T>
//inline auto determinant(Matrix<T, 3, 3> const &m)
//DECL_RET_TYPE((
//                      m[0][0] * m[1][1] * m[2][2] -
//                      m[0][2] * m[1][1] * m[2][0] +
//                      m[0][1] * m[1][2] * m[2][0] -
//                      m[0][1] * m[1][0] * m[2][2] +
//                      m[1][0] * m[2][1] * m[0][2] -
//                      m[1][2] * m[2][1] * m[0][0]))
//
//template<typename T>
//inline auto determinant(Matrix<T, 4, 4> const &m)
//DECL_RET_TYPE((
//                      m[0][3] * m[1][2] * m[2][1] * m[3][0] - m[0][2] * m[1][3] * m[2][1] * m[3][0]//
//                      - m[0][3] * m[1][1] * m[2][2] * m[3][0] + m[0][1] * m[1][3]//
//                                                                * m[2][2] * m[3][0] +
//                      m[0][2] * m[1][1] * m[2][3] * m[3][0] -
//                      m[0][1]//
//                      * m[1][2] * m[2][3] * m[3][0] - m[0][3] * m[1][2] * m[2][0] * m[3][1]//
//                      + m[0][2] * m[1][3] * m[2][0] * m[3][1] + m[0][3] * m[1][0] * m[2][2]//
//                                                                * m[3][1] - m[0][0] * m[1][3] * m[2][2] * m[3][1] -
//                      m[0][2] * m[1][0]//
//                      * m[2][3] * m[3][1] + m[0][0] * m[1][2] * m[2][3] * m[3][1] + m[0][3]//
//                                                                                    * m[1][1] * m[2][0] * m[3][2] -
//                      m[0][1] * m[1][3] * m[2][0] * m[3][2]//
//                      - m[0][3] * m[1][0] * m[2][1] * m[3][2] + m[0][0] * m[1][3] * m[2][1]//
//                                                                * m[3][2] + m[0][1] * m[1][0] * m[2][3] * m[3][2] -
//                      m[0][0] * m[1][1]//
//                      * m[2][3] * m[3][2] - m[0][2] * m[1][1] * m[2][0] * m[3][3] + m[0][1]//
//                                                                                    * m[1][2] * m[2][0] * m[3][3] +
//                      m[0][2] * m[1][0] * m[2][1] * m[3][3]//
//                      - m[0][0] * m[1][2] * m[2][1] * m[3][3] - m[0][1] * m[1][0] * m[2][2]//
//                                                                * m[3][3] + m[0][0] * m[1][1] * m[2][2] * m[3][3]//
//              ))
//
//template<typename T1, size_type ... N1, typename T2, size_type ... N2>
//inline
//nTuple<traits::result_type_t<tags::multiplies(T1, T2)>, 3>
//cross(nTuple<T1, N1...> const &l, nTuple<T2, N2...> const &r)
//{
//    return
//            nTuple<decltype(traits::get_value(l, 0) * traits::get_value(r, 0)), 3> {
//                    traits::get_value(l, 1) * traits::get_value(r, 2)
//                    - traits::get_value(l, 2) * traits::get_value(r, 1),
//                    traits::get_value(l, 2) * traits::get_value(r, 0)
//                    - traits::get_value(l, 0) * traits::get_value(r, 2),
//                    traits::get_value(l, 0) * traits::get_value(r, 1)
//                    - traits::get_value(l, 1) * traits::get_value(r, 0)
//            };
//}
//----------------------------------------------------------------------------------------------------------------------
//template<typename T, size_type ...N>
//auto mod(nTuple<T, N...> const & l)
//DECL_RET_TYPE((std::sqrt(std::abs(inner_product(l,l)))))
//
//#define _SP_DEFINE_nTuple_EXPR_BINARY_BOOLEAN_OPERATOR(_OP_, _NAME_)                                                  \
//    template<typename T1,size_type ...N1,typename  T2> \
//    constexpr  BooleanExpression<tags::_NAME_,nTuple<T1,N1...>,T2>  \
//    operator _OP_(nTuple<T1, N1...> const & l,T2 const&r)  \
//    {return ( BooleanExpression<tags::_NAME_,nTuple<T1,N1...>,T2> (l,r));}                    \
//    \
//    template< typename T1,typename T2 ,size_type ...N2> \
//    constexpr  BooleanExpression< tags::_NAME_,T1,nTuple< T2,N2...>>  \
//    operator _OP_(T1 const & l, nTuple< T2,N2...>const &r)                    \
//    {return ( BooleanExpression< tags::_NAME_,T1,nTuple< T2,N2...>> (l,r))  ;}                \
//    \
//    template< typename T1,size_type ... N1,typename T2 ,size_type ...N2>  \
//    constexpr BooleanExpression< tags::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>> \
//    operator _OP_(nTuple< T1,N1...> const & l,nTuple< T2,N2...>  const &r)                    \
//    {return ( BooleanExpression< tags::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>> (l,r));}                    \
//
//
//#define _SP_DEFINE_nTuple_EXPR_UNARY_BOOLEAN_OPERATOR(_OP_, _NAME_)                           \
//        template<typename T,size_type ...N> \
//        constexpr  BooleanExpression<tags::_NAME_,nTuple<T,N...> >  \
//        operator _OP_(nTuple<T,N...> const &l)  \
//        {return ( BooleanExpression<tags::_NAME_,nTuple<T,N...> > (l)) ;}    \
//
//
//DEFINE_EXPRESSOPM_TEMPLATE_BOOLEAN_ALGEBRA2(nTuple)



//----------------------------------------------------------------------------------------------------------------------

//
//
//template<typename TOP, typename T>
//T
//reduce(TOP const &op, T const &v) { return v; }
//
//template<typename TOP, typename T, size_type N0, size_type ...N>
//traits::value_type_t<nTuple<T, N0, N...>>
//reduce(TOP const &op, nTuple<T, N0, N...> const &v)
//{
//    static constexpr size_type n = N0;
//
//    traits::value_type_t<nTuple<T, N0, N...> > res = reduce(op, traits::get_value(v, 0));
//    if (n > 1)
//    {
//        for (size_type s = 1; s < n; ++s) { res = op(res, reduce(op, traits::get_value(v, s))); }
//    }
//    return
//            res;
//}
//
//template<typename TOP, typename ...T>
//traits::value_type_t<nTuple<Expression<T...> > >
//reduce(TOP const &op, nTuple<Expression<T...>> const &v)
//{
//    traits::primary_type_t<nTuple<Expression<T...> > >
//            res = v;
//
//    return reduce(op, res);
//}
//
//template<typename TOP, typename ...Args>
//void for_each(TOP const &op, index_sequence<>, Args &&... args)
//{
//    op(std::forward<Args>(args) ...);
//}
//
//template<size_type N, size_type ...M, typename TOP, typename ...Args>
//void for_each(TOP const &op, index_sequence<N, M...>,
//              Args &&... args)
//{
//    for (size_type s = 0; s < N; ++s)
//    {
//        for_each(op, index_sequence<M...>(), traits::get_value(std::forward<Args>(args), s)...);
//    }
//
//}
//
//template<typename TR, typename T, size_type ... N>
//auto inner_product(nTuple<T, N...> const &l, TR const &r)
//DECL_RET_TYPE((reduce(tags::plus(), l * r)))
//
//inline constexpr double dot(double const &l, double const &r) { return r * l; }
//
//inline constexpr float dot(float const &l, float const &r) { return r * l; }
//
//template<typename TR, typename T, size_type ... N>
//auto dot(nTuple<T, N...> const &l, TR const &r) DECL_RET_TYPE((inner_product(l, r)))
//
//template<typename T, size_type ... N>
//auto normal(nTuple<T, N...> const &l) DECL_RET_TYPE((std::sqrt(inner_product(l, l))))
//
//template<typename T>
//auto sp_abs(T const &v) DECL_RET_TYPE((std::abs(v)))
//
//template<typename T, size_type ...N>
//auto sp_abs(nTuple<T, N...> const &m) DECL_RET_TYPE((std::sqrt(std::abs(inner_product(m, m)))))
//
//template<typename ... T>
//auto sp_abs(nTuple<Expression<T...>> const &m) DECL_RET_TYPE((std::sqrt(std::abs(inner_product(m, m)))))
//
//template<typename T>
//auto mod(T const &v) DECL_RET_TYPE((sp_abs(v)))
//
//template<typename T, size_type ...N>
//auto abs(nTuple<T, N...> const &v) DECL_RET_TYPE((sp_abs(v)))
//
//template<typename T, size_type ...N>
//inline auto NProduct(nTuple<T, N...> const &v) DECL_RET_TYPE((reduce(tags::multiplies(), v)))
//
//template<typename T, size_type ...N>
//inline auto NSum(nTuple<T, N...> const &v) DECL_RET_TYPE((reduce(tags::plus(), v)))



#endif //SIMPLA_NTUPLEEXPR_H
