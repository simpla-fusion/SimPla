/**
 * @file  calculate_fvm.h
 *
 *  Created on: 2014-9-23
 *      Author: salmon
 */

#ifndef CALCULATE_FVM_H_
#define CALCULATE_FVM_H_

#include <complex>
#include <cstddef>
#include <type_traits>


#include "../../gtl/macro.h"
#include "../../gtl/primitives.h"
#include "../../gtl/type_traits.h"
#include "../Calculus.h"
#include "../ManifoldTraits.h"


namespace simpla
{
template<typename ...> class Field;

template<typename ...> class Expression;

template<typename _Tp, _Tp...> class integer_sequence;

template<typename T, int ...> class nTuple;

}
namespace simpla { namespace manifold { namespace policy
{


#define DECLARE_FUNCTION_PREFIX inline static
#define DECLARE_FUNCTION_SUFFIX /*const*/

namespace ct=calculus::tags;

/**
 * @ingroup diff_scheme
 *
 * finite volume
 */
struct FiniteVolume
{
private:


    typedef FiniteVolume this_type;


public:
    typedef this_type calculus_policy;


private:
    ///***************************************************************************************************
    /// @name general_algebra General algebra
    /// @{
    ///***************************************************************************************************


    template<typename M> DECLARE_FUNCTION_PREFIX constexpr Real
    eval_(M const &m, Real v, typename M::id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    template<typename M> DECLARE_FUNCTION_PREFIX constexpr int
    eval_(M const &m, int v, typename M::id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    template<typename M> DECLARE_FUNCTION_PREFIX constexpr std::complex<Real>
    eval_(M const &m, std::complex<Real> v, typename M::id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    template<typename M, typename T, int ...N>
    DECLARE_FUNCTION_PREFIX constexpr nTuple<T, N...> const &
    eval_(M const &m, nTuple<T, N...> const &v, typename M::id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    template<typename M, typename ...T>
    DECLARE_FUNCTION_PREFIX traits::primary_type_t<nTuple<Expression<T...>>>
    eval_(M const &m, nTuple<Expression<T...>> const &v, typename M::id_type s) DECLARE_FUNCTION_SUFFIX
    {
        traits::primary_type_t<nTuple<Expression<T...> > > res;
        res = v;
        return std::move(res);
    }

    template<typename M, typename TV, typename ... Others>
    DECLARE_FUNCTION_PREFIX constexpr TV
    eval_(M const &m, Field<TV, M, Others...> const &f, typename M::id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return f[s];
    }

    template<typename M, typename TV, typename ... Others>
    DECLARE_FUNCTION_PREFIX constexpr TV &
    eval_(M const &m, Field<TV, M, Others...> &f, typename M::id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return f[s];
    }


    template<typename M, typename FExpr>
    DECLARE_FUNCTION_PREFIX constexpr traits::value_type_t<FExpr>
    get_v(M const &m, FExpr const &f, typename M::id_type const s) DECLARE_FUNCTION_SUFFIX
    {
        return eval_(m, f, s) * m.volume(s);
    }

    template<typename M, typename FExpr>
    DECLARE_FUNCTION_PREFIX constexpr traits::value_type_t<FExpr>
    get_d(M const &m, FExpr const &f, typename M::id_type const s) DECLARE_FUNCTION_SUFFIX
    {
        return eval_(m, f, s) * m.dual_volume(s);
    }


    template<typename M, typename ...T, int ... index>
    DECLARE_FUNCTION_PREFIX traits::primary_type_t<traits::value_type_t<Field<Expression<T...> >>>
    _invoke_helper(M const &m, Field<Expression<T...> > const &expr, typename M::id_type s,
                   index_sequence<index...>) DECLARE_FUNCTION_SUFFIX
    {
        traits::primary_type_t<traits::value_type_t<Field<Expression<T...> >>> res =
                (expr.m_op_(eval_(m, std::get<index>(expr.args), s)...));

        return std::move(res);
    }


    template<typename M, typename TOP, typename ... T>
    DECLARE_FUNCTION_PREFIX constexpr traits::primary_type_t<traits::value_type_t<Field<Expression<TOP, T...> >>>
    eval_(M const &m, Field<Expression<TOP, T...> > const &expr, typename M::id_type const &s,
          traits::iform_list_t<T...>) DECLARE_FUNCTION_SUFFIX
    {
        return _invoke_helper(m, expr, s, typename make_index_sequence<sizeof...(T)>::type());
    }

    //***************************************************************************************************
    // Exterior algebra
    //***************************************************************************************************

    //! grad<0>
    template<typename M, typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::ExteriorDerivative, T>>>
    eval_(M const &m, Field<Expression<ct::ExteriorDerivative, T> > const &f,
          typename M::id_type s, ::simpla::integer_sequence<int, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        typename M::id_type D = M::delta_index(s);


        return (get_v(m, std::get<0>(f.args), s + D) - get_v(m, std::get<0>(f.args), s - D))
               * m.inv_volume(s);


    }


    //! curl<1>
    template<typename M, typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::ExteriorDerivative, T>>>
    eval_(M const &m, Field<Expression<ct::ExteriorDerivative, T> > const &expr,
          typename M::id_type s, ::simpla::integer_sequence<int, EDGE>) DECLARE_FUNCTION_SUFFIX
    {

        typename M::id_type X = M::delta_index(M::dual(s));
        typename M::id_type Y = M::rotate(X);
        typename M::id_type Z = M::inverse_rotate(X);


        return (
                       (get_v(m, std::get<0>(expr.args), s + Y)
                        - get_v(m, std::get<0>(expr.args), s - Y))

                       - (get_v(m, std::get<0>(expr.args), s + Z)
                          - get_v(m, std::get<0>(expr.args), s - Z))

               ) * m.inv_volume(s);


    }

    //! div<2>
    template<typename M, typename T>
    constexpr DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::ExteriorDerivative, T>>>
    eval_(M const &m, Field<Expression<ct::ExteriorDerivative, T> > const &expr,
          typename M::id_type s, ::simpla::integer_sequence<int, FACE>) DECLARE_FUNCTION_SUFFIX
    {

        return (get_v(m, std::get<0>(expr.args), s + M::_DI)

                - get_v(m, std::get<0>(expr.args), s - M::_DI)

                + get_v(m, std::get<0>(expr.args), s + M::_DJ)

                - get_v(m, std::get<0>(expr.args), s - M::_DJ)

                + get_v(m, std::get<0>(expr.args), s + M::_DK)

                - get_v(m, std::get<0>(expr.args), s - M::_DK)


               ) * m.inv_volume(s);

    }


    //! div<1>
    template<typename M, typename T>
    constexpr DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::CodifferentialDerivative, T>>>
    eval_(M const &m, Field<Expression<ct::CodifferentialDerivative, T>> const &expr,
          typename M::id_type s, ::simpla::integer_sequence<int, EDGE>) DECLARE_FUNCTION_SUFFIX
    {

        return -(get_d(m, std::get<0>(expr.args), s + M::_DI)
                 - get_d(m, std::get<0>(expr.args), s - M::_DI)
                 + get_d(m, std::get<0>(expr.args), s + M::_DJ)
                 - get_d(m, std::get<0>(expr.args), s - M::_DJ)
                 + get_d(m, std::get<0>(expr.args), s + M::_DK)
                 - get_d(m, std::get<0>(expr.args), s - M::_DK)

        ) * m.inv_dual_volume(s);


    }

    //! curl<2>
    template<typename M, typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<
            Field<Expression<ct::CodifferentialDerivative, T>>>
    eval_(M const &m, Field<Expression<ct::CodifferentialDerivative, T>> const &expr,
          typename M::id_type s, ::simpla::integer_sequence<int, FACE>) DECLARE_FUNCTION_SUFFIX
    {

        typename M::id_type X = M::delta_index(s);
        typename M::id_type Y = M::rotate(X);
        typename M::id_type Z = M::inverse_rotate(X);


        return -((get_d(m, std::get<0>(expr.args), s + Y) - get_d(m, std::get<0>(expr.args), s - Y))
                 - (get_d(m, std::get<0>(expr.args), s + Z) - get_d(m, std::get<0>(expr.args), s - Z))
        ) * m.inv_dual_volume(s);
    }


    //! grad<3>
    template<typename M, typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<
            Field<Expression<ct::CodifferentialDerivative, T>>>
    eval_(M const &m, Field<Expression<ct::CodifferentialDerivative, T> > const &expr,
          typename M::id_type s, ::simpla::integer_sequence<int, VOLUME>) DECLARE_FUNCTION_SUFFIX
    {
        typename M::id_type D = M::delta_index(M::dual(s));

        return -(get_d(m, std::get<0>(expr.args), s + D) - get_d(m, std::get<0>(expr.args), s - D)) *
               m.inv_dual_volume(s);


    }

    //! *Form<IR> => Form<N-IL>


    template<typename M, typename T, int I>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::HodgeStar, T> >>
    eval_(M const &m, Field<Expression<ct::HodgeStar, T> > const &expr, typename M::id_type s,
          ::simpla::integer_sequence<int, I>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = std::get<0>(expr.args);

        int i = M::iform(s);
        typename M::id_type X = (i == VERTEX || i == VOLUME) ? M::_DI : M::delta_index(
                M::dual(s));
        typename M::id_type Y = M::rotate(X);
        typename M::id_type Z = M::inverse_rotate(X);


        return (
                       get_v(m, l, ((s - X) - Y) - Z) +
                       get_v(m, l, ((s - X) - Y) + Z) +
                       get_v(m, l, ((s - X) + Y) - Z) +
                       get_v(m, l, ((s - X) + Y) + Z) +
                       get_v(m, l, ((s + X) - Y) - Z) +
                       get_v(m, l, ((s + X) - Y) + Z) +
                       get_v(m, l, ((s + X) + Y) - Z) +
                       get_v(m, l, ((s + X) + Y) + Z)

               ) * m.inv_dual_volume(s) * 0.125;


    };


////***************************************************************************************************
//
////! map_to

    template<typename M, typename TF, int I>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<TF>
    mapto(M const &m, TF const &expr, typename M::id_type s,
          ::simpla::integer_sequence<int, I, I>) DECLARE_FUNCTION_SUFFIX
    {
        return eval_(m, expr, s);
    }


    template<typename M, typename TF>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<TF>
    mapto(M const &m, TF const &expr, typename M::id_type s,
          ::simpla::integer_sequence<int, VERTEX, EDGE>) DECLARE_FUNCTION_SUFFIX
    {

        typename M::id_type X = s & M::_DA;

        s = (s | m.FULL_OVERFLOW_FLAG) - M::_DA;

        return (eval_(m, expr, s + M::_DA - X) + eval_(m, expr, s + M::_DA + X)) *
               0.5;
    }


    template<typename M, typename TV, typename ...Others>
    DECLARE_FUNCTION_PREFIX TV
    mapto(M const &m, Field<nTuple<TV, 3>, Others...> const &expr, typename M::id_type s,
          ::simpla::integer_sequence<int, VERTEX, EDGE>) DECLARE_FUNCTION_SUFFIX
    {
        int n = M::sub_index(s);
        typename M::id_type X = s & M::_DA;
        s = (s | m.FULL_OVERFLOW_FLAG) - M::_DA;
        return (eval_(m, expr, s + M::_DA - X)[n] +
                eval_(m, expr, s + M::_DA + X)[n]) * 0.5;

    }


    template<typename M, typename TF>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<TF>
    mapto(M const &m, TF const &expr, typename M::id_type s,
          ::simpla::integer_sequence<int, VERTEX, FACE>) DECLARE_FUNCTION_SUFFIX
    {

        auto const &l = expr;
        auto X = M::delta_index(M::dual(s));
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);
        s = (s | m.FULL_OVERFLOW_FLAG) - M::_DA;

        return (
                eval_(m, l, (s + M::_DA - Y - Z)) +
                eval_(m, l, (s + M::_DA - Y + Z)) +
                eval_(m, l, (s + M::_DA + Y - Z)) +
                eval_(m, l, (s + M::_DA + Y + Z))
        );
    }


    template<typename M, typename TV, typename ...Others>
    DECLARE_FUNCTION_PREFIX TV
    mapto(M const &m, Field<nTuple<TV, 3>, Others...> const &expr, typename M::id_type s,
          ::simpla::integer_sequence<int, VERTEX, FACE>) DECLARE_FUNCTION_SUFFIX
    {

        int n = M::sub_index(s);
        auto const &l = expr;
        auto X = M::delta_index(M::dual(s));
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);
        s = (s | m.FULL_OVERFLOW_FLAG) - M::_DA;

        return (
                eval_(m, l, (s + M::_DA - Y - Z))[n] +
                eval_(m, l, (s + M::_DA - Y + Z))[n] +
                eval_(m, l, (s + M::_DA + Y - Z))[n] +
                eval_(m, l, (s + M::_DA + Y + Z))[n]
        );
    }

    template<typename M, typename TF>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<TF>
    mapto(M const &m, TF const &expr, typename M::id_type s,
          ::simpla::integer_sequence<int, VERTEX, VOLUME>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;

        auto X = m.DI(0, s);
        auto Y = m.DI(1, s);
        auto Z = m.DI(2, s);
        s = (s | m.FULL_OVERFLOW_FLAG) - M::_DA;

        return (eval_(m, l, s + M::_DA - X - Y - Z) +
                eval_(m, l, s + M::_DA - X - Y + Z) +
                eval_(m, l, s + M::_DA - X + Y - Z) +
                eval_(m, l, s + M::_DA - X + Y + Z) +
                eval_(m, l, s + M::_DA + X - Y - Z) +
                eval_(m, l, s + M::_DA + X - Y + Z) +
                eval_(m, l, s + M::_DA + X + Y - Z) +
                eval_(m, l, s + M::_DA + X + Y + Z)

        );
    }


    template<typename M, typename TF>
    DECLARE_FUNCTION_PREFIX nTuple<typename traits::value_type<TF>::type, 3>
    mapto(M const &m, TF const &expr, typename M::id_type s,
          ::simpla::integer_sequence<int, EDGE, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        typedef nTuple<typename traits::value_type<TF>::type, 3> field_value_type;

        auto const &l = expr;

        typename M::id_type DA = M::_DA;
        typename M::id_type X = M::_D;
        typename M::id_type Y = X << M::ID_DIGITS;
        typename M::id_type Z = Y << M::ID_DIGITS;

        s = (s | m.FULL_OVERFLOW_FLAG) - M::_DA;

        return field_value_type
                {
                        (eval_(m, l, s + M::_DA - X) +
                         eval_(m, l, s + M::_DA + X)) * 0.5,
                        (eval_(m, l, s + M::_DA - Y) +
                         eval_(m, l, s + M::_DA + Y)) * 0.5,
                        (eval_(m, l, s + M::_DA - Z) +
                         eval_(m, l, s + M::_DA + Z)) * 0.5

                };


    }


    template<typename M, typename TF>
    DECLARE_FUNCTION_PREFIX nTuple<typename traits::value_type<TF>::type, 3>
    mapto(M const &m, TF const &expr, typename M::id_type s,
          ::simpla::integer_sequence<int, FACE, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;

        typename M::id_type X = M::_D;
        typename M::id_type Y = X << M::ID_DIGITS;;
        typename M::id_type Z = Y << M::ID_DIGITS;;

        s = (s | m.FULL_OVERFLOW_FLAG) - M::_DA;

        return nTuple<typename ::simpla::traits::value_type<TF>::type, 3>
                {
                        (eval_(m, l, (s + M::_DA - Y - Z)) +
                         eval_(m, l, (s + M::_DA - Y + Z)) +
                         eval_(m, l, (s + M::_DA + Y - Z)) +
                         eval_(m, l, (s + M::_DA + Y + Z))),
                        (eval_(m, l, (s + M::_DA - Z - X)) +
                         eval_(m, l, (s + M::_DA - Z + X)) +
                         eval_(m, l, (s + M::_DA + Z - X)) +
                         eval_(m, l, (s + M::_DA + Z + X))),
                        (eval_(m, l, (s + M::_DA - X - Y)) +
                         eval_(m, l, (s + M::_DA - X + Y)) +
                         eval_(m, l, (s + M::_DA + X - Y)) +
                         eval_(m, l, (s + M::_DA + X + Y)))
                };


    }


    template<typename M, typename TF>
    DECLARE_FUNCTION_PREFIX typename ::simpla::traits::value_type<TF>::type
    mapto(M const &m, TF const &expr, typename M::id_type s,
          ::simpla::integer_sequence<int, VOLUME, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;

        auto X = m.DI(0, s);
        auto Y = m.DI(1, s);
        auto Z = m.DI(2, s);
        s = (s | M::FULL_OVERFLOW_FLAG) - M::_DA;

        return (
                       eval_(m, l, ((s + M::_DA - X - Y - Z))) +
                       eval_(m, l, ((s + M::_DA - X - Y + Z))) +
                       eval_(m, l, ((s + M::_DA - X + Y - Z))) +
                       eval_(m, l, ((s + M::_DA - X + Y + Z))) +
                       eval_(m, l, ((s + M::_DA + X - Y - Z))) +
                       eval_(m, l, ((s + M::_DA + X - Y + Z))) +
                       eval_(m, l, ((s + M::_DA + X + Y - Z))) +
                       eval_(m, l, ((s + M::_DA + X + Y + Z)))

               ) * 0.125;
    }


    template<typename M, typename TF>
    DECLARE_FUNCTION_PREFIX typename ::simpla::traits::value_type<TF>::type
    mapto(M const &m, TF const &expr, typename M::id_type s,
          ::simpla::integer_sequence<int, VOLUME, FACE>) DECLARE_FUNCTION_SUFFIX
    {
        auto X = M::delta_index(M::dual(s));
        s = (s | M::FULL_OVERFLOW_FLAG) - M::DA;

        return (eval_(m, expr, s - X) + eval_(m, expr, s + X)) * 0.5;
    }


    template<typename M, typename TF>
    DECLARE_FUNCTION_PREFIX typename ::simpla::traits::value_type<TF>::type
    mapto(M const &m, TF const &expr, typename M::id_type s,
          ::simpla::integer_sequence<int, VOLUME, EDGE>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;
        auto X = M::delta_index(s);
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);
        s = (s | M::FULL_OVERFLOW_FLAG) - M::_DA;

        return (
                eval_(m, l, s + M::_DA - Y - Z) +
                eval_(m, l, s + M::_DA - Y + Z) +
                eval_(m, l, s + M::_DA + Y - Z) +
                eval_(m, l, s + M::_DA + Y + Z)
        );
    }


    template<typename M, typename TF>
    DECLARE_FUNCTION_PREFIX ::simpla::nTuple<typename ::simpla::traits::value_type<TF>::type, 3>
    mapto(M const &m, TF const &expr, typename M::id_type s,
          ::simpla::integer_sequence<int, FACE, VOLUME>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;

        auto X = m.DI(0, M::dual(s));
        auto Y = m.DI(1, M::dual(s));
        auto Z = m.DI(2, M::dual(s));
        s = (s | M::FULL_OVERFLOW_FLAG) - M::_DA;

        return nTuple<traits::value_type_t<TF>, 3>
                {
                        (eval_(m, l, s + M::_DA - X) +
                         eval_(m, l, s + M::_DA + X)) * 0.5,
                        (eval_(m, l, s + M::_DA - Y) +
                         eval_(m, l, s + M::_DA + Y)) * 0.5,
                        (eval_(m, l, s + M::_DA - Z) +
                         eval_(m, l, s + M::_DA + Z)) * 0.5

                };


    }


    template<typename M, typename TF>
    DECLARE_FUNCTION_PREFIX ::simpla::nTuple<typename ::simpla::traits::value_type<TF>::type, 3>
    mapto(M const &m, TF const &expr, typename M::id_type s,
          ::simpla::integer_sequence<int, EDGE, VOLUME>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;

        auto X = m.DI(0, s);
        auto Y = m.DI(1, s);
        auto Z = m.DI(2, s);
        s = (s | M::FULL_OVERFLOW_FLAG) - M::_DA;

        return ::simpla::nTuple<typename ::simpla::traits::value_type<TF>::type, 3>
                {
                        (eval_(m, l, s + M::_DA - Y - Z) +
                         eval_(m, l, s + M::_DA - Y + Z) +
                         eval_(m, l, s + M::_DA + Y - Z) +
                         eval_(m, l, s + M::_DA + Y + Z)),
                        (eval_(m, l, s + M::_DA - Z - X) +
                         eval_(m, l, s + M::_DA - Z + X) +
                         eval_(m, l, s + M::_DA + Z - X) +
                         eval_(m, l, s + M::_DA + Z + X)),
                        (eval_(m, l, s + M::_DA - X - Y) +
                         eval_(m, l, s + M::_DA - X + Y) +
                         eval_(m, l, s + M::_DA + X - Y) +
                         eval_(m, l, s + M::_DA + X + Y))
                };


    }


    //***************************************************************************************************
    //
    //! Form<IL> ^ Form<IR> => Form<IR+IL>
    template<typename M, typename ...T, int IL, int IR>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Wedge, T...>>>
    eval_(M const &m, Field<Expression<ct::Wedge, T...>> const &expr,
          typename M::id_type s, ::simpla::integer_sequence<int, IL, IR>) DECLARE_FUNCTION_SUFFIX
    {
        return m.inner_product(mapto(m, std::get<0>(expr.args), s, ::simpla::integer_sequence<int, IL, IR + IL>()),
                               mapto(m, std::get<1>(expr.args), s, ::simpla::integer_sequence<int, IR, IR + IL>()),
                               s);
    }


    template<typename M, typename TL, typename TR>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Wedge, TL, TR>>>
    eval_(M const &m, Field<Expression<ct::Wedge, TL, TR>> const &expr,
          typename M::id_type s, ::simpla::integer_sequence<int, EDGE, EDGE>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = std::get<0>(expr.args);
        auto const &r = std::get<1>(expr.args);

        auto Y = M::delta_index(M::rotate(M::dual(s)));
        auto Z = M::delta_index(
                M::inverse_rotate(M::dual(s)));

        return ((eval_(m, l, s - Y) + eval_(m, l, s + Y))
                * (eval_(m, l, s - Z) + eval_(m, l, s + Z)) * 0.25);
    }


    template<typename M, typename TL, typename TR, int I>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Cross, TL, TR>>>
    eval_(M const &m, Field<Expression<ct::Cross, TL, TR>> const &expr,
          typename M::id_type s, ::simpla::integer_sequence<int, I, I>) DECLARE_FUNCTION_SUFFIX
    {
        return cross(eval_(m, std::get<0>(expr.args), s), eval_(m, std::get<1>(expr.args), s));
    }

    template<typename M, typename TL, typename TR, int I>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Dot, TL, TR>>>
    eval_(M const &m, Field<Expression<ct::Dot, TL, TR>> const &expr,
          typename M::id_type s, ::simpla::integer_sequence<int, I, I>) DECLARE_FUNCTION_SUFFIX
    {
        return dot(eval_(m, std::get<0>(expr.args), s), eval_(m, std::get<1>(expr.args), s));
    }


    template<typename M, typename ...T, int ...I>
    constexpr DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::MapTo, T...> >>
    eval_(M const &m, Field<Expression<ct::MapTo, T...>> const &expr, typename M::id_type s,
          ::simpla::integer_sequence<int, I...>) DECLARE_FUNCTION_SUFFIX
    {
        return mapto(m, std::get<0>(expr.args), s, ::simpla::integer_sequence<int, I...>());
    };


    template<typename M, typename TOP, typename ... T>
    DECLARE_FUNCTION_PREFIX constexpr traits::primary_type_t<traits::value_type_t<Field<Expression<TOP, T...> >>>
    eval_(M const &m, Field<Expression<TOP, T...> > const &expr, typename M::id_type const &s) DECLARE_FUNCTION_SUFFIX
    {
        return eval_(m, expr, s, traits::iform_list_t<T...>());
    }

public:

    FiniteVolume() { }

    virtual ~FiniteVolume() { }


    template<typename M, typename TV, typename ... Others>
    DECLARE_FUNCTION_PREFIX constexpr TV &
    eval(M const &m, Field<TV, M, Others...> &f, typename M::id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return f[s];
    }

    template<typename M, typename T>
    DECLARE_FUNCTION_PREFIX auto
    eval(M const &m, T const &expr, typename M::id_type const &s) DECLARE_FUNCTION_SUFFIX
    DECL_RET_TYPE((eval_(m, expr, s)))


};// struct DiffScheme<TGeo, diff_scheme::tags::finite_volume>

#undef DECLARE_FUNCTION_PREFIX
#undef DECLARE_FUNCTION_SUFFIX

} //namespace policy
} //namespace Manifold

}// namespace simpla

#endif /* FDM_H_ */
