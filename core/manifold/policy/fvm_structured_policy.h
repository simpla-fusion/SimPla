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

#include "../calculus.h"

#include "../../gtl/macro.h"
#include "../../gtl/primitives.h"
#include "../../gtl/type_traits.h"
#include "../../manifold/manifold_traits.h"


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
template<typename TGeo>
struct FiniteVolume
{
private:

    typedef TGeo mesh_type;

    typedef FiniteVolume<mesh_type> this_type;

    typedef typename mesh_type::id_type id_type;

    static const int ndims = mesh_type::ndims;

    static constexpr id_type _DA = mesh_type::_DA;
    static constexpr id_type _DI = mesh_type::_DI;
    static constexpr id_type _DJ = mesh_type::_DJ;
    static constexpr id_type _DK = mesh_type::_DK;

public:
    typedef this_type calculus_policy;


private:
    ///***************************************************************************************************
    /// @name general_algebra General algebra
    /// @{
    ///***************************************************************************************************



    DECLARE_FUNCTION_PREFIX constexpr Real
    eval_(mesh_type const &m, Real v, id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    DECLARE_FUNCTION_PREFIX constexpr int
    eval_(mesh_type const &m, int v, id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    DECLARE_FUNCTION_PREFIX constexpr std::complex<Real>
    eval_(mesh_type const &m, std::complex<Real> v, id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    template<typename T, int ...N>
    DECLARE_FUNCTION_PREFIX constexpr nTuple<T, N...> const &
    eval_(mesh_type const &m, nTuple<T, N...> const &v, id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    template<typename ...T>
    DECLARE_FUNCTION_PREFIX traits::primary_type_t<nTuple<Expression<T...>>>
    eval_(mesh_type const &m, nTuple<Expression<T...>> const &v, id_type s) DECLARE_FUNCTION_SUFFIX
    {
        traits::primary_type_t<nTuple<Expression<T...> > > res;
        res = v;
        return std::move(res);
    }

    template<typename TV, typename mesh_type, typename ... Others>
    DECLARE_FUNCTION_PREFIX constexpr TV
    eval_(mesh_type const &m, Field<TV, mesh_type, Others...> const &f, id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return f[s];
    }

    template<typename TV, typename mesh_type, typename ... Others>
    DECLARE_FUNCTION_PREFIX constexpr TV &
    eval_(mesh_type const &m, Field<TV, mesh_type, Others...> &f, id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return f[s];
    }


    template<typename FExpr>
    DECLARE_FUNCTION_PREFIX constexpr traits::value_type_t<FExpr>
    get_v(mesh_type const &m, FExpr const &f, id_type const s) DECLARE_FUNCTION_SUFFIX
    {
        return eval_(m, f, s) * m.volume(s);
    }

    template<typename FExpr>
    DECLARE_FUNCTION_PREFIX constexpr traits::value_type_t<FExpr>
    get_d(mesh_type const &m, FExpr const &f, id_type const s) DECLARE_FUNCTION_SUFFIX
    {
        return eval_(m, f, s) * m.dual_volume(s);
    }


    template<typename ...T, int ... index>
    DECLARE_FUNCTION_PREFIX traits::primary_type_t<traits::value_type_t<Field<Expression<T...> >>>
    _invoke_helper(mesh_type const &m, Field<Expression<T...> > const &expr, id_type s,
                   index_sequence<index...>) DECLARE_FUNCTION_SUFFIX
    {
        traits::primary_type_t<traits::value_type_t<Field<Expression<T...> >>> res =
                (expr.m_op_(eval_(m, std::get<index>(expr.args), s)...));

        return std::move(res);
    }


    template<typename TOP, typename ... T>
    DECLARE_FUNCTION_PREFIX constexpr traits::primary_type_t<traits::value_type_t<Field<Expression<TOP, T...> >>>
    eval_(mesh_type const &m, Field<Expression<TOP, T...> > const &expr, id_type const &s,
          traits::iform_list_t<T...>) DECLARE_FUNCTION_SUFFIX
    {
        return _invoke_helper(m, expr, s, typename make_index_sequence<sizeof...(T)>::type());
    }

    //***************************************************************************************************
    // Exterior algebra
    //***************************************************************************************************

    //! grad<0>
    template<typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::ExteriorDerivative, T>>>
    eval_(mesh_type const &m, Field<Expression<ct::ExteriorDerivative, T> > const &f,
          id_type s, ::simpla::integer_sequence<int, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        id_type D = mesh_type::delta_index(s);


        return (get_v(m, std::get<0>(f.args), s + D) - get_v(m, std::get<0>(f.args), s - D))
               * m.inv_volume(s);


    }


    //! curl<1>
    template<typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::ExteriorDerivative, T>>>
    eval_(mesh_type const &m, Field<Expression<ct::ExteriorDerivative, T> > const &expr,
          id_type s, ::simpla::integer_sequence<int, EDGE>) DECLARE_FUNCTION_SUFFIX
    {

        id_type X = mesh_type::delta_index(mesh_type::dual(s));
        id_type Y = mesh_type::rotate(X);
        id_type Z = mesh_type::inverse_rotate(X);


        return (
                       (get_v(m, std::get<0>(expr.args), s + Y)
                        - get_v(m, std::get<0>(expr.args), s - Y))

                       - (get_v(m, std::get<0>(expr.args), s + Z)
                          - get_v(m, std::get<0>(expr.args), s - Z))

               ) * m.inv_volume(s);


    }

    //! div<2>
    template<typename T>
    constexpr DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::ExteriorDerivative, T>>>
    eval_(mesh_type const &m, Field<Expression<ct::ExteriorDerivative, T> > const &expr,
          id_type s, ::simpla::integer_sequence<int, FACE>) DECLARE_FUNCTION_SUFFIX
    {

        return (get_v(m, std::get<0>(expr.args), s + _DI)

                - get_v(m, std::get<0>(expr.args), s - _DI)

                + get_v(m, std::get<0>(expr.args), s + _DJ)

                - get_v(m, std::get<0>(expr.args), s - _DJ)

                + get_v(m, std::get<0>(expr.args), s + _DK)

                - get_v(m, std::get<0>(expr.args), s - _DK)


               ) * m.inv_volume(s);

    }


    //! div<1>
    template<typename T>
    constexpr DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::CodifferentialDerivative, T>>>
    eval_(mesh_type const &m, Field<Expression<ct::CodifferentialDerivative, T>> const &expr,
          id_type s, ::simpla::integer_sequence<int, EDGE>) DECLARE_FUNCTION_SUFFIX
    {

        return -(get_d(m, std::get<0>(expr.args), s + _DI)
                 - get_d(m, std::get<0>(expr.args), s - _DI)
                 + get_d(m, std::get<0>(expr.args), s + _DJ)
                 - get_d(m, std::get<0>(expr.args), s - _DJ)
                 + get_d(m, std::get<0>(expr.args), s + _DK)
                 - get_d(m, std::get<0>(expr.args), s - _DK)

        ) * m.inv_dual_volume(s);


    }

    //! curl<2>
    template<typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<
            Field<Expression<ct::CodifferentialDerivative, T>>>
    eval_(mesh_type const &m, Field<Expression<ct::CodifferentialDerivative, T>> const &expr,
          id_type s, ::simpla::integer_sequence<int, FACE>) DECLARE_FUNCTION_SUFFIX
    {

        id_type X = mesh_type::delta_index(s);
        id_type Y = mesh_type::rotate(X);
        id_type Z = mesh_type::inverse_rotate(X);


        return -((get_d(m, std::get<0>(expr.args), s + Y) - get_d(m, std::get<0>(expr.args), s - Y))
                 - (get_d(m, std::get<0>(expr.args), s + Z) - get_d(m, std::get<0>(expr.args), s - Z))
        ) * m.inv_dual_volume(s);
    }


    //! grad<3>
    template<typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<
            Field<Expression<ct::CodifferentialDerivative, T>>>
    eval_(mesh_type const &m, Field<Expression<ct::CodifferentialDerivative, T> > const &expr,
          id_type s, ::simpla::integer_sequence<int, VOLUME>) DECLARE_FUNCTION_SUFFIX
    {
        id_type D = mesh_type::delta_index(mesh_type::dual(s));

        return -(get_d(m, std::get<0>(expr.args), s + D) - get_d(m, std::get<0>(expr.args), s - D)) *
               m.inv_dual_volume(s);


    }

    //! *Form<IR> => Form<N-IL>


    template<typename T, int I>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::HodgeStar, T> >>
    eval_(mesh_type const &m, Field<Expression<ct::HodgeStar, T> > const &expr, id_type s,
          ::simpla::integer_sequence<int, I>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = std::get<0>(expr.args);

        int i = mesh_type::iform(s);
        id_type X = (i == VERTEX || i == VOLUME) ? _DI : mesh_type::delta_index(
                mesh_type::dual(s));
        id_type Y = mesh_type::rotate(X);
        id_type Z = mesh_type::inverse_rotate(X);


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

    template<typename TF, int I>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, id_type s,
          ::simpla::integer_sequence<int, I, I>) DECLARE_FUNCTION_SUFFIX
    {
        return eval_(m, expr, s);
    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, id_type s,
          ::simpla::integer_sequence<int, VERTEX, EDGE>) DECLARE_FUNCTION_SUFFIX
    {

        id_type X = s & _DA;

        s = (s | m.FULL_OVERFLOW_FLAG) - _DA;

        return (eval_(m, expr, s + _DA - X) + eval_(m, expr, s + _DA + X)) * 0.5;
    }


    template<typename TV, typename ...Others>
    DECLARE_FUNCTION_PREFIX TV
    mapto(mesh_type const &m, Field<nTuple<TV, 3>, Others...> const &expr, id_type s,
          ::simpla::integer_sequence<int, VERTEX, EDGE>) DECLARE_FUNCTION_SUFFIX
    {
        int n = mesh_type::sub_index(s);
        id_type X = s & _DA;
        s = (s | m.FULL_OVERFLOW_FLAG) - _DA;
        return (eval_(m, expr, s + _DA - X)[n] + eval_(m, expr, s + _DA + X)[n]) * 0.5;

    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, id_type s,
          ::simpla::integer_sequence<int, VERTEX, FACE>) DECLARE_FUNCTION_SUFFIX
    {

        auto const &l = expr;
        auto X = mesh_type::delta_index(mesh_type::dual(s));
        auto Y = mesh_type::rotate(X);
        auto Z = mesh_type::inverse_rotate(X);
        s = (s | m.FULL_OVERFLOW_FLAG) - _DA;

        return (
                eval_(m, l, (s + _DA - Y - Z)) +
                eval_(m, l, (s + _DA - Y + Z)) +
                eval_(m, l, (s + _DA + Y - Z)) +
                eval_(m, l, (s + _DA + Y + Z))
        );
    }


    template<typename TV, typename ...Others>
    DECLARE_FUNCTION_PREFIX TV
    mapto(mesh_type const &m, Field<nTuple<TV, 3>, Others...> const &expr, id_type s,
          ::simpla::integer_sequence<int, VERTEX, FACE>) DECLARE_FUNCTION_SUFFIX
    {

        int n = mesh_type::sub_index(s);
        auto const &l = expr;
        auto X = mesh_type::delta_index(mesh_type::dual(s));
        auto Y = mesh_type::rotate(X);
        auto Z = mesh_type::inverse_rotate(X);
        s = (s | m.FULL_OVERFLOW_FLAG) - _DA;

        return (
                eval_(m, l, (s + _DA - Y - Z))[n] +
                eval_(m, l, (s + _DA - Y + Z))[n] +
                eval_(m, l, (s + _DA + Y - Z))[n] +
                eval_(m, l, (s + _DA + Y + Z))[n]
        );
    }

    template<typename TF>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, id_type s,
          ::simpla::integer_sequence<int, VERTEX, VOLUME>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;

        auto X = m.DI(0, s);
        auto Y = m.DI(1, s);
        auto Z = m.DI(2, s);
        s = (s | m.FULL_OVERFLOW_FLAG) - _DA;

        return (eval_(m, l, s + _DA - X - Y - Z) +
                eval_(m, l, s + _DA - X - Y + Z) +
                eval_(m, l, s + _DA - X + Y - Z) +
                eval_(m, l, s + _DA - X + Y + Z) +
                eval_(m, l, s + _DA + X - Y - Z) +
                eval_(m, l, s + _DA + X - Y + Z) +
                eval_(m, l, s + _DA + X + Y - Z) +
                eval_(m, l, s + _DA + X + Y + Z)

        );
    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX nTuple<typename traits::value_type<TF>::type, 3>
    mapto(mesh_type const &m, TF const &expr, id_type s,
          ::simpla::integer_sequence<int, EDGE, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        typedef nTuple<typename traits::value_type<TF>::type, 3> field_value_type;

        auto const &l = expr;

        id_type DA = _DA;
        id_type X = m._D;
        id_type Y = X << m.ID_DIGITS;
        id_type Z = Y << m.ID_DIGITS;

        s = (s | m.FULL_OVERFLOW_FLAG) - _DA;

        return field_value_type
                {
                        (eval_(m, l, s + _DA - X) + eval_(m, l, s + _DA + X)) * 0.5,
                        (eval_(m, l, s + _DA - Y) + eval_(m, l, s + _DA + Y)) * 0.5,
                        (eval_(m, l, s + _DA - Z) + eval_(m, l, s + _DA + Z)) * 0.5

                };


    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX nTuple<typename traits::value_type<TF>::type, 3>
    mapto(mesh_type const &m, TF const &expr, id_type s,
          ::simpla::integer_sequence<int, FACE, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;

        id_type X = m._D;
        id_type Y = X << m.ID_DIGITS;
        id_type Z = Y << m.ID_DIGITS;

        s = (s | m.FULL_OVERFLOW_FLAG) - _DA;

        return nTuple<typename ::simpla::traits::value_type<TF>::type, 3>
                {
                        (eval_(m, l, (s + _DA - Y - Z)) +
                         eval_(m, l, (s + _DA - Y + Z)) +
                         eval_(m, l, (s + _DA + Y - Z)) +
                         eval_(m, l, (s + _DA + Y + Z))),
                        (eval_(m, l, (s + _DA - Z - X)) +
                         eval_(m, l, (s + _DA - Z + X)) +
                         eval_(m, l, (s + _DA + Z - X)) +
                         eval_(m, l, (s + _DA + Z + X))),
                        (eval_(m, l, (s + _DA - X - Y)) +
                         eval_(m, l, (s + _DA - X + Y)) +
                         eval_(m, l, (s + _DA + X - Y)) +
                         eval_(m, l, (s + _DA + X + Y)))
                };


    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX typename ::simpla::traits::value_type<TF>::type
    mapto(mesh_type const &m, TF const &expr, id_type s,
          ::simpla::integer_sequence<int, VOLUME, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;

        auto X = m.DI(0, s);
        auto Y = m.DI(1, s);
        auto Z = m.DI(2, s);
        s = (s | m.FULL_OVERFLOW_FLAG) - _DA;

        return (
                       eval_(m, l, ((s + _DA - X - Y - Z))) +
                       eval_(m, l, ((s + _DA - X - Y + Z))) +
                       eval_(m, l, ((s + _DA - X + Y - Z))) +
                       eval_(m, l, ((s + _DA - X + Y + Z))) +
                       eval_(m, l, ((s + _DA + X - Y - Z))) +
                       eval_(m, l, ((s + _DA + X - Y + Z))) +
                       eval_(m, l, ((s + _DA + X + Y - Z))) +
                       eval_(m, l, ((s + _DA + X + Y + Z)))

               ) * 0.125;
    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX typename ::simpla::traits::value_type<TF>::type
    mapto(mesh_type const &m, TF const &expr, id_type s,
          ::simpla::integer_sequence<int, VOLUME, FACE>) DECLARE_FUNCTION_SUFFIX
    {
        auto X = mesh_type::delta_index(mesh_type::dual(s));
        s = (s | m.FULL_OVERFLOW_FLAG) - m.DA;

        return (eval_(m, expr, s - X) + eval_(m, expr, s + X)) * 0.5;
    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX typename ::simpla::traits::value_type<TF>::type
    mapto(mesh_type const &m, TF const &expr, id_type s,
          ::simpla::integer_sequence<int, VOLUME, EDGE>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;
        auto X = mesh_type::delta_index(s);
        auto Y = mesh_type::rotate(X);
        auto Z = mesh_type::inverse_rotate(X);
        s = (s | m.FULL_OVERFLOW_FLAG) - _DA;

        return (
                eval_(m, l, s + _DA - Y - Z) +
                eval_(m, l, s + _DA - Y + Z) +
                eval_(m, l, s + _DA + Y - Z) +
                eval_(m, l, s + _DA + Y + Z)
        );
    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX ::simpla::nTuple<typename ::simpla::traits::value_type<TF>::type, 3>
    mapto(mesh_type const &m, TF const &expr, id_type s,
          ::simpla::integer_sequence<int, FACE, VOLUME>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;

        auto X = m.DI(0, mesh_type::dual(s));
        auto Y = m.DI(1, mesh_type::dual(s));
        auto Z = m.DI(2, mesh_type::dual(s));
        s = (s | mesh_type::FULL_OVERFLOW_FLAG) - _DA;

        return nTuple<traits::value_type_t<TF>, 3>
                {
                        (eval_(m, l, s + _DA - X) + eval_(m, l, s + _DA + X)) * 0.5,
                        (eval_(m, l, s + _DA - Y) + eval_(m, l, s + _DA + Y)) * 0.5,
                        (eval_(m, l, s + _DA - Z) + eval_(m, l, s + _DA + Z)) * 0.5

                };


    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX ::simpla::nTuple<typename ::simpla::traits::value_type<TF>::type, 3>
    mapto(mesh_type const &m, TF const &expr, id_type s,
          ::simpla::integer_sequence<int, EDGE, VOLUME>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;

        auto X = m.DI(0, s);
        auto Y = m.DI(1, s);
        auto Z = m.DI(2, s);
        s = (s | m.FULL_OVERFLOW_FLAG) - _DA;

        return ::simpla::nTuple<typename ::simpla::traits::value_type<TF>::type, 3>
                {
                        (eval_(m, l, s + _DA - Y - Z) + eval_(m, l, s + _DA - Y + Z) +
                         eval_(m, l, s + _DA + Y - Z) + eval_(m, l, s + _DA + Y + Z)),
                        (eval_(m, l, s + _DA - Z - X) + eval_(m, l, s + _DA - Z + X) +
                         eval_(m, l, s + _DA + Z - X) + eval_(m, l, s + _DA + Z + X)),
                        (eval_(m, l, s + _DA - X - Y) + eval_(m, l, s + _DA - X + Y) +
                         eval_(m, l, s + _DA + X - Y) + eval_(m, l, s + _DA + X + Y))
                };


    }


    //***************************************************************************************************
    //
    //! Form<IL> ^ Form<IR> => Form<IR+IL>
    template<typename ...T, int IL, int IR>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Wedge, T...>>>
    eval_(mesh_type const &m, Field<Expression<ct::Wedge, T...>> const &expr,
          id_type s, ::simpla::integer_sequence<int, IL, IR>) DECLARE_FUNCTION_SUFFIX
    {
        return m.inner_product(mapto(m, std::get<0>(expr.args), s, ::simpla::integer_sequence<int, IL, IR + IL>()),
                               mapto(m, std::get<1>(expr.args), s, ::simpla::integer_sequence<int, IR, IR + IL>()),
                               s);
    }


    template<typename TL, typename TR>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Wedge, TL, TR>>>
    eval_(mesh_type const &m, Field<Expression<ct::Wedge, TL, TR>> const &expr,
          id_type s, ::simpla::integer_sequence<int, EDGE, EDGE>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = std::get<0>(expr.args);
        auto const &r = std::get<1>(expr.args);

        auto Y = mesh_type::delta_index(mesh_type::rotate(mesh_type::dual(s)));
        auto Z = mesh_type::delta_index(
                mesh_type::inverse_rotate(mesh_type::dual(s)));

        return ((eval_(m, l, s - Y) + eval_(m, l, s + Y))
                * (eval_(m, l, s - Z) + eval_(m, l, s + Z)) * 0.25);
    }


    template<typename TL, typename TR, int I>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Cross, TL, TR>>>
    eval_(mesh_type const &m, Field<Expression<ct::Cross, TL, TR>> const &expr,
          id_type s, ::simpla::integer_sequence<int, I, I>) DECLARE_FUNCTION_SUFFIX
    {
        return cross(eval_(m, std::get<0>(expr.args), s), eval_(m, std::get<1>(expr.args), s));
    }

    template<typename TL, typename TR, int I>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Dot, TL, TR>>>
    eval_(mesh_type const &m, Field<Expression<ct::Dot, TL, TR>> const &expr,
          id_type s, ::simpla::integer_sequence<int, I, I>) DECLARE_FUNCTION_SUFFIX
    {
        return dot(eval_(m, std::get<0>(expr.args), s), eval_(m, std::get<1>(expr.args), s));
    }


    template<typename ...T, int ...I>
    constexpr DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::MapTo, T...> >>
    eval_(mesh_type const &m, Field<Expression<ct::MapTo, T...>> const &expr, id_type s,
          ::simpla::integer_sequence<int, I...>) DECLARE_FUNCTION_SUFFIX
    {
        return mapto(m, std::get<0>(expr.args), s, ::simpla::integer_sequence<int, I...>());
    };


    template<typename TOP, typename ... T>
    DECLARE_FUNCTION_PREFIX constexpr traits::primary_type_t<traits::value_type_t<Field<Expression<TOP, T...> >>>
    eval_(mesh_type const &m, Field<Expression<TOP, T...> > const &expr, id_type const &s) DECLARE_FUNCTION_SUFFIX
    {
        return eval_(m, expr, s, traits::iform_list_t<T...>());
    }

public:

    FiniteVolume() { }

    virtual ~FiniteVolume() { }

    template<typename T, int ...N>
    DECLARE_FUNCTION_PREFIX constexpr nTuple<T, N...> const &
    eval(mesh_type const &m, nTuple<T, N...> const &v, id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    template<typename TV, typename mesh_type, typename ... Others>
    DECLARE_FUNCTION_PREFIX constexpr TV &
    eval(mesh_type const &m, Field<TV, mesh_type, Others...> &f, id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return f[s];
    }

    template<typename T>
    DECLARE_FUNCTION_PREFIX
    constexpr traits::primary_type_t<traits::value_type_t<T>>
    eval(mesh_type const &m, T const &expr, id_type const &s) DECLARE_FUNCTION_SUFFIX
    {
        return (eval_(m, expr, s));
    }


};// struct DiffScheme<TGeo, diff_scheme::tags::finite_volume>

#undef DECLARE_FUNCTION_PREFIX
#undef DECLARE_FUNCTION_SUFFIX

} //namespace policy
} //namespace manifold

}// namespace simpla

#endif /* FDM_H_ */
