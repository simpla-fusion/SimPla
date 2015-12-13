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


#define DECLARE_FUNCTION_SUFFIX const
#define DECLARE_FUNCTION_PREFIX inline

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

    typedef TGeo geometry_type;

    typedef FiniteVolume<geometry_type> this_type;

    typedef typename geometry_type::id_type id_type;

    static const int ndims = geometry_type::ndims;

    geometry_type &m_geo_;

public:
    typedef this_type calculus_policy;

    template<typename TDict>
    void load(TDict const &) { }

    template<typename OS>
    OS &print(OS &os) const
    {
        os << "\t DiffScheme = { Type = \"Finite Volume\" }," << std::endl;

        return os;
    }

    void deploy() { }


private:
    ///***************************************************************************************************
    /// @name general_algebra General algebra
    /// @{
    ///***************************************************************************************************



    DECLARE_FUNCTION_PREFIX constexpr Real eval_(Real v, id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    DECLARE_FUNCTION_PREFIX constexpr int
    eval_(int v, id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    DECLARE_FUNCTION_PREFIX constexpr std::complex<Real>
    eval_(std::complex<Real> v, id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    template<typename T, size_t ...N>
    DECLARE_FUNCTION_PREFIX constexpr nTuple<T, N...> const &
    eval_(nTuple<T, N...> const &v, id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    template<typename ...T>
    DECLARE_FUNCTION_PREFIX traits::primary_type_t <nTuple<Expression<T...>>>
    eval_(nTuple<Expression<T...>> const &v, id_type s) DECLARE_FUNCTION_SUFFIX
    {
        traits::primary_type_t<nTuple<Expression<T...> > > res;
        res = v;
        return std::move(res);
    }

    template<typename TV, typename TM, typename ... Others>
    DECLARE_FUNCTION_PREFIX constexpr TV
    eval_(Field<TV, TM, Others...> const &f, id_type s) DECLARE_FUNCTION_SUFFIX
    {
        return f[s];
    }


    template<typename FExpr>
    DECLARE_FUNCTION_PREFIX constexpr traits::value_type_t <FExpr>
    get_v(FExpr const &f, id_type const s) DECLARE_FUNCTION_SUFFIX
    {
        return eval_(f, s) * m_geo_.volume(s);
    }

    template<typename FExpr>
    DECLARE_FUNCTION_PREFIX constexpr traits::value_type_t <FExpr>
    get_d(FExpr const &f, id_type const s) DECLARE_FUNCTION_SUFFIX
    {
        return eval_(f, s) * m_geo_.dual_volume(s);
    }


    template<typename ...T, int ... index>
    DECLARE_FUNCTION_PREFIX traits::primary_type_t <traits::value_type_t<Field<Expression<T...> >>>
    _invoke_helper(Field<Expression<T...> > const &expr, id_type s, index_sequence<index...>) DECLARE_FUNCTION_SUFFIX
    {
        traits::primary_type_t<traits::value_type_t<Field<Expression<T...> >>> res = (expr.m_op_(
                eval_(std::get<index>(expr.args), s)...));

        return std::move(res);
    }


    template<typename TOP, typename ... T>
    DECLARE_FUNCTION_PREFIX constexpr traits::primary_type_t <traits::value_type_t<Field<Expression<TOP, T...> >>>
    eval_(Field<Expression<TOP, T...> > const &expr, id_type const &s,
          traits::iform_list_t<T...>) DECLARE_FUNCTION_SUFFIX
    {
        return _invoke_helper(expr, s, typename make_index_sequence<sizeof...(T)>::type());
    }

    //***************************************************************************************************
    // Exterior algebra
    //***************************************************************************************************

    //! grad<0>
    template<typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t <Field<Expression<ct::ExteriorDerivative, T>>>
    eval_(Field<Expression<ct::ExteriorDerivative, T> > const &f,
          id_type s, ::simpla::integer_sequence<int, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        id_type D = geometry_type::delta_index(s);


        return (get_v(std::get<0>(f.args), s + D) - get_v(std::get<0>(f.args), s - D)) * m_geo_.inv_volume(s);


    }


    //! curl<1>
    template<typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t <Field<Expression<ct::ExteriorDerivative, T>>>
    eval_(Field<Expression<ct::ExteriorDerivative, T> > const &expr,
          id_type s, ::simpla::integer_sequence<int, EDGE>) DECLARE_FUNCTION_SUFFIX
    {

        id_type X = geometry_type::delta_index(geometry_type::dual(s));
        id_type Y = geometry_type::rotate(X);
        id_type Z = geometry_type::inverse_rotate(X);


        return (
                       (get_v(std::get<0>(expr.args), s + Y) - get_v(std::get<0>(expr.args), s - Y))

                       - (get_v(std::get<0>(expr.args), s + Z) - get_v(std::get<0>(expr.args), s - Z))

               ) * m_geo_.inv_volume(s);


    }

    //! div<2>
    template<typename T>
    constexpr DECLARE_FUNCTION_PREFIX traits::value_type_t <Field<Expression<ct::ExteriorDerivative, T>>>
    eval_(Field<Expression<ct::ExteriorDerivative, T> > const &expr,
          id_type s, ::simpla::integer_sequence<int, FACE>) DECLARE_FUNCTION_SUFFIX
    {

        return (get_v(std::get<0>(expr.args), s + geometry_type::_DI)

                - get_v(std::get<0>(expr.args), s - geometry_type::_DI)

                + get_v(std::get<0>(expr.args), s + geometry_type::_DJ)

                - get_v(std::get<0>(expr.args), s - geometry_type::_DJ)

                + get_v(std::get<0>(expr.args), s + geometry_type::_DK)

                - get_v(std::get<0>(expr.args), s - geometry_type::_DK)


               ) * m_geo_.inv_volume(s);

    }


    //! div<1>
    template<typename T>
    constexpr DECLARE_FUNCTION_PREFIX traits::value_type_t <Field<Expression<ct::CodifferentialDerivative, T>>>
    eval_(Field<Expression<ct::CodifferentialDerivative, T>> const &expr,
          id_type s, ::simpla::integer_sequence<int, EDGE>) DECLARE_FUNCTION_SUFFIX
    {

        return -(get_d(std::get<0>(expr.args), s + geometry_type::_DI)
                 - get_d(std::get<0>(expr.args), s - geometry_type::_DI)
                 + get_d(std::get<0>(expr.args), s + geometry_type::_DJ)
                 - get_d(std::get<0>(expr.args), s - geometry_type::_DJ)
                 + get_d(std::get<0>(expr.args), s + geometry_type::_DK)
                 - get_d(std::get<0>(expr.args), s - geometry_type::_DK)

        ) * m_geo_.inv_dual_volume(s);


    }

    //! curl<2>
    template<typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t <
    Field<Expression<ct::CodifferentialDerivative, T>>>
    eval_(Field<Expression<ct::CodifferentialDerivative, T>> const &expr,
          id_type s, ::simpla::integer_sequence<int, FACE>) DECLARE_FUNCTION_SUFFIX
    {

        id_type X = geometry_type::delta_index(s);
        id_type Y = geometry_type::rotate(X);
        id_type Z = geometry_type::inverse_rotate(X);


        return -((get_d(std::get<0>(expr.args), s + Y) - get_d(std::get<0>(expr.args), s - Y))
                 - (get_d(std::get<0>(expr.args), s + Z) - get_d(std::get<0>(expr.args), s - Z))
        ) * m_geo_.inv_dual_volume(s);
    }


    //! grad<3>
    template<typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t <
    Field<Expression<ct::CodifferentialDerivative, T>>>
    eval_(Field<Expression<ct::CodifferentialDerivative, T> > const &expr,
          id_type s, ::simpla::integer_sequence<int, VOLUME>) DECLARE_FUNCTION_SUFFIX
    {
        id_type D = geometry_type::delta_index(geometry_type::dual(s));

        return -(get_d(std::get<0>(expr.args), s + D) - get_d(std::get<0>(expr.args), s - D)) *
               m_geo_.inv_dual_volume(s);


    }

    //! *Form<IR> => Form<N-IL>


    template<typename T, int I>
    DECLARE_FUNCTION_PREFIX traits::value_type_t <Field<Expression<ct::HodgeStar, T> >>
    eval_(Field<Expression<ct::HodgeStar, T> > const &expr, id_type s,
          ::simpla::integer_sequence<int, I>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = std::get<0>(expr.args);

        int i = geometry_type::iform(s);
        id_type X = (i == VERTEX || i == VOLUME) ? geometry_type::_DI : geometry_type::delta_index(
                geometry_type::dual(s));
        id_type Y = geometry_type::rotate(X);
        id_type Z = geometry_type::inverse_rotate(X);


        return (
                       get_v(l, ((s - X) - Y) - Z) +
                       get_v(l, ((s - X) - Y) + Z) +
                       get_v(l, ((s - X) + Y) - Z) +
                       get_v(l, ((s - X) + Y) + Z) +
                       get_v(l, ((s + X) - Y) - Z) +
                       get_v(l, ((s + X) - Y) + Z) +
                       get_v(l, ((s + X) + Y) - Z) +
                       get_v(l, ((s + X) + Y) + Z)

               ) * m_geo_.inv_dual_volume(s) * 0.125;


    };


////***************************************************************************************************
//
////! map_to

    template<typename TF, int I>
    DECLARE_FUNCTION_PREFIX traits::value_type_t <TF>
    mapto(TF const &expr, id_type s, ::simpla::integer_sequence<int, I, I>) DECLARE_FUNCTION_SUFFIX
    {
        return eval_(expr, s);
    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX traits::value_type_t <TF>
    mapto(TF const &expr, id_type s, ::simpla::integer_sequence<int, VERTEX, EDGE>) DECLARE_FUNCTION_SUFFIX
    {

        id_type X = s & m_geo_._DA;

        s = (s | m_geo_.FULL_OVERFLOW_FLAG) - m_geo_._DA;

        return (eval_(expr, s + m_geo_._DA - X) + eval_(expr, s + m_geo_._DA + X)) * 0.5;
    }


    template<typename TV, typename ...Others>
    DECLARE_FUNCTION_PREFIX TV
    mapto(Field<nTuple<TV, 3>, Others...> const &expr, id_type s,
          ::simpla::integer_sequence<int, VERTEX, EDGE>) DECLARE_FUNCTION_SUFFIX
    {
        int n = geometry_type::sub_index(s);
        id_type X = s & m_geo_._DA;
        s = (s | m_geo_.FULL_OVERFLOW_FLAG) - m_geo_._DA;
        return (eval_(expr, s + m_geo_._DA - X)[n] + eval_(expr, s + m_geo_._DA + X)[n]) * 0.5;

    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX traits::value_type_t <TF>
    mapto(TF const &expr, id_type s, ::simpla::integer_sequence<int, VERTEX, FACE>) DECLARE_FUNCTION_SUFFIX
    {

        auto const &l = expr;
        auto X = geometry_type::delta_index(geometry_type::dual(s));
        auto Y = geometry_type::rotate(X);
        auto Z = geometry_type::inverse_rotate(X);
        s = (s | m_geo_.FULL_OVERFLOW_FLAG) - m_geo_._DA;

        return (
                eval_(l, (s + m_geo_._DA - Y - Z)) +
                eval_(l, (s + m_geo_._DA - Y + Z)) +
                eval_(l, (s + m_geo_._DA + Y - Z)) +
                eval_(l, (s + m_geo_._DA + Y + Z))
        );
    }


    template<typename TV, typename ...Others>
    DECLARE_FUNCTION_PREFIX TV
    mapto(Field<nTuple<TV, 3>, Others...> const &expr, id_type s,
          ::simpla::integer_sequence<int, VERTEX, FACE>) DECLARE_FUNCTION_SUFFIX
    {

        int n = geometry_type::sub_index(s);
        auto const &l = expr;
        auto X = geometry_type::delta_index(geometry_type::dual(s));
        auto Y = geometry_type::rotate(X);
        auto Z = geometry_type::inverse_rotate(X);
        s = (s | m_geo_.FULL_OVERFLOW_FLAG) - m_geo_._DA;

        return (
                eval_(l, (s + m_geo_._DA - Y - Z))[n] +
                eval_(l, (s + m_geo_._DA - Y + Z))[n] +
                eval_(l, (s + m_geo_._DA + Y - Z))[n] +
                eval_(l, (s + m_geo_._DA + Y + Z))[n]
        );
    }

    template<typename TF>
    DECLARE_FUNCTION_PREFIX traits::value_type_t <TF>
    mapto(TF const &expr, id_type s, ::simpla::integer_sequence<int, VERTEX, VOLUME>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;

        auto X = m_geo_.DI(0, s);
        auto Y = m_geo_.DI(1, s);
        auto Z = m_geo_.DI(2, s);
        s = (s | m_geo_.FULL_OVERFLOW_FLAG) - m_geo_._DA;

        return (eval_(l, s + m_geo_._DA - X - Y - Z) +
                eval_(l, s + m_geo_._DA - X - Y + Z) +
                eval_(l, s + m_geo_._DA - X + Y - Z) +
                eval_(l, s + m_geo_._DA - X + Y + Z) +
                eval_(l, s + m_geo_._DA + X - Y - Z) +
                eval_(l, s + m_geo_._DA + X - Y + Z) +
                eval_(l, s + m_geo_._DA + X + Y - Z) +
                eval_(l, s + m_geo_._DA + X + Y + Z)

        );
    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX nTuple<typename traits::value_type<TF>::type, 3>
    mapto(TF const &expr, id_type s, ::simpla::integer_sequence<int, EDGE, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        typedef nTuple<typename traits::value_type<TF>::type, 3> field_value_type;

        auto const &l = expr;

        id_type DA = m_geo_._DA;
        id_type X = m_geo_._D;
        id_type Y = X << m_geo_.ID_DIGITS;
        id_type Z = Y << m_geo_.ID_DIGITS;

        s = (s | m_geo_.FULL_OVERFLOW_FLAG) - m_geo_._DA;

        return field_value_type
                {
                        (eval_(l, s + m_geo_._DA - X) + eval_(l, s + m_geo_._DA + X)) * 0.5,
                        (eval_(l, s + m_geo_._DA - Y) + eval_(l, s + m_geo_._DA + Y)) * 0.5,
                        (eval_(l, s + m_geo_._DA - Z) + eval_(l, s + m_geo_._DA + Z)) * 0.5

                };


    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX nTuple<typename traits::value_type<TF>::type, 3>
    mapto(TF const &expr, id_type s, ::simpla::integer_sequence<int, FACE, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;

        id_type X = m_geo_._D;
        id_type Y = X << m_geo_.ID_DIGITS;
        id_type Z = Y << m_geo_.ID_DIGITS;

        s = (s | m_geo_.FULL_OVERFLOW_FLAG) - m_geo_._DA;

        return nTuple<typename ::simpla::traits::value_type<TF>::type, 3>
                {
                        (eval_(l, (s + m_geo_._DA - Y - Z)) + eval_(l, (s + m_geo_._DA - Y + Z)) +
                         eval_(l, (s + m_geo_._DA + Y - Z)) + eval_(l, (s + m_geo_._DA + Y + Z))),
                        (eval_(l, (s + m_geo_._DA - Z - X)) + eval_(l, (s + m_geo_._DA - Z + X)) +
                         eval_(l, (s + m_geo_._DA + Z - X)) + eval_(l, (s + m_geo_._DA + Z + X))),
                        (eval_(l, (s + m_geo_._DA - X - Y)) + eval_(l, (s + m_geo_._DA - X + Y)) +
                         eval_(l, (s + m_geo_._DA + X - Y)) + eval_(l, (s + m_geo_._DA + X + Y)))
                };


    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX typename ::simpla::traits::value_type<TF>::type
    mapto(TF const &expr, id_type s, ::simpla::integer_sequence<int, VOLUME, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;

        auto X = m_geo_.DI(0, s);
        auto Y = m_geo_.DI(1, s);
        auto Z = m_geo_.DI(2, s);
        s = (s | m_geo_.FULL_OVERFLOW_FLAG) - m_geo_._DA;

        return (
                       eval_(l, ((s + m_geo_._DA - X - Y - Z))) +
                       eval_(l, ((s + m_geo_._DA - X - Y + Z))) +
                       eval_(l, ((s + m_geo_._DA - X + Y - Z))) +
                       eval_(l, ((s + m_geo_._DA - X + Y + Z))) +
                       eval_(l, ((s + m_geo_._DA + X - Y - Z))) +
                       eval_(l, ((s + m_geo_._DA + X - Y + Z))) +
                       eval_(l, ((s + m_geo_._DA + X + Y - Z))) +
                       eval_(l, ((s + m_geo_._DA + X + Y + Z)))

               ) * 0.125;
    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX typename ::simpla::traits::value_type<TF>::type
    mapto(TF const &expr, id_type s, ::simpla::integer_sequence<int, VOLUME, FACE>) DECLARE_FUNCTION_SUFFIX
    {
        auto X = geometry_type::delta_index(geometry_type::dual(s));
        s = (s | m_geo_.FULL_OVERFLOW_FLAG) - m_geo_.DA;

        return (eval_(expr, s - X) + eval_(expr, s + X)) * 0.5;
    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX typename ::simpla::traits::value_type<TF>::type
    mapto(TF const &expr, id_type s, ::simpla::integer_sequence<int, VOLUME, EDGE>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;
        auto X = geometry_type::delta_index(s);
        auto Y = geometry_type::rotate(X);
        auto Z = geometry_type::inverse_rotate(X);
        s = (s | m_geo_.FULL_OVERFLOW_FLAG) - m_geo_._DA;

        return (
                eval_(l, s + m_geo_._DA - Y - Z) +
                eval_(l, s + m_geo_._DA - Y + Z) +
                eval_(l, s + m_geo_._DA + Y - Z) +
                eval_(l, s + m_geo_._DA + Y + Z)
        );
    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX ::simpla::nTuple<typename ::simpla::traits::value_type<TF>::type, 3>
    mapto(TF const &expr, id_type s, ::simpla::integer_sequence<int, FACE, VOLUME>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;

        auto X = m_geo_.DI(0, geometry_type::dual(s));
        auto Y = m_geo_.DI(1, geometry_type::dual(s));
        auto Z = m_geo_.DI(2, geometry_type::dual(s));
        s = (s | m_geo_.FULL_OVERFLOW_FLAG) - m_geo_._DA;

        return nTuple<traits::value_type_t<TF>, 3>
                {
                        (eval_(l, s + m_geo_._DA - X) + eval_(l, s + m_geo_._DA + X)) * 0.5,
                        (eval_(l, s + m_geo_._DA - Y) + eval_(l, s + m_geo_._DA + Y)) * 0.5,
                        (eval_(l, s + m_geo_._DA - Z) + eval_(l, s + m_geo_._DA + Z)) * 0.5

                };


    }


    template<typename TF>
    DECLARE_FUNCTION_PREFIX ::simpla::nTuple<typename ::simpla::traits::value_type<TF>::type, 3>
    mapto(TF const &expr, id_type s, ::simpla::integer_sequence<int, EDGE, VOLUME>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = expr;

        auto X = m_geo_.DI(0, s);
        auto Y = m_geo_.DI(1, s);
        auto Z = m_geo_.DI(2, s);
        s = (s | m_geo_.FULL_OVERFLOW_FLAG) - m_geo_._DA;

        return ::simpla::nTuple<typename ::simpla::traits::value_type<TF>::type, 3>
                {
                        (eval_(l, s + m_geo_._DA - Y - Z) + eval_(l, s + m_geo_._DA - Y + Z) +
                         eval_(l, s + m_geo_._DA + Y - Z) + eval_(l, s + m_geo_._DA + Y + Z)),
                        (eval_(l, s + m_geo_._DA - Z - X) + eval_(l, s + m_geo_._DA - Z + X) +
                         eval_(l, s + m_geo_._DA + Z - X) + eval_(l, s + m_geo_._DA + Z + X)),
                        (eval_(l, s + m_geo_._DA - X - Y) + eval_(l, s + m_geo_._DA - X + Y) +
                         eval_(l, s + m_geo_._DA + X - Y) + eval_(l, s + m_geo_._DA + X + Y))
                };


    }


    //***************************************************************************************************
    //
    //! Form<IL> ^ Form<IR> => Form<IR+IL>
    template<typename ...T, int IL, int IR>
    DECLARE_FUNCTION_PREFIX traits::value_type_t <Field<Expression<ct::Wedge, T...>>>
    eval_(Field<Expression<ct::Wedge, T...>> const &expr,
          id_type s, ::simpla::integer_sequence<int, IL, IR>) DECLARE_FUNCTION_SUFFIX
    {
        return m_geo_.inner_product(mapto(std::get<0>(expr.args), s, ::simpla::integer_sequence<int, IL, IR + IL>()),
                                    mapto(std::get<1>(expr.args), s, ::simpla::integer_sequence<int, IR, IR + IL>()),
                                    s);
    }


    template<typename TL, typename TR>
    DECLARE_FUNCTION_PREFIX traits::value_type_t <Field<Expression<ct::Wedge, TL, TR>>>
    eval_(Field<Expression<ct::Wedge, TL, TR>> const &expr,
          id_type s, ::simpla::integer_sequence<int, EDGE, EDGE>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = std::get<0>(expr.args);
        auto const &r = std::get<1>(expr.args);

        auto Y = geometry_type::delta_index(geometry_type::rotate(geometry_type::dual(s)));
        auto Z = geometry_type::delta_index(
                geometry_type::inverse_rotate(geometry_type::dual(s)));

        return ((eval_(l, s - Y) + eval_(l, s + Y))
                * (eval_(l, s - Z) + eval_(l, s + Z)) * 0.25);
    }


    template<typename TL, typename TR, int I>
    DECLARE_FUNCTION_PREFIX traits::value_type_t <Field<Expression<ct::Cross, TL, TR>>>
    eval_(Field<Expression<ct::Cross, TL, TR>> const &expr,
          id_type s, ::simpla::integer_sequence<int, I, I>) DECLARE_FUNCTION_SUFFIX
    {
        return cross(eval_(std::get<0>(expr.args), s), eval_(std::get<1>(expr.args), s));
    }

    template<typename TL, typename TR, int I>
    DECLARE_FUNCTION_PREFIX traits::value_type_t <Field<Expression<ct::Dot, TL, TR>>>
    eval_(Field<Expression<ct::Dot, TL, TR>> const &expr,
          id_type s, ::simpla::integer_sequence<int, I, I>) DECLARE_FUNCTION_SUFFIX
    {
        return dot(eval_(std::get<0>(expr.args), s), eval_(std::get<1>(expr.args), s));
    }


    template<typename ...T, int ...I>
    constexpr DECLARE_FUNCTION_PREFIX traits::value_type_t <Field<Expression<ct::MapTo, T...> >>
    eval_(Field<Expression<ct::MapTo, T...>> const &expr, id_type s,
          ::simpla::integer_sequence<int, I...>) DECLARE_FUNCTION_SUFFIX
    {
        return mapto(std::get<0>(expr.args), s, ::simpla::integer_sequence<int, I...>());
    };


    template<typename TOP, typename ... T>
    DECLARE_FUNCTION_PREFIX constexpr traits::primary_type_t <traits::value_type_t<Field<Expression<TOP, T...> >>>
    eval_(Field<Expression<TOP, T...> > const
          &expr,
          id_type const &s
    ) DECLARE_FUNCTION_SUFFIX
    {
        return eval_(expr, s, traits::iform_list_t<T...>());
    }

public:

    FiniteVolume(geometry_type &geo) :
            m_geo_(geo)
    {
    }

    virtual ~FiniteVolume()
    {
    }

    template<typename T>
    DECLARE_FUNCTION_PREFIX
    constexpr traits::primary_type_t <traits::value_type_t<T>>
    eval(T const &expr, id_type const &s) DECLARE_FUNCTION_SUFFIX
    {
        return (eval_(expr, s));
    };


};// struct DiffScheme<TGeo, diff_scheme::tags::finite_volume>

#undef DECLARE_FUNCTION_PREFIX
#undef DECLARE_FUNCTION_SUFFIX

} //namespace policy
} //namespace manifold

}// namespace simpla

#endif /* FDM_H_ */
