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
#include "../../gtl/ExpressionTemplate.h"

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
using namespace simpla::mesh;


namespace ct=calculus::tags;
namespace st=simpla::traits;


/**
 * @ingroup diff_scheme
 *
 * finite volume
 */
template<typename TM>
struct FiniteVolume
{
public:
    typedef FiniteVolume<TM> calculus_policy;

private:

    typedef TM mesh_type;

    typedef FiniteVolume<TM> this_type;


    ///***************************************************************************************************
    /// @name general_algebra General algebra
    /// @{
    ///***************************************************************************************************


    template<typename T> inline constexpr auto
    eval_(T const v, id_type s, st::is_primary_t<T> *_p = nullptr) const { return v; }


    template<typename T> inline constexpr auto
    eval_(T const &v, id_type s, st::is_expression_ntuple_t<T> *_p = nullptr) const
    {
        traits::primary_type_t<T> res;
        res = v;
        return std::move(res);
    }

    template<typename T> inline constexpr auto
    eval_(T const &f, id_type s, st::is_primary_field_t<T> *_p = nullptr) const { return f[s]; }

    template<typename T> inline constexpr auto
    eval_(T &f, id_type s, st::is_primary_field_t<T> *_p = nullptr) const { return f[s]; }


    template<typename T> inline constexpr auto
    get_v(T const &f, id_type s, st::is_field_t<T> *_p = nullptr) const { return eval_(f, s) * m_.volume(s); }

    template<typename T> inline constexpr auto
    get_d(T const &f, id_type s, st::is_field_t<T> *_p = nullptr) const { return eval_(f, s) * m_.dual_volume(s); }


    template<typename ...T, int ... index> inline auto
    _invoke_helper(Field<Expression<T...> > const &expr, id_type s, index_sequence<index...>) const
    {
        traits::primary_type_t<traits::value_type_t<Field<Expression<T...> >>> res =
                (expr.m_op_(eval_(std::get<index>(expr.args), s)...));

        return std::move(res);
    }


    template<typename TOP, typename ... T> inline constexpr auto
    eval_(Field<Expression<TOP, T...> > const &expr, id_type const &s, traits::iform_list_t<T...>) const
    {
        return _invoke_helper(expr, s, typename make_index_sequence<sizeof...(T)>::type());
    }

    //***************************************************************************************************
    // Exterior algebra
    //***************************************************************************************************

    //! grad<0>
    template<typename T> inline auto
    eval_(Field<Expression<ct::ExteriorDerivative, T> > const &f, id_type s, index_sequence<VERTEX>) const
    {
        id_type D = mesh_type::delta_index(s);
        return (get_v(std::get<0>(f.args), s + D) - get_v(std::get<0>(f.args), s - D)) * m_.inv_volume(s);
    }


    //! curl<1>
    template<typename T> inline auto
    eval_(Field<Expression<ct::ExteriorDerivative, T> > const &expr, id_type s, index_sequence<EDGE>) const
    {

        id_type X = mesh_type::delta_index(mesh_type::dual(s));
        id_type Y = mesh_type::rotate(X);
        id_type Z = mesh_type::inverse_rotate(X);


        return ((get_v(std::get<0>(expr.args), s + Y) - get_v(std::get<0>(expr.args), s - Y))
                - (get_v(std::get<0>(expr.args), s + Z) - get_v(std::get<0>(expr.args), s - Z))
               ) * m_.inv_volume(s);


    }

    //! div<2>
    template<typename T> inline constexpr auto
    eval_(Field<Expression<ct::ExteriorDerivative, T> > const &expr, id_type s, index_sequence<FACE>) const
    {

        return (get_v(std::get<0>(expr.args), s + mesh_type::_DI)

                - get_v(std::get<0>(expr.args), s - mesh_type::_DI)

                + get_v(std::get<0>(expr.args), s + mesh_type::_DJ)

                - get_v(std::get<0>(expr.args), s - mesh_type::_DJ)

                + get_v(std::get<0>(expr.args), s + mesh_type::_DK)

                - get_v(std::get<0>(expr.args), s - mesh_type::_DK)


               ) * m_.inv_volume(s);

    }


    //! div<1>
    template<typename T> constexpr inline auto
    eval_(Field<Expression<ct::CodifferentialDerivative, T>> const &expr, id_type s, index_sequence<EDGE>) const
    {

        return -(get_d(std::get<0>(expr.args), s + mesh_type::_DI)
                 - get_d(std::get<0>(expr.args), s - mesh_type::_DI)
                 + get_d(std::get<0>(expr.args), s + mesh_type::_DJ)
                 - get_d(std::get<0>(expr.args), s - mesh_type::_DJ)
                 + get_d(std::get<0>(expr.args), s + mesh_type::_DK)
                 - get_d(std::get<0>(expr.args), s - mesh_type::_DK)

        ) * m_.inv_dual_volume(s);


    }

    //! curl<2>
    template<typename T> constexpr inline auto
    eval_(Field<Expression<ct::CodifferentialDerivative, T>> const &expr, id_type s, index_sequence<FACE>) const
    {

        id_type X = mesh_type::delta_index(s);
        id_type Y = mesh_type::rotate(X);
        id_type Z = mesh_type::inverse_rotate(X);


        return -((get_d(std::get<0>(expr.args), s + Y) - get_d(std::get<0>(expr.args), s - Y))
                 - (get_d(std::get<0>(expr.args), s + Z) - get_d(std::get<0>(expr.args), s - Z))
        ) * m_.inv_dual_volume(s);
    }


    //! grad<3>
    template<typename T> constexpr inline auto
    eval_(Field<Expression<ct::CodifferentialDerivative, T> > const &expr, id_type s, index_sequence<VOLUME>) const
    {
        id_type D = mesh_type::delta_index(mesh_type::dual(s));

        return -(get_d(std::get<0>(expr.args), s + D) - get_d(std::get<0>(expr.args), s - D)) *
               m_.inv_dual_volume(s);


    }

    //! *Form<IR> => Form<N-IL>


    template<typename T, int I> constexpr inline auto
    eval_(Field<Expression<ct::HodgeStar, T> > const &expr, id_type s, index_sequence<I>) const
    {
        auto const &l = std::get<0>(expr.args);

        int i = mesh_type::iform(s);
        id_type X = (i == VERTEX || i == VOLUME) ? mesh_type::_DI : mesh_type::delta_index(
                mesh_type::dual(s));
        id_type Y = mesh_type::rotate(X);
        id_type Z = mesh_type::inverse_rotate(X);


        return (
                       get_v(l, ((s - X) - Y) - Z) +
                       get_v(l, ((s - X) - Y) + Z) +
                       get_v(l, ((s - X) + Y) - Z) +
                       get_v(l, ((s - X) + Y) + Z) +
                       get_v(l, ((s + X) - Y) - Z) +
                       get_v(l, ((s + X) - Y) + Z) +
                       get_v(l, ((s + X) + Y) - Z) +
                       get_v(l, ((s + X) + Y) + Z)

               ) * m_.inv_dual_volume(s) * 0.125;


    };


////***************************************************************************************************
//
////! map_to

    template<typename TF, int I> constexpr inline auto
    mapto(TF const &expr, id_type s, index_sequence<I, I>) const { return eval_(expr, s); }


    template<typename TF> constexpr inline auto
    mapto(TF const &expr, id_type s, index_sequence<VERTEX, EDGE>) const
    {

        id_type X = s & mesh_type::_DA;

        s = (s | m_.FULL_OVERFLOW_FLAG) - mesh_type::_DA;

        return (eval_(expr, s + mesh_type::_DA - X) + eval_(expr, s + mesh_type::_DA + X)) *
               0.5;
    }


    template<typename TV, typename ...Others> constexpr inline auto
    mapto(Field<nTuple<TV, 3>, Others...> const &expr, id_type s, index_sequence<VERTEX, EDGE>) const
    {
        int n = mesh_type::sub_index(s);
        id_type X = s & mesh_type::_DA;
        s = (s | m_.FULL_OVERFLOW_FLAG) - mesh_type::_DA;
        return (eval_(expr, s + mesh_type::_DA - X)[n] +
                eval_(expr, s + mesh_type::_DA + X)[n]) * 0.5;

    }


    template<typename TF> constexpr inline auto
    mapto(TF const &expr, id_type s, index_sequence<VERTEX, FACE>) const
    {

        auto const &l = expr;
        auto X = mesh_type::delta_index(mesh_type::dual(s));
        auto Y = mesh_type::rotate(X);
        auto Z = mesh_type::inverse_rotate(X);
        s = (s | m_.FULL_OVERFLOW_FLAG) - mesh_type::_DA;

        return (
                eval_(l, (s + mesh_type::_DA - Y - Z)) +
                eval_(l, (s + mesh_type::_DA - Y + Z)) +
                eval_(l, (s + mesh_type::_DA + Y - Z)) +
                eval_(l, (s + mesh_type::_DA + Y + Z))
        );
    }


    template<typename TV, typename ...Others> constexpr inline auto
    mapto(Field<nTuple<TV, 3>, Others...> const &expr, id_type s, index_sequence<VERTEX, FACE>) const
    {

        int n = mesh_type::sub_index(s);
        auto const &l = expr;
        auto X = mesh_type::delta_index(mesh_type::dual(s));
        auto Y = mesh_type::rotate(X);
        auto Z = mesh_type::inverse_rotate(X);
        s = (s | m_.FULL_OVERFLOW_FLAG) - mesh_type::_DA;

        return (
                eval_(l, (s + mesh_type::_DA - Y - Z))[n] +
                eval_(l, (s + mesh_type::_DA - Y + Z))[n] +
                eval_(l, (s + mesh_type::_DA + Y - Z))[n] +
                eval_(l, (s + mesh_type::_DA + Y + Z))[n]
        );
    }

    template<typename TF> constexpr inline auto
    mapto(TF const &expr, id_type s, index_sequence<VERTEX, VOLUME>) const
    {
        auto const &l = expr;

        auto X = m_.DI(0, s);
        auto Y = m_.DI(1, s);
        auto Z = m_.DI(2, s);
        s = (s | m_.FULL_OVERFLOW_FLAG) - mesh_type::_DA;

        return (eval_(l, s + mesh_type::_DA - X - Y - Z) +
                eval_(l, s + mesh_type::_DA - X - Y + Z) +
                eval_(l, s + mesh_type::_DA - X + Y - Z) +
                eval_(l, s + mesh_type::_DA - X + Y + Z) +
                eval_(l, s + mesh_type::_DA + X - Y - Z) +
                eval_(l, s + mesh_type::_DA + X - Y + Z) +
                eval_(l, s + mesh_type::_DA + X + Y - Z) +
                eval_(l, s + mesh_type::_DA + X + Y + Z)

        );
    }


    template<typename TF> constexpr inline auto
    mapto(TF const &expr, id_type s, index_sequence<EDGE, VERTEX>) const
    {
        typedef nTuple<typename traits::value_type<TF>::type, 3> field_value_type;

        auto const &l = expr;

        id_type DA = mesh_type::_DA;
        id_type X = mesh_type::_D;
        id_type Y = X << mesh_type::ID_DIGITS;
        id_type Z = Y << mesh_type::ID_DIGITS;

        s = (s | m_.FULL_OVERFLOW_FLAG) - mesh_type::_DA;

        return field_value_type
                {
                        (eval_(l, s + mesh_type::_DA - X) +
                         eval_(l, s + mesh_type::_DA + X)) * 0.5,
                        (eval_(l, s + mesh_type::_DA - Y) +
                         eval_(l, s + mesh_type::_DA + Y)) * 0.5,
                        (eval_(l, s + mesh_type::_DA - Z) +
                         eval_(l, s + mesh_type::_DA + Z)) * 0.5

                };


    }


    template<typename TF> constexpr inline auto
    mapto(TF const &expr, id_type s, index_sequence<FACE, VERTEX>) const
    {
        auto const &l = expr;

        id_type X = mesh_type::_D;
        id_type Y = X << mesh_type::ID_DIGITS;;
        id_type Z = Y << mesh_type::ID_DIGITS;;

        s = (s | m_.FULL_OVERFLOW_FLAG) - mesh_type::_DA;

        return nTuple<typename ::simpla::traits::value_type<TF>::type, 3>
                {
                        (eval_(l, (s + mesh_type::_DA - Y - Z)) +
                         eval_(l, (s + mesh_type::_DA - Y + Z)) +
                         eval_(l, (s + mesh_type::_DA + Y - Z)) +
                         eval_(l, (s + mesh_type::_DA + Y + Z))),
                        (eval_(l, (s + mesh_type::_DA - Z - X)) +
                         eval_(l, (s + mesh_type::_DA - Z + X)) +
                         eval_(l, (s + mesh_type::_DA + Z - X)) +
                         eval_(l, (s + mesh_type::_DA + Z + X))),
                        (eval_(l, (s + mesh_type::_DA - X - Y)) +
                         eval_(l, (s + mesh_type::_DA - X + Y)) +
                         eval_(l, (s + mesh_type::_DA + X - Y)) +
                         eval_(l, (s + mesh_type::_DA + X + Y)))
                };


    }


    template<typename TF> constexpr inline auto
    mapto(TF const &expr, id_type s, index_sequence<VOLUME, VERTEX>) const
    {
        auto const &l = expr;

        auto X = m_.DI(0, s);
        auto Y = m_.DI(1, s);
        auto Z = m_.DI(2, s);
        s = (s | mesh_type::FULL_OVERFLOW_FLAG) - mesh_type::_DA;

        return (
                       eval_(l, ((s + mesh_type::_DA - X - Y - Z))) +
                       eval_(l, ((s + mesh_type::_DA - X - Y + Z))) +
                       eval_(l, ((s + mesh_type::_DA - X + Y - Z))) +
                       eval_(l, ((s + mesh_type::_DA - X + Y + Z))) +
                       eval_(l, ((s + mesh_type::_DA + X - Y - Z))) +
                       eval_(l, ((s + mesh_type::_DA + X - Y + Z))) +
                       eval_(l, ((s + mesh_type::_DA + X + Y - Z))) +
                       eval_(l, ((s + mesh_type::_DA + X + Y + Z)))

               ) * 0.125;
    }


    template<typename TF> constexpr inline auto
    mapto(TF const &expr, id_type s, index_sequence<VOLUME, FACE>) const
    {
        auto X = mesh_type::delta_index(mesh_type::dual(s));
        s = (s | mesh_type::FULL_OVERFLOW_FLAG) - mesh_type::DA;

        return (eval_(expr, s - X) + eval_(expr, s + X)) * 0.5;
    }


    template<typename TF> constexpr inline auto
    mapto(TF const &expr, id_type s, index_sequence<VOLUME, EDGE>) const
    {
        auto const &l = expr;
        auto X = mesh_type::delta_index(s);
        auto Y = mesh_type::rotate(X);
        auto Z = mesh_type::inverse_rotate(X);
        s = (s | mesh_type::FULL_OVERFLOW_FLAG) - mesh_type::_DA;

        return (
                eval_(l, s + mesh_type::_DA - Y - Z) +
                eval_(l, s + mesh_type::_DA - Y + Z) +
                eval_(l, s + mesh_type::_DA + Y - Z) +
                eval_(l, s + mesh_type::_DA + Y + Z)
        );
    }


    template<typename TF> constexpr inline auto
    mapto(TF const &expr, id_type s, index_sequence<FACE, VOLUME>) const
    {
        auto const &l = expr;

        auto X = m_.DI(0, mesh_type::dual(s));
        auto Y = m_.DI(1, mesh_type::dual(s));
        auto Z = m_.DI(2, mesh_type::dual(s));
        s = (s | mesh_type::FULL_OVERFLOW_FLAG) - mesh_type::_DA;

        return nTuple<traits::value_type_t<TF>, 3>
                {
                        (eval_(l, s + mesh_type::_DA - X) +
                         eval_(l, s + mesh_type::_DA + X)) * 0.5,
                        (eval_(l, s + mesh_type::_DA - Y) +
                         eval_(l, s + mesh_type::_DA + Y)) * 0.5,
                        (eval_(l, s + mesh_type::_DA - Z) +
                         eval_(l, s + mesh_type::_DA + Z)) * 0.5

                };


    }


    template<typename TF> constexpr inline auto
    mapto(TF const &expr, id_type s, index_sequence<EDGE, VOLUME>) const
    {
        auto const &l = expr;

        auto X = m_.DI(0, s);
        auto Y = m_.DI(1, s);
        auto Z = m_.DI(2, s);
        s = (s | mesh_type::FULL_OVERFLOW_FLAG) - mesh_type::_DA;

        return ::simpla::nTuple<typename ::simpla::traits::value_type<TF>::type, 3>
                {
                        (eval_(l, s + mesh_type::_DA - Y - Z) +
                         eval_(l, s + mesh_type::_DA - Y + Z) +
                         eval_(l, s + mesh_type::_DA + Y - Z) +
                         eval_(l, s + mesh_type::_DA + Y + Z)),
                        (eval_(l, s + mesh_type::_DA - Z - X) +
                         eval_(l, s + mesh_type::_DA - Z + X) +
                         eval_(l, s + mesh_type::_DA + Z - X) +
                         eval_(l, s + mesh_type::_DA + Z + X)),
                        (eval_(l, s + mesh_type::_DA - X - Y) +
                         eval_(l, s + mesh_type::_DA - X + Y) +
                         eval_(l, s + mesh_type::_DA + X - Y) +
                         eval_(l, s + mesh_type::_DA + X + Y))
                };


    }


    //***************************************************************************************************
    //
    //! Form<IL> ^ Form<IR> => Form<IR+IL>
    template<typename ...T, int IL, int IR> constexpr inline auto
    eval_(Field<Expression<ct::Wedge, T...>> const &expr, id_type s, index_sequence<IL, IR>) const
    {
        return m_.inner_product(mapto(std::get<0>(expr.args), s, index_sequence<IL, IR + IL>()),
                                mapto(std::get<1>(expr.args), s, index_sequence<IR, IR + IL>()),
                                s);
    }


    template<typename TL, typename TR> constexpr inline auto
    eval_(Field<Expression<ct::Wedge, TL, TR>> const &expr, id_type s, index_sequence<EDGE, EDGE>) const
    {
        auto const &l = std::get<0>(expr.args);
        auto const &r = std::get<1>(expr.args);

        auto Y = mesh_type::delta_index(mesh_type::rotate(mesh_type::dual(s)));
        auto Z = mesh_type::delta_index(
                mesh_type::inverse_rotate(mesh_type::dual(s)));

        return ((eval_(l, s - Y) + eval_(l, s + Y))
                * (eval_(l, s - Z) + eval_(l, s + Z)) * 0.25);
    }


    template<typename TL, typename TR, int I> constexpr inline auto
    eval_(Field<Expression<ct::Cross, TL, TR>> const &expr,
          id_type s, index_sequence<I, I>) const
    {
        return cross(eval_(std::get<0>(expr.args), s), eval_(std::get<1>(expr.args), s));
    }

    template<typename TL, typename TR, int I> constexpr inline auto
    eval_(Field<Expression<ct::Dot, TL, TR>> const &expr,
          id_type s, index_sequence<I, I>) const
    {
        return dot(eval_(std::get<0>(expr.args), s), eval_(std::get<1>(expr.args), s));
    }


    template<typename ...T, int ...I> constexpr inline auto
    eval_(Field<Expression<ct::MapTo, T...>> const &expr, id_type s,
          index_sequence<I...>) const
    {
        return mapto(std::get<0>(expr.args), s, index_sequence<I...>());
    };


    template<typename TOP, typename ... T> constexpr inline auto
    eval_(Field<Expression<TOP, T...> > const &expr, id_type const &s) const
    {
        return eval_(expr, s, traits::iform_list_t<T...>());
    }

public:

    FiniteVolume(mesh_type const &m) : m_(m) { }

    virtual ~FiniteVolume() { }


    template<typename T> inline constexpr auto
    eval(T &f, id_type s, traits::is_primary_field_t<T> *p = nullptr) const { return f[s]; }

    template<typename T> inline constexpr auto
    eval(T const &f, id_type s, traits::is_primary_field_t<T> *p = nullptr) const { return f[s]; }

    template<typename T> inline constexpr auto
    eval(T const &expr, id_type s, traits::is_expression_field_t<T> *p = nullptr) const { return (eval_(expr, s)); }

    template<typename T> inline constexpr auto
    eval(T const &expr, id_type s, traits::is_primary_t<T> *p = nullptr) const { return (eval_(expr, s)); }

private:
    mesh_type const &m_;

};// struct DiffScheme<TGeo, diff_scheme::tags::finite_volume>

#undef inline
#undef const

} //namespace policy
} //namespace Manifold

}// namespace simpla

#endif /* FDM_H_ */
