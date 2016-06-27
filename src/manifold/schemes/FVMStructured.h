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
#include "../../sp_def.h"
#include "../../gtl/type_traits.h"
#include "../../gtl/ExpressionTemplate.h"
#include "../Calculus.h"
#include "../ManifoldTraits.h"


namespace simpla { namespace manifold { namespace schemes
{
using namespace simpla::mesh;


template<typename TM, class Enable = void>
struct FiniteVolume { };

namespace ct=calculus::tags;
namespace st=simpla::traits;


/**
 * @ingroup diff_scheme
 *
 * finite volume
 */
template<typename TM>
struct FiniteVolume<TM, std::enable_if_t<std::is_base_of<mesh::MeshEntityIdCoder, TM>::value>>
{
    typedef FiniteVolume<TM, std::enable_if_t<std::is_base_of<mesh::MeshEntityIdCoder, TM>::value>> this_type;
    typedef TM mesh_type;
    mesh_type const &m;
    typedef mesh::MeshEntityIdCoder M;
    typedef mesh::MeshEntityId id_type;
public:
    typedef this_type calculus_policy;

    FiniteVolume(TM const &m_) : m(m_) { }

    virtual ~FiniteVolume() { }

    static std::string class_name() { return "FiniteVolume"; }

    virtual std::ostream &print(std::ostream &os, int indent = 1) const
    {
        os << std::setw(indent) << " " << "[FiniteVolume]," << std::endl;
        return os;
    }

    void deploy() { }

public:

    template<typename ...T>
    inline constexpr traits::value_type_t<Field<T...> >
    eval(Field<T...> &expr, id_type s) const { return eval_(expr, s); }

    template<typename ...T>
    inline constexpr traits::value_type_t<Field<T...> >
    eval(Field<T...> const &expr, id_type s) const { return eval_(expr, s); }

    template<typename T>
    inline constexpr T &eval(T &f, id_type s) const { return f; }


    template<typename T>
    inline constexpr T const &eval(T const &f, id_type s) const { return f; }

private:


    ///***************************************************************************************************
    /// @name general_algebra General algebra
    /// @{
    ///***************************************************************************************************


    template<typename T>
    inline constexpr T const &
    eval_(T const &v, id_type s, st::is_primary_t<T> *_p = nullptr) const { return v; }


    template<typename T>
    inline constexpr traits::primary_type_t<T>
    eval_(T const &v, id_type s, st::is_expression_ntuple_t<T> *_p = nullptr) const
    {
        traits::primary_type_t<T> res;
        res = v;
        return std::move(res);
    }

    template<typename TV, typename OM, typename ...Others>
    inline constexpr
    typename traits::value_type<Field<TV, OM, Others...>>::type &
    eval_(Field<TV, OM, Others...> &f, id_type s) const { return f[s]; };

    template<typename TV, typename OM, typename ...Others>
    inline constexpr
    typename traits::value_type<Field<TV, OM, Others...>>::type const &
    eval_(Field<TV, OM, Others...> const &f, id_type s) const { return f[s]; };

    template<typename TOP, typename ... T>
    inline constexpr traits::value_type_t<Field<Expression<TOP, T...> > >
    eval_(Field<Expression<TOP, T...> > const &expr, id_type const &s) const
    {
        return eval_(expr, s, traits::iform_list_t<T...>());
    }


    template<typename FExpr>
    inline constexpr traits::value_type_t<FExpr>
    get_v(FExpr const &f, id_type const s) const
    {
        return eval_(f, s) * m.volume(s);
    }

    template<typename FExpr>
    inline constexpr traits::value_type_t<FExpr>
    get_d(FExpr const &f, id_type const s) const
    {
        return eval_(f, s) * m.dual_volume(s);
    }


    //***************************************************************************************************
    // Exterior algebra
    //***************************************************************************************************

    //! grad<0>
    template<typename T>
    inline traits::value_type_t<Field<Expression<ct::ExteriorDerivative, T>>>
    eval_(Field<Expression<ct::ExteriorDerivative, T> > const &f, id_type s, index_sequence<mesh::VERTEX>) const
    {
        id_type D = M::delta_index(s);
        return (get_v(std::get<0>(f.args), s + D) - get_v(std::get<0>(f.args), s - D)) * m.inv_volume(s);
    }


    //! curl<1>
    template<typename T>
    inline traits::value_type_t<Field<Expression<ct::ExteriorDerivative, T>>>
    eval_(Field<Expression<ct::ExteriorDerivative, T> > const &expr, id_type s, index_sequence<EDGE>) const
    {

        id_type X = M::delta_index(M::dual(s));
        id_type Y = M::rotate(X);
        id_type Z = M::inverse_rotate(X);


        return ((get_v(std::get<0>(expr.args), s + Y) - get_v(std::get<0>(expr.args), s - Y))
                - (get_v(std::get<0>(expr.args), s + Z) - get_v(std::get<0>(expr.args), s - Z))
               ) * m.inv_volume(s);


    }

    //! div<2>
    template<typename T>
    constexpr inline traits::value_type_t<Field<Expression<ct::ExteriorDerivative, T>>>
    eval_(Field<Expression<ct::ExteriorDerivative, T> > const &expr, id_type s, index_sequence<FACE>) const
    {

        return (get_v(std::get<0>(expr.args), s + M::_DI)

                - get_v(std::get<0>(expr.args), s - M::_DI)

                + get_v(std::get<0>(expr.args), s + M::_DJ)

                - get_v(std::get<0>(expr.args), s - M::_DJ)

                + get_v(std::get<0>(expr.args), s + M::_DK)

                - get_v(std::get<0>(expr.args), s - M::_DK)


               ) * m.inv_volume(s);

    }


    //! div<1>
    template<typename T>
    constexpr inline traits::value_type_t<Field<Expression<ct::CodifferentialDerivative, T>>>
    eval_(Field<Expression<ct::CodifferentialDerivative, T>> const &expr, id_type s, index_sequence<EDGE>) const
    {

        return -(get_d(std::get<0>(expr.args), s + M::_DI)
                 - get_d(std::get<0>(expr.args), s - M::_DI)
                 + get_d(std::get<0>(expr.args), s + M::_DJ)
                 - get_d(std::get<0>(expr.args), s - M::_DJ)
                 + get_d(std::get<0>(expr.args), s + M::_DK)
                 - get_d(std::get<0>(expr.args), s - M::_DK)

        ) * m.inv_dual_volume(s);


    }

    //! curl<2>
    template<typename T>
    inline traits::value_type_t<
            Field<Expression<ct::CodifferentialDerivative, T>>>
    eval_(Field<Expression<ct::CodifferentialDerivative, T>> const &expr, id_type s, index_sequence<FACE>) const
    {

        id_type X = M::delta_index(s);
        id_type Y = M::rotate(X);
        id_type Z = M::inverse_rotate(X);


        return -((get_d(std::get<0>(expr.args), s + Y) - get_d(std::get<0>(expr.args), s - Y))
                 - (get_d(std::get<0>(expr.args), s + Z) - get_d(std::get<0>(expr.args), s - Z))
        ) * m.inv_dual_volume(s);
    }


    //! grad<3>
    template<typename T>
    inline traits::value_type_t<
            Field<Expression<ct::CodifferentialDerivative, T>>>
    eval_(Field<Expression<ct::CodifferentialDerivative, T> > const &expr, id_type s, index_sequence<VOLUME>) const
    {
        id_type D = M::delta_index(M::dual(s));

        return -(get_d(std::get<0>(expr.args), s + D) - get_d(std::get<0>(expr.args), s - D)) *
               m.inv_dual_volume(s);


    }

    //! *Form<IR> => Form<N-IL>


    template<typename T, size_t I>
    inline traits::value_type_t<Field<Expression<ct::HodgeStar, T> >>
    eval_(Field<Expression<ct::HodgeStar, T>> const &expr, id_type s, index_sequence<I>) const
    {
        auto const &l = std::get<0>(expr.args);

        size_t i = M::iform(s);
        id_type X = (i == VERTEX || i == VOLUME) ? M::_DI : M::delta_index(
                M::dual(s));
        id_type Y = M::rotate(X);
        id_type Z = M::inverse_rotate(X);


        return (
                       get_v(l, ((s - X) - Y) - Z) +
                       get_v(l, ((s - X) - Y) + Z) +
                       get_v(l, ((s - X) + Y) - Z) +
                       get_v(l, ((s - X) + Y) + Z) +
                       get_v(l, ((s + X) - Y) - Z) +
                       get_v(l, ((s + X) - Y) + Z) +
                       get_v(l, ((s + X) + Y) - Z) +
                       get_v(l, ((s + X) + Y) + Z)

               ) * m.inv_dual_volume(s) * 0.125;


    };


////***************************************************************************************************
//
////! map_to

    template<typename TF, size_t I>
    inline traits::value_type_t<TF>
    mapto(TF const &expr, id_type s, index_sequence<I, I>) const
    {
        return eval_(expr, s);
    }


    template<typename TF>
    inline traits::value_type_t<TF>
    mapto(TF const &expr, id_type s, index_sequence<VERTEX, EDGE>) const
    {

        id_type X = s & M::_DA;

        s = (s | m.FULL_OVERFLOW_FLAG) - M::_DA;

        return (eval_(expr, s + M::_DA - X) + eval_(expr, s + M::_DA + X)) *
               0.5;
    }


    template<typename TV, typename ...Others>
    inline TV
    mapto(Field<nTuple<TV, 3>, Others...> const &expr, id_type s, index_sequence<VERTEX, EDGE>) const
    {
        size_t n = M::sub_index(s);
        id_type X = s & M::_DA;
        s = (s | m.FULL_OVERFLOW_FLAG) - M::_DA;
        return (eval_(expr, s + M::_DA - X)[n] +
                eval_(expr, s + M::_DA + X)[n]) * 0.5;

    }


    template<typename TF>
    inline traits::value_type_t<TF>
    mapto(TF const &expr, id_type s, index_sequence<VERTEX, FACE>) const
    {

        auto const &l = expr;
        auto X = M::delta_index(M::dual(s));
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);
        s = (s | m.FULL_OVERFLOW_FLAG) - M::_DA;

        return (
                eval_(l, (s + M::_DA - Y - Z)) +
                eval_(l, (s + M::_DA - Y + Z)) +
                eval_(l, (s + M::_DA + Y - Z)) +
                eval_(l, (s + M::_DA + Y + Z))
        );
    }


    template<typename TV, typename ...Others>
    constexpr inline
    TV
    mapto(Field<nTuple<TV, 3>, Others...> const &expr, id_type s, index_sequence<VERTEX, FACE>) const
    {

        size_t n = M::sub_index(s);
        auto const &l = expr;
        auto X = M::delta_index(M::dual(s));
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);
        s = (s | m.FULL_OVERFLOW_FLAG) - M::_DA;

        return (
                eval_(l, (s + M::_DA - Y - Z))[n] +
                eval_(l, (s + M::_DA - Y + Z))[n] +
                eval_(l, (s + M::_DA + Y - Z))[n] +
                eval_(l, (s + M::_DA + Y + Z))[n]
        );
    }

    template<typename TF>
    constexpr inline
    traits::value_type_t<TF>
    mapto(TF const &expr, id_type s, index_sequence<VERTEX, VOLUME>) const
    {
        auto const &l = expr;

        auto X = m.DI(0, s);
        auto Y = m.DI(1, s);
        auto Z = m.DI(2, s);
        s = (s | m.FULL_OVERFLOW_FLAG) - M::_DA;

        return (eval_(l, s + M::_DA - X - Y - Z) +
                eval_(l, s + M::_DA - X - Y + Z) +
                eval_(l, s + M::_DA - X + Y - Z) +
                eval_(l, s + M::_DA - X + Y + Z) +
                eval_(l, s + M::_DA + X - Y - Z) +
                eval_(l, s + M::_DA + X - Y + Z) +
                eval_(l, s + M::_DA + X + Y - Z) +
                eval_(l, s + M::_DA + X + Y + Z)

        );
    }


    template<typename TF>
    const inline
    nTuple<typename traits::value_type<TF>::type, 3>
    mapto(TF const &expr, id_type s, index_sequence<EDGE, VERTEX>) const
    {
        typedef nTuple<typename traits::value_type<TF>::type, 3> field_value_type;

        auto const &l = expr;

        id_type DA = M::_DA;
        id_type X = M::_D;
        id_type Y = X << M::ID_DIGITS;
        id_type Z = Y << M::ID_DIGITS;

        s = (s | m.FULL_OVERFLOW_FLAG) - M::_DA;

        return nTuple<typename traits::value_type<TF>::type, 3>
                {
                        static_cast<typename traits::value_type<TF>::type>((eval_(l, s + M::_DA - X) +
                                                                            eval_(l, s + M::_DA + X)) * 0.5),
                        static_cast<typename traits::value_type<TF>::type>((eval_(l, s + M::_DA - Y) +
                                                                            eval_(l, s + M::_DA + Y)) * 0.5),
                        static_cast<typename traits::value_type<TF>::type>((eval_(l, s + M::_DA - Z) +
                                                                            eval_(l, s + M::_DA + Z)) * 0.5)

                };


    }


    template<typename TF>
    constexpr inline
    nTuple<typename traits::value_type<TF>::type, 3>
    mapto(TF const &expr, id_type s, index_sequence<FACE, VERTEX>) const
    {
        auto const &l = expr;

        id_type X = M::_D;
        id_type Y = X << M::ID_DIGITS;;
        id_type Z = Y << M::ID_DIGITS;;

        s = (s | m.FULL_OVERFLOW_FLAG) - M::_DA;

        return nTuple<typename traits::value_type<TF>::type, 3>
                {
                        static_cast<typename traits::value_type<TF>::type>(eval_(l, (s + M::_DA - Y - Z)) +
                                                                           eval_(l, (s + M::_DA - Y + Z)) +
                                                                           eval_(l, (s + M::_DA + Y - Z)) +
                                                                           eval_(l, (s + M::_DA + Y + Z))),
                        static_cast<typename traits::value_type<TF>::type>(eval_(l, (s + M::_DA - Z - X)) +
                                                                           eval_(l, (s + M::_DA - Z + X)) +
                                                                           eval_(l, (s + M::_DA + Z - X)) +
                                                                           eval_(l, (s + M::_DA + Z + X))),
                        static_cast<typename traits::value_type<TF>::type>(eval_(l, (s + M::_DA - X - Y)) +
                                                                           eval_(l, (s + M::_DA - X + Y)) +
                                                                           eval_(l, (s + M::_DA + X - Y)) +
                                                                           eval_(l, (s + M::_DA + X + Y)))
                };


    }


    template<typename TF>
    constexpr inline
    typename traits::value_type<TF>::type
    mapto(TF const &expr, id_type s, index_sequence<VOLUME, VERTEX>) const
    {
        auto const &l = expr;

        auto X = m.DI(0, s);
        auto Y = m.DI(1, s);
        auto Z = m.DI(2, s);
        s = (s | M::FULL_OVERFLOW_FLAG) - M::_DA;

        return (
                       eval_(l, ((s + M::_DA - X - Y - Z))) +
                       eval_(l, ((s + M::_DA - X - Y + Z))) +
                       eval_(l, ((s + M::_DA - X + Y - Z))) +
                       eval_(l, ((s + M::_DA - X + Y + Z))) +
                       eval_(l, ((s + M::_DA + X - Y - Z))) +
                       eval_(l, ((s + M::_DA + X - Y + Z))) +
                       eval_(l, ((s + M::_DA + X + Y - Z))) +
                       eval_(l, ((s + M::_DA + X + Y + Z)))

               ) * 0.125;
    }


    template<typename TF>
    constexpr inline
    typename traits::value_type<TF>::type
    mapto(TF const &expr, id_type s, index_sequence<VOLUME, FACE>) const
    {
        auto X = M::delta_index(M::dual(s));
        s = (s | M::FULL_OVERFLOW_FLAG) - M::_DA;

        return (eval_(expr, s - X) + eval_(expr, s + X)) * 0.5;
    }


    template<typename TF>
    constexpr inline
    typename traits::value_type<TF>::type
    mapto(TF const &expr, id_type s, index_sequence<VOLUME, EDGE>) const
    {
        auto const &l = expr;
        auto X = M::delta_index(s);
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);
        s = (s | M::FULL_OVERFLOW_FLAG) - M::_DA;

        return (
                eval_(l, s + M::_DA - Y - Z) +
                eval_(l, s + M::_DA - Y + Z) +
                eval_(l, s + M::_DA + Y - Z) +
                eval_(l, s + M::_DA + Y + Z)
        );
    }


    template<typename TF>
    constexpr inline
    nTuple<typename traits::value_type<TF>::type, 3>
    mapto(TF const &expr, id_type s, index_sequence<FACE, VOLUME>) const
    {
        auto const &l = expr;

        auto X = m.DI(0, M::dual(s));
        auto Y = m.DI(1, M::dual(s));
        auto Z = m.DI(2, M::dual(s));
        s = (s | M::FULL_OVERFLOW_FLAG) - M::_DA;

        return nTuple<traits::value_type_t<TF>, 3>
                {
                        (eval_(l, s + M::_DA - X) +
                         eval_(l, s + M::_DA + X)) * 0.5,
                        (eval_(l, s + M::_DA - Y) +
                         eval_(l, s + M::_DA + Y)) * 0.5,
                        (eval_(l, s + M::_DA - Z) +
                         eval_(l, s + M::_DA + Z)) * 0.5

                };


    }


    template<typename TF>
    constexpr inline
    nTuple<typename traits::value_type<TF>::type, 3>
    mapto(TF const &expr, id_type s,
          index_sequence<EDGE, VOLUME>) const
    {
        auto const &l = expr;

        auto X = m.DI(0, s);
        auto Y = m.DI(1, s);
        auto Z = m.DI(2, s);
        s = (s | M::FULL_OVERFLOW_FLAG) - M::_DA;

        return nTuple<typename traits::value_type<TF>::type, 3>
                {
                        (eval_(l, s + M::_DA - Y - Z) +
                         eval_(l, s + M::_DA - Y + Z) +
                         eval_(l, s + M::_DA + Y - Z) +
                         eval_(l, s + M::_DA + Y + Z)),
                        (eval_(l, s + M::_DA - Z - X) +
                         eval_(l, s + M::_DA - Z + X) +
                         eval_(l, s + M::_DA + Z - X) +
                         eval_(l, s + M::_DA + Z + X)),
                        (eval_(l, s + M::_DA - X - Y) +
                         eval_(l, s + M::_DA - X + Y) +
                         eval_(l, s + M::_DA + X - Y) +
                         eval_(l, s + M::_DA + X + Y))
                };


    }


    //***************************************************************************************************
    //
    //! Form<IL> ^ Form<IR> => Form<IR+IL>
    template<typename ...T, size_t IL, size_t IR>
    constexpr inline
    traits::value_type_t<Field<Expression<ct::Wedge, T...>>>
    eval_(Field<Expression<ct::Wedge, T...>> const &expr, id_type s, index_sequence<IL, IR>) const
    {
        return m.inner_product(mapto(std::get<0>(expr.args), s, index_sequence<IL, IR + IL>()),
                               mapto(std::get<1>(expr.args), s, index_sequence<IR, IR + IL>()),
                               s);

    }


    template<typename TL, typename TR>
    constexpr inline
    traits::value_type_t<Field<Expression<ct::Wedge, TL, TR>>>
    eval_(Field<Expression<ct::Wedge, TL, TR>> const &expr, id_type s, index_sequence<EDGE, EDGE>) const
    {
        auto const &l = std::get<0>(expr.args);
        auto const &r = std::get<1>(expr.args);

        auto Y = M::delta_index(M::rotate(M::dual(s)));
        auto Z = M::delta_index(
                M::inverse_rotate(M::dual(s)));

        return ((eval_(l, s - Y) + eval_(l, s + Y))
                * (eval_(l, s - Z) + eval_(l, s + Z)) * 0.25);
    }


    template<typename TL, typename TR, size_t I>
    constexpr inline
    traits::value_type_t<Field<Expression<ct::Cross, TL, TR>>>
    eval_(Field<Expression<ct::Cross, TL, TR>> const &expr, id_type s, index_sequence<I, I>) const
    {
        return cross(eval_(std::get<0>(expr.args), s), eval_(std::get<1>(expr.args), s));
    }

    template<typename TL, typename TR, size_t I>
    constexpr inline
    traits::value_type_t<Field<Expression<ct::Dot, TL, TR>>>
    eval_(Field<Expression<ct::Dot, TL, TR>> const &expr, id_type s, index_sequence<I, I>) const
    {
        return dot(eval_(std::get<0>(expr.args), s), eval_(std::get<1>(expr.args), s));
    }


    template<typename ...T, size_t ...I>
    constexpr inline
    traits::value_type_t<Field<Expression<ct::MapTo, T...> >>
    eval_(Field<Expression<ct::MapTo, T...>> const &expr, id_type s, index_sequence<I...>) const
    {
        return mapto(std::get<0>(expr.args), s, index_sequence<I...>());
    };

    template<typename TOP, typename ...T, size_t ... I>
    inline constexpr
    typename traits::value_type<Field<Expression<TOP, T...> > >::type
    _invoke_helper(Field<Expression<TOP, T...> > const &expr, id_type s, index_sequence<I...>) const
    {
        return expr.m_op_(eval_(std::get<I>(expr.args), s)...);
    };


    template<typename TOP, typename ... T, size_t ...I>
    inline constexpr traits::value_type_t<Field<Expression<TOP, T...> >>
    eval_(Field<Expression<TOP, T...> > const &expr, id_type const &s, index_sequence<I...>) const
    {
        return _invoke_helper(expr, s, index_sequence_for<T...>());
    }


};// struct DiffScheme<TGeo, diff_scheme::tags::finite_volume>



}}}// namespace simpla

#endif /* FDM_H_ */
