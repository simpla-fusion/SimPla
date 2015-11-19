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

#include "diff_scheme.h"

#include "../../gtl/macro.h"
#include "../../gtl/primitives.h"
#include "../../gtl/type_traits.h"
#include "../../manifold/manifold_traits.h"

namespace simpla
{
template<typename ...> class Field;
namespace manifold { namespace policy
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
struct DiffScheme<TGeo, diff_scheme::tags::finite_volume>
{
private:

    typedef TGeo geometry_type;

    typedef DiffScheme<geometry_type, diff_scheme::tags::finite_volume> this_type;

    typedef typename geometry_type::id_type id_t;

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

    DECLARE_FUNCTION_PREFIX constexpr Real eval(Real v, id_t s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    DECLARE_FUNCTION_PREFIX constexpr int eval(int v, id_t s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    DECLARE_FUNCTION_PREFIX constexpr std::complex<Real> eval(std::complex<Real> v, id_t s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    template<typename T, size_t ...N>
    DECLARE_FUNCTION_PREFIX constexpr nTuple<T, N...> const &eval(nTuple<T, N...> const &v,
                                                                  id_t s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    template<typename ...T>
    DECLARE_FUNCTION_PREFIX traits::primary_type_t<nTuple<Expression<T...>>>
    eval(nTuple<Expression<T...>> const &v, id_t s) DECLARE_FUNCTION_SUFFIX
    {
        traits::primary_type_t<nTuple<Expression<T...> > > res;
        res = v;
        return std::move(res);
    }

    template<typename TV, typename TM, typename ... Others>
    DECLARE_FUNCTION_PREFIX constexpr TV eval(Field<TV, TM, Others...> const &f, id_t s) DECLARE_FUNCTION_SUFFIX
    {
        return traits::index(f, s);
    }

public:
    template<typename TOP, typename ... T>
    DECLARE_FUNCTION_PREFIX constexpr traits::primary_type_t<
            traits::value_type_t<Field<Expression<TOP, T...> >>>
    eval(Field<Expression<TOP, T...> > const &expr, id_t const &s) DECLARE_FUNCTION_SUFFIX
    {
        return eval(expr, s, traits::iform_list_t<T...>());
    }

private:

    template<typename Expr, int ... index>
    DECLARE_FUNCTION_PREFIX traits::primary_type_t<traits::value_type_t<Expr>> _invoke_helper(
            Expr const &expr, id_t s, index_sequence<index...>) DECLARE_FUNCTION_SUFFIX
    {
        traits::primary_type_t<traits::value_type_t<Expr>> res = (expr.m_op_(
                eval(std::get<index>(expr.args), s)...));

        return std::move(res);
    }


    template<typename TOP, typename ... T>
    DECLARE_FUNCTION_PREFIX constexpr traits::primary_type_t<traits::value_type_t<Field<Expression<TOP, T...> >>>
    eval(Field<Expression<TOP, T...> > const &expr, id_t const &s,
         traits::iform_list_t<T...>) DECLARE_FUNCTION_SUFFIX
    {
        return _invoke_helper(expr, s, typename make_index_sequence<sizeof...(T)>::type());
    }


    template<typename FExpr>
    DECLARE_FUNCTION_PREFIX constexpr traits::value_type_t<FExpr>
    get_v(FExpr const &f, id_type const s) DECLARE_FUNCTION_SUFFIX
    {
        return eval(f, s) * m_geo_.volume(s);
    }

    template<typename FExpr>
    DECLARE_FUNCTION_PREFIX constexpr traits::value_type_t<FExpr>
    get_d(FExpr const &f, id_type const s) DECLARE_FUNCTION_SUFFIX
    {
        return eval(f, s) * m_geo_.dual_volume(s);
    }


    //***************************************************************************************************
    // Exterior algebra
    //***************************************************************************************************

    //! grad<0>
    template<typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::ExteriorDerivative, T>>>
    eval(Field<Expression<ct::ExteriorDerivative, T> > const &f,
         id_t s, integer_sequence<int, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        id_t D = geometry_type::delta_index(s);


        return (get_v(std::get<0>(f.args), s + D) - get_v(std::get<0>(f.args), s - D)) * m_geo_.inv_volume(s);

//        return (eval(std::get<0>(f.args), s + D) * m_geo_.volume(s + D)
//                - eval(std::get<0>(f.args), s - D) * m_geo_.volume(s - D))
//               * m_geo_.inv_volume(s);
    }


    //! curl<1>
    template<typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::ExteriorDerivative, T>>>
    eval(Field<Expression<ct::ExteriorDerivative, T> > const &expr,
         id_t s, integer_sequence<int, EDGE>) DECLARE_FUNCTION_SUFFIX
    {

        id_t X = geometry_type::delta_index(geometry_type::dual(s));
        id_t Y = geometry_type::rotate(X);
        id_t Z = geometry_type::inverse_rotate(X);


        return (
                       (get_v(std::get<0>(expr.args), s + Y) - get_v(std::get<0>(expr.args), s - Y))

                       - (get_v(std::get<0>(expr.args), s + Z) - get_v(std::get<0>(expr.args), s - Z))

               ) * m_geo_.inv_volume(s);


//        return ((eval(std::get<0>(expr.args), s + Y) * m_geo_.volume(s + Y) //
//                 - eval(std::get<0>(expr.args), s - Y) * m_geo_.volume(s - Y))
//                - (eval(std::get<0>(expr.args), s + Z) * m_geo_.volume(s + Z) //
//                   - eval(std::get<0>(expr.args), s - Z) * m_geo_.volume(s - Z) //
//                )
//
//               ) * m_geo_.inv_volume(s);

    }

    //! div<2>
    template<typename T>
    constexpr DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::ExteriorDerivative, T>>>
    eval(Field<Expression<ct::ExteriorDerivative, T> > const &expr,
         id_t s, integer_sequence<int, FACE>) DECLARE_FUNCTION_SUFFIX
    {

        return (get_v(std::get<0>(expr.args), s + geometry_type::_DI)

                - get_v(std::get<0>(expr.args), s - geometry_type::_DI)

                + get_v(std::get<0>(expr.args), s + geometry_type::_DJ)

                - get_v(std::get<0>(expr.args), s - geometry_type::_DJ)

                + get_v(std::get<0>(expr.args), s + geometry_type::_DK)

                - get_v(std::get<0>(expr.args), s - geometry_type::_DK)


               ) * m_geo_.inv_volume(s);

//        return (eval(std::get<0>(expr.args), s + base_manifold_type::_DI)
//                * m_geo_.volume(s + base_manifold_type::_DI)
//                - eval(std::get<0>(expr.args), s - base_manifold_type::_DI)
//                  * m_geo_.volume(s - base_manifold_type::_DI)
//                + eval(std::get<0>(expr.args), s + base_manifold_type::_DJ)
//                  * m_geo_.volume(s + base_manifold_type::_DJ)
//                - eval(std::get<0>(expr.args), s - base_manifold_type::_DJ)
//                  * m_geo_.volume(s - base_manifold_type::_DJ)
//                + eval(std::get<0>(expr.args), s + base_manifold_type::_DK)
//                  * m_geo_.volume(s + base_manifold_type::_DK)
//                - eval(std::get<0>(expr.args), s - base_manifold_type::_DK)
//                  * m_geo_.volume(s - base_manifold_type::_DK)
//
//               ) * m_geo_.inv_volume(s);
    }
//
////	template<typename base_manifold_type,typename TM, int IL, typename TL> void eval(
////			ct::ExteriorDerivative, Field<Domain<TM, IL>, TL> const & f,
////					typename base_manifold_type::id_type   s)  = delete;
////
////	template<typename base_manifold_type,typename TM, int IL, typename TL> void eval(
////			ct::CodifferentialDerivative,
////			Field<TL...> const & f, 		typename base_manifold_type::id_type   s)  = delete;

    //! div<1>
    template<typename T>
    constexpr DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::CodifferentialDerivative, T>>>
    eval(Field<Expression<ct::CodifferentialDerivative, T>> const &expr,
         id_t s, integer_sequence<int, EDGE>) DECLARE_FUNCTION_SUFFIX
    {

        return -(get_d(std::get<0>(expr.args), s + geometry_type::_DI)
                 - get_d(std::get<0>(expr.args), s - geometry_type::_DI)
                 + get_d(std::get<0>(expr.args), s + geometry_type::_DJ)
                 - get_d(std::get<0>(expr.args), s - geometry_type::_DJ)
                 + get_d(std::get<0>(expr.args), s + geometry_type::_DK)
                 - get_d(std::get<0>(expr.args), s - geometry_type::_DK)

        ) * m_geo_.inv_dual_volume(s);


//        return -(eval(std::get<0>(expr.args), s + base_manifold_type::_DI)
//                 * m_geo_.dual_volume(s + base_manifold_type::_DI)
//                 - eval(std::get<0>(expr.args), s - base_manifold_type::_DI)
//                   * m_geo_.dual_volume(s - base_manifold_type::_DI)
//                 + eval(std::get<0>(expr.args), s + base_manifold_type::_DJ)
//                   * m_geo_.dual_volume(s + base_manifold_type::_DJ)
//                 - eval(std::get<0>(expr.args), s - base_manifold_type::_DJ)
//                   * m_geo_.dual_volume(s - base_manifold_type::_DJ)
//                 + eval(std::get<0>(expr.args), s + base_manifold_type::_DK)
//                   * m_geo_.dual_volume(s + base_manifold_type::_DK)
//                 - eval(std::get<0>(expr.args), s - base_manifold_type::_DK)
//                   * m_geo_.dual_volume(s - base_manifold_type::_DK)
//
//        ) * m_geo_.inv_dual_volume(s);

    }

    //! curl<2>
    template<typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<
            Field<Expression<ct::CodifferentialDerivative, T>>>
    eval(Field<Expression<ct::CodifferentialDerivative, T>> const &expr,
         id_t s, integer_sequence<int, FACE>) DECLARE_FUNCTION_SUFFIX
    {

        id_t X = geometry_type::delta_index(s);
        id_t Y = geometry_type::rotate(X);
        id_t Z = geometry_type::inverse_rotate(X);


        return

                -((get_d(std::get<0>(expr.args), s + Y) - get_d(std::get<0>(expr.args), s - Y))
                  - (get_d(std::get<0>(expr.args), s + Z) - get_d(std::get<0>(expr.args), s - Z))
                ) * m_geo_.inv_dual_volume(s);

//        return
//
//                -(  (eval(std::get<0>(expr.args), s + Y) * (m_geo_.dual_volume(s + Y))
//                   - eval(std::get<0>(expr.args), s - Y) * (m_geo_.dual_volume(s - Y)))
//                  - (eval(std::get<0>(expr.args), s + Z) * (m_geo_.dual_volume(s + Z))
//                   - eval(std::get<0>(expr.args), s - Z) * (m_geo_.dual_volume(s - Z)))
//                 ) * m_geo_.inv_dual_volume(s);
    }


    //! grad<3>
    template<typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<
            Field<Expression<ct::CodifferentialDerivative, T>>>
    eval(Field<Expression<ct::CodifferentialDerivative, T> > const &expr,
         id_t s, integer_sequence<int, VOLUME>) DECLARE_FUNCTION_SUFFIX
    {
        id_t D = geometry_type::delta_index(geometry_type::dual(s));

        return -(get_d(std::get<0>(expr.args), s + D) - get_d(std::get<0>(expr.args), s - D)) *
               m_geo_.inv_dual_volume(s);


//        return -(  eval(std::get<0>(expr.args), s + D) * (m_geo_.dual_volume(s + D))
//
//                 - eval(std::get<0>(expr.args), s - D) * (m_geo_.dual_volume(s - D))
//        ) * m_geo_.inv_dual_volume(s);
    }

////***************************************************************************************************
//
////! Form<IR> ^ Form<IR> => Form<IR+IL>

    template<typename TL, typename TR>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Wedge, TL, TR>>>
    eval(Field<Expression<ct::Wedge, TL, TR>> const &expr,
         id_t s, integer_sequence<int, VERTEX, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        return (eval(std::get<0>(expr.args), s)
                * eval(std::get<1>(expr.args), s));
    }

    template<typename TL, typename TR>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Wedge, TL, TR>>>
    eval(Field<Expression<ct::Wedge, TL, TR>> const &expr,
         id_t s, integer_sequence<int, VERTEX, EDGE>) DECLARE_FUNCTION_SUFFIX
    {
        auto X = geometry_type::delta_index(s);

        return (eval(std::get<0>(expr.args), s - X)
                + eval(std::get<0>(expr.args), s + X)) * 0.5
               * eval(std::get<1>(expr.args), s);
    }

    template<typename TL, typename TR>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Wedge, TL, TR>>>
    eval(Field<Expression<ct::Wedge, TL, TR>> const &expr,
         id_t s, integer_sequence<int, VERTEX, FACE>) DECLARE_FUNCTION_SUFFIX
    {
        auto X = geometry_type::delta_index(geometry_type::dual(s));
        auto Y = geometry_type::rotate(X);
        auto Z = geometry_type::inverse_rotate(X);

        return (eval(std::get<0>(expr.args), (s - Y) - Z)
                + eval(std::get<0>(expr.args), (s - Y) + Z)
                + eval(std::get<0>(expr.args), (s + Y) - Z)
                + eval(std::get<0>(expr.args), (s + Y) + Z)
               ) * 0.25 * eval(std::get<1>(expr.args), s);
    }

    template<typename TL, typename TR>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Wedge, TL, TR>>>
    eval(Field<Expression<ct::Wedge, TL, TR>> const &expr,
         id_t s, integer_sequence<int, VERTEX, VOLUME>) DECLARE_FUNCTION_SUFFIX
    {

        auto const &l = std::get<0>(expr.args);
        auto const &r = std::get<1>(expr.args);

        auto X = m_geo_.DI(0, s);
        auto Y = m_geo_.DI(1, s);
        auto Z = m_geo_.DI(2, s);

        return (eval(l, ((s - X) - Y) - Z) +

                eval(l, ((s - X) - Y) + Z) +

                eval(l, ((s - X) + Y) - Z) +

                eval(l, ((s - X) + Y) + Z) +

                eval(l, ((s + X) - Y) - Z) +

                eval(l, ((s + X) - Y) + Z) +

                eval(l, ((s + X) + Y) - Z) +

                eval(l, ((s + X) + Y) + Z)

               ) * 0.125 * eval(r, s);
    }

    template<typename TL, typename TR>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Wedge, TL, TR>>>
    eval(Field<Expression<ct::Wedge, TL, TR>> const &expr,
         id_t s, integer_sequence<int, EDGE, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {

        auto const &l = std::get<0>(expr.args);
        auto const &r = std::get<1>(expr.args);

        auto X = geometry_type::delta_index(s);
        return eval(l, s) * (eval(r, s - X) + eval(r, s + X)) * 0.5;
    }

    template<typename TL, typename TR>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Wedge, TL, TR>>>
    eval(Field<Expression<ct::Wedge, TL, TR>> const &expr,
         id_t s, integer_sequence<int, EDGE, EDGE>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = std::get<0>(expr.args);
        auto const &r = std::get<1>(expr.args);

        auto Y = geometry_type::delta_index(geometry_type::rotate(geometry_type::dual(s)));
        auto Z = geometry_type::delta_index(
                geometry_type::inverse_rotate(geometry_type::dual(s)));

        return ((eval(l, s - Y) + eval(l, s + Y))
                * (eval(l, s - Z) + eval(l, s + Z)) * 0.25);
    }

    template<typename TL, typename TR>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Wedge, TL, TR>>>
    eval(Field<Expression<ct::Wedge, TL, TR>> const &expr,
         id_t s, integer_sequence<int, EDGE, FACE>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = std::get<0>(expr.args);
        auto const &r = std::get<1>(expr.args);
        auto X = m_geo_.DI(0, s);
        auto Y = m_geo_.DI(1, s);
        auto Z = m_geo_.DI(2, s);

        return

                (

                        (eval(l, (s - Y) - Z) + eval(l, (s - Y) + Z) + eval(l, (s + Y) - Z) + eval(l, (s + Y) + Z))
                        * (eval(r, s - X) + eval(r, s + X))
                        +

                        (eval(l, (s - Z) - X) + eval(l, (s - Z) + X) + eval(l, (s + Z) - X) + eval(l, (s + Z) + X))
                        * (eval(r, s - Y) + eval(r, s + Y))
                        +

                        (eval(l, (s - X) - Y) + eval(l, (s - X) + Y) + eval(l, (s + X) - Y) + eval(l, (s + X) + Y))
                        * (eval(r, s - Z) + eval(r, s + Z))

                ) * 0.125;
    }

    template<typename TL, typename TR>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Wedge, TL, TR>>>
    eval(Field<Expression<ct::Wedge, TL, TR> > const &expr,
         id_t s, integer_sequence<int, FACE, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = std::get<0>(expr.args);
        auto const &r = std::get<1>(expr.args);
        auto Y = geometry_type::delta_index(geometry_type::rotate(geometry_type::dual(s)));
        auto Z = geometry_type::delta_index(
                geometry_type::inverse_rotate(geometry_type::dual(s)));

        return eval(l, s) *
               (eval(r, (s - Y) - Z) + eval(r, (s - Y) + Z) + eval(r, (s + Y) - Z) + eval(r, (s + Y) + Z)) * 0.25;
    }

    template<typename TL, typename TR>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Wedge, TL, TR>>>
    eval(Field<Expression<ct::Wedge, TL, TR> > const &expr,
         id_t s, integer_sequence<int, FACE, EDGE>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = std::get<0>(expr.args);
        auto const &r = std::get<1>(expr.args);
        auto X = m_geo_.DI(0, s);
        auto Y = m_geo_.DI(1, s);
        auto Z = m_geo_.DI(2, s);

        return ((eval(r, (s - Y) - Z) + eval(r, (s - Y) + Z)
                 + eval(r, (s + Y) - Z) + eval(r, (s + Y) + Z))
                * (eval(l, s - X) + eval(l, s + X))

                + (eval(r, (s - Z) - X) + eval(r, (s - Z) + X)
                   + eval(r, (s + Z) - X) + eval(r, (s + Z) + X))
                  * (eval(l, s - Y) + eval(l, s + Y))

                + (eval(r, (s - X) - Y) + eval(r, (s - X) + Y)
                   + eval(r, (s + X) - Y) + eval(r, (s + X) + Y))
                  * (eval(l, s - Z) + eval(l, s + Z))

               ) * 0.125;
    }

    template<typename TL, typename TR>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<Field<Expression<ct::Wedge, TL, TR>>>
    eval(Field<Expression<ct::Wedge, TL, TR>> const &expr,
         id_t s, integer_sequence<int, VOLUME, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        auto const &l = std::get<0>(expr.args);
        auto const &r = std::get<1>(expr.args);
        auto X = m_geo_.DI(0, s);
        auto Y = m_geo_.DI(1, s);
        auto Z = m_geo_.DI(2, s);

        return

                eval(l, s) * (eval(r, ((s - X) - Y) - Z) + //
                              eval(r, ((s - X) - Y) + Z) + //
                              eval(r, ((s - X) + Y) - Z) + //
                              eval(r, ((s - X) + Y) + Z) + //
                              eval(r, ((s + X) - Y) - Z) + //
                              eval(r, ((s + X) - Y) + Z) + //
                              eval(r, ((s + X) + Y) - Z) + //
                              eval(r, ((s + X) + Y) + Z) //

                ) * 0.125;
    }

public:

    DiffScheme(geometry_type &geo) : m_geo_(geo)
    {
    }

    virtual ~DiffScheme()
    {
    }


};// struct DiffScheme<TGeo, diff_scheme::tags::finite_volume>

#undef DECLARE_FUNCTION_PREFIX
#undef DECLARE_FUNCTION_SUFFIX

} //namespace policy
} //namespace manifold

}// namespace simpla

#endif /* FDM_H_ */
