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

#include "calculate.h"

#include "../../gtl/macro.h"
#include "../../gtl/primitives.h"
#include "../../gtl/type_traits.h"
#include "../../manifold/manifold_traits.h"

namespace simpla {

template<typename ...>
class Field;

/** @ingroup diff_scheme
 *  @brief   FdMesh
 */



#define DECLARE_FUNCTION_SUFFIX const
#define DECLARE_FUNCTION_PREFIX inline
namespace ct = calculate::tags;

template<typename TGeo>
struct Calculate<TGeo, ct::finite_volume>
{
private:

    typedef TGeo geometry_type;

    typedef Calculate<geometry_type, ct::finite_volume> this_type;

    typedef typename geometry_type::id_type id_t;

    geometry_type &m_geo_;

public:
    template<typename TDict>
    void load(TDict const &) { }

    template<typename OS>
    OS &print(OS &os) const
    {
        os << "\t Calculate = { Type = \"Finite Volume\" }," << std::endl;

        return os;
    }

    void deploy() { }

private:
    ///***************************************************************************************************
    /// @name general_algebra General algebra
    /// @{
    ///***************************************************************************************************

    constexpr Real eval(Real v, id_t s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    constexpr int eval(int v, id_t s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    constexpr std::complex<Real> eval(std::complex<Real> v, id_t s) DECLARE_FUNCTION_SUFFIX
    {
        return v;
    }

    template<typename T, size_t ...N>
    constexpr nTuple<T, N...> const &eval(nTuple<T, N...> const &v, id_t s) DECLARE_FUNCTION_SUFFIX
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

    template<typename TM, typename TV, typename ... Others>
    constexpr TV eval(Field<TM, TV, Others...> const &f, id_t s) DECLARE_FUNCTION_SUFFIX
    {
        return traits::index(f, s);
    }

    template<typename TOP, typename ... T>
    constexpr traits::primary_type_t<
            traits::value_type_t<Field<Expression<TOP, T...> >>>
    eval(Field<Expression<TOP, T...> > const &expr, id_t const &s) DECLARE_FUNCTION_SUFFIX
    {
        return eval(expr, s, traits::iform_list_t<T...>());
    }


    template<typename Expr, size_t ... index>
    traits::primary_type_t<traits::value_type_t<Expr>> _invoke_helper(
            Expr const &expr, id_t s, index_sequence<index...>) DECLARE_FUNCTION_SUFFIX
    {
        traits::primary_type_t<traits::value_type_t<Expr>> res = (expr.m_op_(
                eval(std::get<index>(expr.args), s)...));

        return std::move(res);
    }


    template<typename TOP, typename ... T>
    constexpr traits::primary_type_t<traits::value_type_t<Field<Expression<TOP, T...> >>>
    eval(Field<Expression<TOP, T...> > const &expr, id_t const &s,
         traits::iform_list_t<T...>) DECLARE_FUNCTION_SUFFIX
    {
        return _invoke_helper(expr, s, typename make_index_sequence<sizeof...(T)>::type());
    }



    //***************************************************************************************************
    // Exterior algebra
    //***************************************************************************************************

    template<typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<
            Field<Expression<ct::ExteriorDerivative, T>>>
    eval(Field<Expression<ct::ExteriorDerivative, T> > const &f,
         id_t s, integer_sequence<int, VERTEX>) DECLARE_FUNCTION_SUFFIX
    {
        id_t D = geometry_type::delta_index(s);
        return (eval(std::get<0>(f.args), s + D) * m_geo_.volume(s + D)
                - eval(std::get<0>(f.args), s - D) * m_geo_.volume(s - D))
               * m_geo_.inv_volume(s);
    }

    template<typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<
            Field<Expression<ct::ExteriorDerivative, T>>>
    eval(Field<Expression<ct::ExteriorDerivative, T> > const &expr,
         id_t s, integer_sequence<int, EDGE>) DECLARE_FUNCTION_SUFFIX
    {

        id_t X = geometry_type::delta_index(geometry_type::dual(s));
        id_t Y = geometry_type::rotate(X);
        id_t Z = geometry_type::inverse_rotate(X);

        return ((eval(std::get<0>(expr.args), s + Y) * m_geo_.volume(s + Y) //
                 - eval(std::get<0>(expr.args), s - Y) * m_geo_.volume(s - Y))
                - (eval(std::get<0>(expr.args), s + Z) * m_geo_.volume(s + Z) //
                   - eval(std::get<0>(expr.args), s - Z) * m_geo_.volume(s - Z) //
                )

               ) * m_geo_.inv_volume(s);

    }

    template<typename T>
    constexpr DECLARE_FUNCTION_PREFIX traits::value_type_t<
            Field<Expression<ct::ExteriorDerivative, T>>>
    eval(Field<Expression<ct::ExteriorDerivative, T> > const &expr,
         id_t s, integer_sequence<int, FACE>) DECLARE_FUNCTION_SUFFIX
    {
        return (eval(std::get<0>(expr.args), s + geometry_type::_DI)
                * m_geo_.volume(s + geometry_type::_DI)
                - eval(std::get<0>(expr.args), s - geometry_type::_DI)
                  * m_geo_.volume(s - geometry_type::_DI)
                + eval(std::get<0>(expr.args), s + geometry_type::_DJ)
                  * m_geo_.volume(s + geometry_type::_DJ)
                - eval(std::get<0>(expr.args), s - geometry_type::_DJ)
                  * m_geo_.volume(s - geometry_type::_DJ)
                + eval(std::get<0>(expr.args), s + geometry_type::_DK)
                  * m_geo_.volume(s + geometry_type::_DK)
                - eval(std::get<0>(expr.args), s - geometry_type::_DK)
                  * m_geo_.volume(s - geometry_type::_DK)

               ) * m_geo_.inv_volume(s);
    }
//
////	template<typename geometry_type,typename TM, int IL, typename TL> void eval(
////			ct::ExteriorDerivative, Field<Domain<TM, IL>, TL> const & f,
////					typename geometry_type::id_type   s)  = delete;
////
////	template<typename geometry_type,typename TM, int IL, typename TL> void eval(
////			ct::CodifferentialDerivative,
////			Field<TL...> const & f, 		typename geometry_type::id_type   s)  = delete;

    template<typename T>
    constexpr DECLARE_FUNCTION_PREFIX traits::value_type_t<
            Field<Expression<ct::CodifferentialDerivative, T>>>
    eval(Field<Expression<ct::CodifferentialDerivative, T>> const &expr,
         id_t s, integer_sequence<int, EDGE>) DECLARE_FUNCTION_SUFFIX
    {
        return -(eval(std::get<0>(expr.args), s + geometry_type::_DI)
                 * m_geo_.dual_volume(s + geometry_type::_DI)
                 - eval(std::get<0>(expr.args), s - geometry_type::_DI)
                   * m_geo_.dual_volume(s - geometry_type::_DI)
                 + eval(std::get<0>(expr.args), s + geometry_type::_DJ)
                   * m_geo_.dual_volume(s + geometry_type::_DJ)
                 - eval(std::get<0>(expr.args), s - geometry_type::_DJ)
                   * m_geo_.dual_volume(s - geometry_type::_DJ)
                 + eval(std::get<0>(expr.args), s + geometry_type::_DK)
                   * m_geo_.dual_volume(s + geometry_type::_DK)
                 - eval(std::get<0>(expr.args), s - geometry_type::_DK)
                   * m_geo_.dual_volume(s - geometry_type::_DK)

        ) * m_geo_.inv_dual_volume(s);

    }

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

                -((eval(std::get<0>(expr.args), s + Y) * (m_geo_.dual_volume(s + Y))
                   - eval(std::get<0>(expr.args), s - Y)
                     * (m_geo_.dual_volume(s - Y)))

                  - (eval(std::get<0>(expr.args), s + Z)
                     * (m_geo_.dual_volume(s + Z))
                     - eval(std::get<0>(expr.args), s - Z)
                       * (m_geo_.dual_volume(s - Z)))

                ) * m_geo_.inv_dual_volume(s);
    }

    template<typename T>
    DECLARE_FUNCTION_PREFIX traits::value_type_t<
            Field<Expression<ct::CodifferentialDerivative, T>>>
    eval(Field<Expression<ct::CodifferentialDerivative, T> > const &expr,
         id_t s, integer_sequence<int, VOLUME>) DECLARE_FUNCTION_SUFFIX
    {
        id_t D = geometry_type::delta_index(geometry_type::dual(s));
        return -(eval(std::get<0>(expr.args), s + D) * (m_geo_.dual_volume(s + D)) //
                 - eval(std::get<0>(expr.args), s - D) * (m_geo_.dual_volume(s - D))
        ) * m_geo_.inv_dual_volume(s);
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

                        (eval(l, (s - Y) - Z) + eval(l, (s - Y) + Z)
                         + eval(l, (s + Y) - Z) + eval(l, (s + Y) + Z))
                        * (eval(r, s - X) + eval(r, s + X))
                        +

                        (eval(l, (s - Z) - X) + eval(l, (s - Z) + X)
                         + eval(l, (s + Z) - X) + eval(l, (s + Z) + X))
                        * (eval(r, s - Y) + eval(r, s + Y))
                        +

                        (eval(l, (s - X) - Y) + eval(l, (s - X) + Y)
                         + eval(l, (s + X) - Y) + eval(l, (s + X) + Y))
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

        return eval(l, s)
               * (eval(r, (s - Y) - Z) + eval(r, (s - Y) + Z)
                  + eval(r, (s + Y) - Z) + eval(r, (s + Y) + Z))
               * 0.25;
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

    Calculate(geometry_type &geo) : m_geo_(geo)
    {
    }

    virtual ~Calculate()
    {
    }


    template<typename ...Args>
    auto calculate(Args &&...args) DECLARE_FUNCTION_SUFFIX
    DECL_RET_TYPE((this->eval(std::forward<Args>(args)...)))

//    template<typename ...D, typename T>
//    auto calculate(Field<Domain<D...>, T, tags::function> && f,id_type s) DECLARE_FUNCTION_SUFFIX
//    DECL_RET_TYPE((f(this->coordinates(s)))

};// struct Calculate<TGeo, ct::finite_volume>

#undef DECLARE_FUNCTION_PREFIX
#undef DECLARE_FUNCTION_SUFFIX
}// namespace simpla

#endif /* FDM_H_ */
