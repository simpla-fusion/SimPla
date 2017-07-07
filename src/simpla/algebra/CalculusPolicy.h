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

#include <simpla/engine/SPObject.h>
#include <simpla/utilities/Array.h>
#include <simpla/utilities/EntityId.h>
#include <simpla/utilities/ExpressionTemplate.h>
#include <simpla/utilities/FancyStream.h>
#include <simpla/utilities/macro.h>
#include <simpla/utilities/type_traits.h>
#include "Calculus.h"
#include "Field.h"

namespace simpla {
template <typename TM, typename TV, int...>
class Field;

template <typename M>
struct CalculusPolicy {
    typedef M mesh_type;
    typedef CalculusPolicy<mesh_type> this_type;

    template <size_t... I, typename TOP, typename... Args, typename... Others>
    static auto _invoke_helper(std::index_sequence<I...>, mesh_type const& m, Expression<TOP, Args...> const& expr,
                               IdxShift S, Others&&... others) {
        return expr.m_op_(getValue(m, std::get<I>(expr.m_args_), S, std::forward<Others>(others)...)...);
    }

    template <int... I, typename TOP, typename... Args, typename... Others>
    static auto eval(std::integer_sequence<int, I...>, mesh_type const& m, Expression<TOP, Args...> const& expr,
                     IdxShift S, Others&&... others) {
        return _invoke_helper(std::make_index_sequence<sizeof...(I)>(), m, expr, S, std::forward<Others>(others)...);
    }

    template <typename TOP, typename... Args, typename... Others>
    static auto getValue(mesh_type const& m, Expression<TOP, Args...> const& expr, IdxShift S, Others&&... others) {
        return eval(std::integer_sequence<int, traits::iform<Args>::value...>(), m, expr, S,
                    std::forward<Others>(others)...);
    }

    template <typename T, typename... Args>
    static auto getValue(mesh_type const& m, T const& v, IdxShift S, Args&&... args) {
        return calculus::getValue(v, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static auto _getV(std::integral_constant<int, VERTEX>, mesh_type const& m, Args&&... args) {
        return getValue(m, m.m_vertex_volume_, std::forward<Args>(args)...);
    }
    template <typename... Args>
    static auto _getV(std::integral_constant<int, EDGE>, mesh_type const& m, Args&&... args) {
        return getValue(m, m.m_edge_volume_, std::forward<Args>(args)...);
    }
    template <typename... Args>
    static auto _getV(std::integral_constant<int, FACE>, mesh_type const& m, Args&&... args) {
        return getValue(m.m_face_volume_, m, std::forward<Args>(args)...);
    }
    template <typename... Args>
    static auto _getV(std::integral_constant<int, VOLUME>, mesh_type const& m, Args&&... args) {
        return getValue(m, m.m_volume_volume_, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static auto _getDualV(std::integral_constant<int, VERTEX>, mesh_type const& m, Args&&... args) {
        return getValue(m, m.m_vertex_dual_volume_, std::forward<Args>(args)...);
    }
    template <typename... Args>
    static auto _getDualV(std::integral_constant<int, EDGE>, mesh_type const& m, Args&&... args) {
        return getValue(m, m.m_edge_dual_volume_, std::forward<Args>(args)...);
    }
    template <typename... Args>
    static auto _getDualV(std::integral_constant<int, FACE>, mesh_type const& m, Args&&... args) {
        return getValue(m, m.m_face_dual_volume_, std::forward<Args>(args)...);
    }
    template <typename... Args>
    static auto _getDualV(std::integral_constant<int, VOLUME>, mesh_type const& m, Args&&... args) {
        return getValue(m, m.m_volume_dual_volume_, std::forward<Args>(args)...);
    }

    template <typename TExpr, typename... Args>
    static auto getDualV(mesh_type const& m, TExpr const& expr, IdxShift S, int n, Args&&... args) {
        return getValue(m, expr, S, n, std::forward<Args>(args)...) *
               _getDualV(std::integral_constant<int, traits::iform<TExpr>::value>(), m, S, n);
    }
    template <typename TExpr, typename... Args>
    static auto getV(mesh_type const& m, TExpr const& expr, IdxShift S, int n, Args&&... args) {
        return getValue(m, expr, S, n, std::forward<Args>(args)...) *
               _getV(std::integral_constant<int, traits::iform<TExpr>::value>(), m, S, n);
    }

    //******************************************************************************
    // Exterior algebra
    //******************************************************************************

    //! grad<0>

    template <typename TExpr, typename... Others>
    static auto eval(std::integer_sequence<int, VERTEX>, mesh_type const& m,
                     Expression<tags::_exterior_derivative, TExpr> const& expr, IdxShift S, int n, Others&&... others) {
        auto const& l = std::get<0>(expr.m_args_);
        IdxShift D{0, 0, 0};
        D[n] = 1;
        return (getV(m, l, S + D, n, std::forward<Others>(others)...) -
                getV(m, l, S, n, std::forward<Others>(others)...)) *
               getValue(m, m.m_edge_inv_volume_, S, n);
    }

    //! curl<1>

    template <typename TExpr, typename... Others>
    static auto eval(std::integer_sequence<int, EDGE>, mesh_type const& m,
                     Expression<tags::_exterior_derivative, TExpr> const& expr, IdxShift S, int n, Others&&... others) {
        auto const& l = std::get<0>(expr.m_args_);

        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};

        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return ((getV(m, l, S + Z, (n + 1) % 3, std::forward<Others>(others)...) -
                 getV(m, l, S, (n + 1) % 3, std::forward<Others>(others)...)) -
                (getV(m, l, S + Y, (n + 2) % 3, std::forward<Others>(others)...) -
                 getV(m, l, S, (n + 2) % 3, std::forward<Others>(others)...))) *
               getValue(m, m.m_face_inv_volume_, S, n, std::forward<Others>(others)...);
    }

    //! div<2>
    template <typename TExpr, typename... Others>
    static auto eval(std::integer_sequence<int, FACE>, mesh_type const& m,
                     Expression<tags::_exterior_derivative, TExpr> const& expr, IdxShift S, int n, Others&&... others) {
        auto const& l = std::get<0>(expr.m_args_);

        IdxShift X{1, 0, 0};
        IdxShift Y{0, 1, 0};
        IdxShift Z{0, 0, 1};

        return ((getV(m, l, S + X, 0, std::forward<Others>(others)...) -
                 getV(m, l, S, 0, std::forward<Others>(others)...)) +
                (getV(m, l, S + Y, 1, std::forward<Others>(others)...) -
                 getV(m, l, S, 1, std::forward<Others>(others)...)) +
                (getV(m, l, S + Z, 2, std::forward<Others>(others)...) -
                 getV(m, l, S, 2, std::forward<Others>(others)...))) *
               getValue(m.m_volume_inv_volume_, m, S, 0, std::forward<Others>(others)...);
    }

    //! curl<2>
    template <typename TExpr, typename... Others>
    static auto eval(std::integer_sequence<int, FACE>, mesh_type const& m,
                     Expression<tags::_codifferential_derivative, TExpr> const& expr, IdxShift S, int n,
                     Others&&... others) {
        auto const& l = std::get<0>(expr.m_args_);
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};

        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return ((getDualV(m, l, S, (n + 1) % 3, std::forward<Others>(others)...) -
                 getDualV(m, l, S - Z, (n + 1) % 3, std::forward<Others>(others)...)) -
                (getDualV(m, l, S, (n + 2) % 3, std::forward<Others>(others)...) -
                 getDualV(m, l, S - Y, (n + 2) % 3, std::forward<Others>(others)...))) *
               (-getValue(m, m.m_edge_inv_dual_volume_, S, n, std::forward<Others>(others)...));
    }

    //! div<1>

    template <typename TExpr, typename... Others>
    static auto eval(std::integer_sequence<int, EDGE>, mesh_type const& m,
                     Expression<tags::_codifferential_derivative, TExpr> const& expr, IdxShift S, int n,
                     Others&&... others) {
        auto const& l = std::get<0>(expr.m_args_);

        IdxShift X{1, 0, 0};
        IdxShift Y{0, 1, 0};
        IdxShift Z{0, 0, 1};

        return ((getDualV(m, l, S, 0, std::forward<Others>(others)...) -
                 getDualV(m, l, S - X, 0, std::forward<Others>(others)...)) +  //
                (getDualV(m, l, S, 1, std::forward<Others>(others)...) -
                 getDualV(m, l, S - Y, 1, std::forward<Others>(others)...)) +  //
                (getDualV(m, l, S, 2, std::forward<Others>(others)...) -
                 getDualV(m, l, S - Z, 2, std::forward<Others>(others)...))) *
               (-getValue(m, m.m_vertex_inv_dual_volume_, S, 0, std::forward<Others>(others)...));

        ;
    }

    //! grad<3>

    template <typename TExpr, typename... Others>
    static auto eval(std::integer_sequence<int, VOLUME>, mesh_type const& m,
                     Expression<tags::_codifferential_derivative, TExpr> const& expr, IdxShift S, int n,
                     Others&&... others) {
        auto const& l = std::get<0>(expr.m_args_);
        IdxShift D{0, 0, 0};
        D[n] = 1;

        return (getV(m, l, S, 0, std::forward<Others>(others)...) -
                getV(m, l, S - D, 0, std::forward<Others>(others)...)) *
               (-getValue(m, m.m_face_inv_volume_, S, n, std::forward<Others>(others)...));
    }

    //! *Form<IR> => Form<N-IL>

    template <typename TExpr, typename... Others>
    static auto eval(std::integer_sequence<int, VERTEX>, mesh_type const& m,
                     Expression<tags::_hodge_star, TExpr> const& expr, IdxShift S, int n, Others&&... others) {
        auto const& l = std::get<0>(expr.m_args_);
        return (getV(m, l, S + IdxShift{0, 0, 0}, n, std::forward<Others>(others)...) +
                getV(m, l, S + IdxShift{0, 0, 1}, n, std::forward<Others>(others)...) +
                getV(m, l, S + IdxShift{0, 1, 0}, n, std::forward<Others>(others)...) +
                getV(m, l, S + IdxShift{0, 1, 1}, n, std::forward<Others>(others)...) +
                getV(m, l, S + IdxShift{1, 0, 0}, n, std::forward<Others>(others)...) +
                getV(m, l, S + IdxShift{1, 0, 1}, n, std::forward<Others>(others)...) +
                getV(m, l, S + IdxShift{1, 1, 0}, n, std::forward<Others>(others)...) +
                getV(m, l, S + IdxShift{1, 1, 1}, n, std::forward<Others>(others)...)) *
               getValue(m, m.m_volume_inv_volume_, n, std::forward<Others>(others)...) * 0.125;
    };
    ////***************************************************************************************************
    //! p_curl<1>
    //    static constexpr Real m_p_curl_factor_[3] = {0, 1, -1};
    //    template<typename TOP, typename T>   traits::value_type_t
    //    <Expression<TOP, T>>
    //    GetValue(mesh_type const &m, Expression<TOP, T> const &expr,
    //    EntityId const &s,
    //    ENABLE_IF((std::is_same<TOP, tags::_p_exterior_derivative < 0>>
    //                      ::value && traits::GetIFORM<T>::value == EDGE))
    //    )
    //    {
    //        return (get_v(m, std::Serialize<0>(expr.m_args_), s + EntityIdCoder::DI(I)) -
    //                get_v(m, std::Serialize<0>(expr.m_args_), s - EntityIdCoder::DI(I))
    //               ) * m.inv_volume(s) * m_p_curl_factor_[(I + 3 -
    //               EntityIdCoder::sub_index(s)) % 3];
    //    }
    //
    //
    //    template<typename T, size_t I>
    //      traits::value_type_t
    //    <Expression<tags::_p_codifferential_derivative < I>, T>>
    //    GetValue(
    //    mesh_type const &m,
    //    Expression<tags::_p_codifferential_derivative < I>, T
    //    > const &expr,
    //    EntityId const &s,
    //    ENABLE_IF(traits::GetIFORM<T>::value == FACE)
    //    )
    //    {
    //
    //        return (get_v(m, std::Serialize<0>(expr.m_args_), s + EntityIdCoder::DI(I)) -
    //                get_v(m, std::Serialize<0>(expr.m_args_), s - EntityIdCoder::DI(I))
    //               ) * m.inv_dual_volume(s) * m_p_curl_factor_[(I + 3 -
    //               EntityIdCoder::sub_index(s)) % 3];
    //    }

    ////***************************************************************************************************
    //
    ////! map_to

    template <int I, typename TExpr, typename... Others>
    static auto _map_to(std::index_sequence<I, I>, mesh_type const& m, TExpr const& expr, IdxShift S,
                        Others&&... others) {
        return getValue(m, expr, S, std::forward<Others>(others)...);
    };

    template <typename TExpr, typename... Others>
    static auto _map_to(std::index_sequence<VERTEX, EDGE>, mesh_type const& m, TExpr const& expr, IdxShift S, int n,
                        Others&&... others) {
        IdxShift D{0, 0, 0};
        D[n] = 1;

        return (getValue(m, expr, S, 0, std::forward<Others>(others)...) +
                getValue(m, expr, S + D, 0, std::forward<Others>(others)...)) *
               0.5;
    }

    template <typename TExpr, typename... Others>
    static auto _map_to(std::index_sequence<EDGE, VERTEX>, mesh_type const& m, TExpr const& expr, IdxShift S, int n0,
                        int n, Others&&... others) {
        IdxShift D{0, 0, 0};

        D[n] = 1;
        return (getValue(m, expr, S - D, n, std::forward<Others>(others)...) +
                getValue(m, expr, S, n, std::forward<Others>(others)...)) *
               0.5;
    }

    template <typename TExpr, typename... Others>
    static auto _map_to(std::index_sequence<VERTEX, FACE>, mesh_type const& m, TExpr const& expr, IdxShift S, int n,
                        Others&&... others) {
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;
        return (getValue(m, expr, S, 0, std::forward<Others>(others)...) +
                getValue(m, expr, S + Y, 0, std::forward<Others>(others)...) +
                getValue(m, expr, S + Z, 0, std::forward<Others>(others)...) +
                getValue(m, expr, S + Y + Z, 0, std::forward<Others>(others)...)) *
               0.25;
    }

    template <typename TExpr, typename... Others>
    static auto _map_to(std::index_sequence<FACE, VERTEX>, mesh_type const& m, TExpr const& expr, IdxShift S, int n0,
                        int n, Others&&... others) {
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};

        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (getValue(m, expr, S - Y - Z, n, std::forward<Others>(others)...) +
                getValue(m, expr, S - Y, n, std::forward<Others>(others)...) +
                getValue(m, expr, S - Z, n, std::forward<Others>(others)...) +
                getValue(m, expr, S, n, std::forward<Others>(others)...)) *
               0.25;
    }

    template <typename TExpr, typename... Others>
    static auto _map_to(std::index_sequence<VERTEX, VOLUME>, mesh_type const& m, TExpr const& expr, IdxShift S,
                        Others&&... others) {
        return (getValue(m, expr, S + IdxShift{0, 0, 0}, std::forward<Others>(others)...) +
                getValue(m, expr, S + IdxShift{0, 0, 1}, std::forward<Others>(others)...) +
                getValue(m, expr, S + IdxShift{0, 1, 0}, std::forward<Others>(others)...) +
                getValue(m, expr, S + IdxShift{0, 1, 1}, std::forward<Others>(others)...) +
                getValue(m, expr, S + IdxShift{1, 0, 0}, std::forward<Others>(others)...) +
                getValue(m, expr, S + IdxShift{1, 0, 1}, std::forward<Others>(others)...) +
                getValue(m, expr, S + IdxShift{1, 1, 0}, std::forward<Others>(others)...) +
                getValue(m, expr, S + IdxShift{1, 1, 1}, std::forward<Others>(others)...)) *
               0.125;
    }

    template <typename TExpr, typename... Others>
    static auto _map_to(std::index_sequence<VOLUME, VERTEX>, mesh_type const& m, TExpr const& expr, IdxShift S,
                        Others&&... others) {
        return (getValue(m, expr, S - IdxShift{1, 1, 1}, std::forward<Others>(others)...) +
                getValue(m, expr, S - IdxShift{1, 1, 0}, std::forward<Others>(others)...) +
                getValue(m, expr, S - IdxShift{1, 0, 1}, std::forward<Others>(others)...) +
                getValue(m, expr, S - IdxShift{1, 0, 0}, std::forward<Others>(others)...) +
                getValue(m, expr, S - IdxShift{0, 1, 1}, std::forward<Others>(others)...) +
                getValue(m, expr, S - IdxShift{0, 1, 0}, std::forward<Others>(others)...) +
                getValue(m, expr, S - IdxShift{0, 0, 1}, std::forward<Others>(others)...) +
                getValue(m, expr, S, std::forward<Others>(others)...)) *
               0.125;
    }

    template <typename TExpr, typename... Others>
    static auto _map_to(std::index_sequence<VOLUME, FACE>, mesh_type const& m, TExpr const& expr, IdxShift S, int n,
                        Others&&... others) {
        IdxShift D{0, 0, 0};
        D[n] = 1;
        return (getValue(m, expr, S - D, std::forward<Others>(others)...) +
                getValue(m, expr, S, std::forward<Others>(others)...)) *
               0.5;
    }

    template <typename TExpr, typename... Others>
    static auto _map_to(mesh_type const& m, TExpr const& expr, IdxShift S, int n0, int n, Others&&... others,
                        std::index_sequence<FACE, VOLUME>) {
        IdxShift D{0, 0, 0};
        D[n] = 1;

        return (getValue(m, expr, S, n, std::forward<Others>(others)...) +
                getValue(m, expr, S + D, n, std::forward<Others>(others)...)) *
               0.5;
    }

    template <typename TExpr, typename... Others>
    static auto _map_to(std::index_sequence<VOLUME, EDGE>, mesh_type const& m, TExpr const& expr, IdxShift S, int n,
                        Others&&... others) {
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (getValue(m, expr, S - Y, std::forward<Others>(others)...) +
                getValue(m, expr, S - Z, std::forward<Others>(others)...) +
                getValue(m, expr, S - Y - Z, std::forward<Others>(others)...) +
                getValue(m, expr, S, std::forward<Others>(others)...)) *
               0.25;
    }

    template <typename TExpr, typename... Others>
    static auto _map_to(std::index_sequence<EDGE, VOLUME>, mesh_type const& m, TExpr const& expr, IdxShift S, int n0,
                        int n, Others&&... others) {
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};

        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (getValue(m, expr, S, n, std::forward<Others>(others)...) +
                getValue(m, expr, S + Z, n, std::forward<Others>(others)...) +
                getValue(m, expr, S + Y, n, std::forward<Others>(others)...) +
                getValue(m, expr, S + Y + Z, n, std::forward<Others>(others)...)) *
               0.25;
    }

    template <typename TExpr, int ISrc, int IDest, typename... Others>
    static auto eval(std::integer_sequence<int, ISrc>, mesh_type const& m,
                     Expression<tags::map_to<IDest>, TExpr> const& expr, IdxShift S, Others&&... others) {
        return _map_to(std::index_sequence<ISrc, IDest>(), m, std::get<0>(expr.m_args_), S,
                       std::forward<Others>(others)...);
    }
    //***************************************************************************************************
    //
    //! Form<IL> ^ Form<IR> => Form<IR+IL>
    template <typename... TExpr, int IL, int IR, typename... Others>
    static auto eval(std::integer_sequence<int, IL, IR>, mesh_type const& m,
                     Expression<tags::_wedge, TExpr...> const& expr, IdxShift S, Others&&... others) {
        return m.inner_product(_map_to(std::index_sequence<IL, IR + IL>(), m, std::get<0>(expr.m_args_), S,
                                       std::forward<Others>(others)...),
                               _map_to(std::index_sequence<IR, IR + IL>(), m, std::get<1>(expr.m_args_), S,
                                       std::forward<Others>(others)...));
    }

    template <typename... TExpr, typename... Others>
    static auto eval(std::integer_sequence<int, EDGE, EDGE>, mesh_type const& m,
                     Expression<tags::_wedge, TExpr...> const& expr, IdxShift S, int n, Others&&... others) {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (getValue(m, l, S, (n + 2) % 3, std::forward<Others>(others)...) +
                getValue(m, l, S + Y, (n + 2) % 3, std::forward<Others>(others)...)) *
               (getValue(m, l, S, (n + 1) % 3, std::forward<Others>(others)...) +
                getValue(m, l, S + Z, (n + 1) % 3, std::forward<Others>(others)...)) *
               0.25;
    }

    template <typename... TExpr, typename... Others>
    static auto eval(std::integer_sequence<int, FACE, FACE>, mesh_type const& m,
                     Expression<tags::_wedge, TExpr...> const& expr, IdxShift S, int n, Others&&... others) {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;
        return getValue(m, l, S + Z, n, std::forward<Others>(others)...) *
                   getValue(m, r, S + Y, n, std::forward<Others>(others)...) -
               getValue(m, l, S + Z, n, std::forward<Others>(others)...) *
                   getValue(m, r, S + Y, n, std::forward<Others>(others)...);
    }

    template <typename... TExpr, typename... Others>
    static auto eval(std::integer_sequence<int, VERTEX, VERTEX>, mesh_type const& m,
                     Expression<tags::_dot, TExpr...> const& expr, IdxShift S, int n0, Others&&... others) {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        return getValue(m, l, S, 0, std::forward<Others>(others)...) *
                   getValue(m, r, S, 0, std::forward<Others>(others)...) +
               getValue(m, l, S, 1, std::forward<Others>(others)...) *
                   getValue(m, r, S, 1, std::forward<Others>(others)...) +
               getValue(m, l, S, 2, std::forward<Others>(others)...) *
                   getValue(m, r, S, 2, std::forward<Others>(others)...);
    }

    template <typename... TExpr, typename... Others>
    static auto eval(std::integer_sequence<int, VOLUME, VOLUME>, mesh_type const& m,
                     Expression<tags::_dot, TExpr...> const& expr, IdxShift S, int n0, Others&&... others) {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        return getValue(m, l, S, 0, 0, std::forward<Others>(others)...) *
                   getValue(m, r, S, 0, 0, std::forward<Others>(others)...) +
               getValue(m, l, S, 0, 1, std::forward<Others>(others)...) *
                   getValue(m, r, S, 0, 1, std::forward<Others>(others)...) +
               getValue(m, l, S, 0, 2, std::forward<Others>(others)...) *
                   getValue(m, r, S, 0, 2, std::forward<Others>(others)...);
    }

    template <typename... TExpr, typename... Others>
    static auto eval(std::integer_sequence<int, EDGE, EDGE>, mesh_type const& m,
                     Expression<tags::_dot, TExpr...> const& expr, IdxShift S, Others&&... others) {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        return eval(std::integer_sequence<int, VERTEX>(), m, dot_v(map_to<VERTEX>(l), map_to<VERTEX>(r)), S,
                    std::forward<Others>(others)...);
    }
    template <typename... TExpr, typename... Others>
    static auto eval(std::integer_sequence<int, FACE, FACE>, mesh_type const& m,
                     Expression<tags::_dot, TExpr...> const& expr, IdxShift S, Others&&... others) {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);
        return eval(std::integer_sequence<int, VERTEX>(), m, dot_v(map_to<VERTEX>(l), map_to<VERTEX>(r)), S,
                    std::forward<Others>(others)...);
    }
    template <typename... TExpr, typename... Others>
    static auto eval(std::integer_sequence<int, VERTEX, VERTEX>, mesh_type const& m,
                     Expression<tags::_cross, TExpr...> const& expr, IdxShift S, int n0, int n, Others&&... others) {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        return getValue(m, l, S, 0, ((n + 1) % 3), std::forward<Others>(others)...) *
                   getValue(m, r, S, 0, ((n + 2) % 3), std::forward<Others>(others)...) -
               getValue(m, l, S, 0, ((n + 2) % 3), std::forward<Others>(others)...) *
                   getValue(m, r, S, 0, ((n + 1) % 3), std::forward<Others>(others)...);
    }

    template <typename... TExpr, typename... Others>
    static auto eval(std::integer_sequence<int, VOLUME, VOLUME>, mesh_type const& m,
                     Expression<tags::_cross, TExpr...> const& expr, IdxShift S, int n0, int n, Others&&... others) {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);
        return getValue(m, l, S, 0, (n + 1) % 3, std::forward<Others>(others)...) *
                   getValue(m, r, S, 0, (n + 2) % 3, std::forward<Others>(others)...) -
               getValue(m, l, S, 0, (n + 2) % 3, std::forward<Others>(others)...) *
                   getValue(m, r, S, 0, (n + 1) % 3, std::forward<Others>(others)...);
    }

    ///*********************************************************************************************
    /// @name general_algebra General algebra
    /// @{

    //    template <typename V, int I, int... D>
    //    static V const& getValue(mesh_type const& m, Field<M, V, I, D...> const& f, EntityId s) {
    //        return f[EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]][(s.w >> 3)](s.x, s.y, s.z);
    //    };
    //    template <typename V, int I, int... D>
    //    static V& getValue(mesh_type const& m, Field<M, V, I, D...>& f, EntityId s) {
    //        return f[EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]][(s.w >> 3)](s.x, s.y, s.z);
    //    };

    ///*********************************************************************************************
    /**
     * @ingroup interpolate
     * @brief basic linear interpolate
     */
    template <typename TD, typename TIDX>
    static auto gather_impl_(mesh_type const& m, TD const& f, TIDX const& idx) {
        EntityId X = (EntityIdCoder::_DI);
        EntityId Y = (EntityIdCoder::_DJ);
        EntityId Z = (EntityIdCoder::_DK);

        point_type r;  //= std::Serialize<1>(idx);
        EntityId s;    //= std::Serialize<0>(idx);

        return getValue(f, m, ((s + X) + Y) + Z) * (r[0]) * (r[1]) * (r[2]) +    //
               getValue(f, m, (s + X) + Y) * (r[0]) * (r[1]) * (1.0 - r[2]) +    //
               getValue(f, m, (s + X) + Z) * (r[0]) * (1.0 - r[1]) * (r[2]) +    //
               getValue(f, m, (s + X)) * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]) +  //
               getValue(f, m, (s + Y) + Z) * (1.0 - r[0]) * (r[1]) * (r[2]) +    //
               getValue(f, m, (s + Y)) * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]) +  //
               getValue(f, m, s + Z) * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]) +    //
               getValue(f, m, s) * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
    }

   public:
    template <typename TF>
    constexpr static auto gather(mesh_type const& m, TF const& f, point_type const& r,
                                 ENABLE_IF((traits::iform<TF>::value == VERTEX))) {
        return gather_impl_(m, f, m.point_global_to_local(r, 0));
    }

    template <typename TF>
    constexpr static auto gather(mesh_type const& m, TF const& f, point_type const& r,
                                 ENABLE_IF((traits::iform<TF>::value == EDGE))) {
        return traits::field_value_t<TF>{gather_impl_(m, f, m.point_global_to_local(r, 1)),
                                         gather_impl_(m, f, m.point_global_to_local(r, 2)),
                                         gather_impl_(m, f, m.point_global_to_local(r, 4))};
    }

    template <typename TF>
    constexpr static auto gather(mesh_type const& m, TF const& f, point_type const& r,
                                 ENABLE_IF((traits::iform<TF>::value == FACE))) {
        return traits::field_value_t<TF>{gather_impl_(m, f, m.point_global_to_local(r, 6)),
                                         gather_impl_(m, f, m.point_global_to_local(r, 5)),
                                         gather_impl_(m, f, m.point_global_to_local(r, 3))};
    }

    template <typename TF>
    constexpr static auto gather(mesh_type const& m, TF const& f, point_type const& x,
                                 ENABLE_IF((traits::iform<TF>::value == VOLUME))) {
        return gather_impl_(m, f, m.point_global_to_local(x, 7));
    }

    template <typename TF, typename IDX, typename TV>
    static void scatter_impl_(mesh_type const& m, TF& f, IDX const& idx, TV const& v) {
        EntityId X = (EntityIdCoder::_DI);
        EntityId Y = (EntityIdCoder::_DJ);
        EntityId Z = (EntityIdCoder::_DK);

        point_type r = std::get<1>(idx);
        EntityId s = std::get<0>(idx);

        getValue(f, m, ((s + X) + Y) + Z) += v * (r[0]) * (r[1]) * (r[2]);
        getValue(f, m, (s + X) + Y) += v * (r[0]) * (r[1]) * (1.0 - r[2]);
        getValue(f, m, (s + X) + Z) += v * (r[0]) * (1.0 - r[1]) * (r[2]);
        getValue(f, m, s + X) += v * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
        getValue(f, m, (s + Y) + Z) += v * (1.0 - r[0]) * (r[1]) * (r[2]);
        getValue(f, m, s + Y) += v * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]);
        getValue(f, m, s + Z) += v * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]);
        getValue(f, m, s) += v * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
    }

    template <typename TF, typename TX, typename TV>
    static void scatter_(mesh_type const& m, std::integral_constant<int, VERTEX>, TF& f, TX const& x, TV const& u) {
        scatter_impl_(m, f, m.point_global_to_local(x, 0), u);
    }

    template <typename TF, typename TX, typename TV>
    static void scatter_(mesh_type const& m, std::integral_constant<int, EDGE>, TF& f, TX const& x, TV const& u) {
        scatter_impl_(m, f, m.point_global_to_local(x, 1), u[0]);
        scatter_impl_(m, f, m.point_global_to_local(x, 2), u[1]);
        scatter_impl_(m, f, m.point_global_to_local(x, 4), u[2]);
    }

    template <typename TF, typename TX, typename TV>
    static void scatter_(mesh_type const& m, std::integral_constant<int, FACE>, TF& f, TX const& x, TV const& u) {
        scatter_impl_(m, f, m.point_global_to_local(x, 6), u[0]);
        scatter_impl_(m, f, m.point_global_to_local(x, 5), u[1]);
        scatter_impl_(m, f, m.point_global_to_local(x, 3), u[2]);
    }

    template <typename TF, typename TX, typename TV>
    static void scatter_(mesh_type const& m, std::integral_constant<int, VOLUME>, TF& f, TX const& x, TV const& u) {
        scatter_impl_(m, f, m.point_global_to_local(x, 7), u);
    }

   public:
    template <typename TF, typename... Args>
    static void scatter(mesh_type const& m, TF& f, Args&&... args) {
        scatter_(m, traits::iform<TF>(), f, std::forward<Args>(args)...);
    }

    template <typename TV>
    static auto sample_(mesh_type const& m, EntityId s, TV& v) {
        return v;
    }

    template <typename TV, int N>
    static auto sample_(mesh_type const& m, EntityId s, nTuple<TV, N> const& v) {
        return v[((s.w & 0b111) == 0 || (s.w & 0b111) == 7) ? (s.w >> 3) % N
                                                            : EntityIdCoder::m_id_to_sub_index_[s.w & 0b11]];
    }

    template <typename TV>
    static auto sample(mesh_type const& m, EntityId s, TV const& v) {
        return sample_(m, s, v);
    }

    //    template <typename TFun>
    //    static auto getValue(mesh_type const& m, TFun const& fun,  IdxShift S, int n, Others&&... others,
    //                         ENABLE_IF(simpla::concept::is_callable<TFun(simpla::EntityId)>::value)) {
    //        return [&](index_tuple const& idx) {
    //            EntityId s;
    //            s.w = static_cast<int16_t>(tag);
    //            s.x = static_cast<int16_t>(idx[0] + S[0]);
    //            s.y = static_cast<int16_t>(idx[1] + S[1]);
    //            s.z = static_cast<int16_t>(idx[2] + S[2]);
    //            return sample(m, s, fun(s));
    //        };
    //    }
    //
    //    template <typename TFun>
    //    static auto getValue(mesh_type const& m, TFun const& fun,  IdxShift S, int n, Others&&... others,
    //                         ENABLE_IF(simpla::concept::is_callable<TFun(point_type const&)>::value)) {
    //        return [&](index_tuple const& idx) {
    //            EntityId s;
    //            s.w = static_cast<int16_t>(tag);
    //            s.x = static_cast<int16_t>(idx[0] + S[0]);
    //            s.y = static_cast<int16_t>(idx[1] + S[1]);
    //            s.z = static_cast<int16_t>(idx[2] + S[2]);
    //            return sample(m, tag, fun(m.point(s)));
    //        };
    //    }
    //
    //    template <int IFORM, typename TExpr>
    //    static auto  getValue(std::integral_constant<int, IFORM> const&,  TExpr const& expr,mesh_type const&
    //    m,
    //    index_type i,
    //                  index_type j, index_type k, unsigned int n, unsigned int d)  {
    //        return getValue( expr,m, EntityIdCoder::Serialize<IFORM>(i, j, k, n, d));
    //    }
    //    template <typename TField, typename TOP, typename... Args>
    //    void foreach_(mesh_type const& m, TField& self, Range<EntityId> const& r, TOP const& op, Args&&... args) const
    //    {
    //        r.foreach ([&](EntityId s)  { op(getValue(m, self, s), getValue(m, std::forward<Others>(others), s)...);
    //        });
    //    }
    //    template <typename... Args>
    //    void foreach (Args&&... args)  {
    //        foreach_(std::forward<Others>(others)...);
    //    }
    //    template <typename TField, typename... Args>
    //    void foreach (mesh_type const& m, TField & self, mesh::MeshZoneTag const& tag, Args && ... args)  {
    //        foreach_(m, self, m.range(tag, traits::iform<TField>::value, traits::dof<TField>::value),
    //                 std::forward<Others>(others)...);
    //    }
    //    template <typename TField, typename... Args>
    //    void foreach (mesh_type const& m, TField & self, Args && ... args)  {
    //        foreach_(m, self, m.range(SP_ES_ALL, traits::iform<TField>::value, traits::dof<TField>::value),
    //                 std::forward<Others>(others)...);
    //    }

    //**********************************************************************************************
    // for element-wise arithmetic operation

    template <typename U, int IFORM, typename... E>
    static void SetEntity(Field<M, U, IFORM>& lhs, Expression<E...> const& rhs, EntityId s) {
        SetEntity(lhs, getValue(*dynamic_cast<M const*>(lhs.GetMesh()), rhs, IdxShift{0, 0, 0},
                                EntityIdCoder::m_id_to_sub_index_[s.w & 0b111], s.x, s.y, s.z),
                  s);
    }
    template <typename U, int IFORM, int DOF, typename... E>
    static void SetEntity(Field<M, U, IFORM, DOF>& lhs, Expression<E...> const& rhs, EntityId s) {
        SetEntity(lhs, getValue(*dynamic_cast<M const*>(lhs.GetMesh()), rhs, IdxShift{0, 0, 0},
                                EntityIdCoder::m_id_to_sub_index_[s.w & 0b111], (s.w >> 3) & 0b111, s.x, s.y, s.z),
                  s);
    }

    template <typename U, int IFORM, int... DOF, typename RHS>
    static void SetEntity(Field<M, U, IFORM, DOF...>& lhs, RHS const& rhs, EntityId s,
                          ENABLE_IF((std::is_arithmetic<RHS>::value))) {
        lhs[s] = rhs;
    }
    template <typename U, int IFORM, int... DOF, typename V, int N>
    static void SetEntity(Field<M, U, IFORM, DOF...>& lhs, nTuple<V, N> const& rhs, EntityId s) {
        lhs[s] =
            rhs[(IFORM == VERTEX || IFORM == VOLUME) ? (s.w >> 3) : EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]];
    }

    template <typename U, int IFORM, int... DOF, typename RHS>
    static void SetEntity(Field<M, U, IFORM, DOF...>& lhs, RHS const& rhs, EntityId s,
                          ENABLE_IF((traits::is_invocable<RHS, EntityId>::value))) {
        SetEntity(lhs, rhs(s), s);
    }
    template <typename U, int IFORM, int... DOF, typename RHS>
    static void SetEntity(Field<M, U, IFORM, DOF...>& lhs, RHS const& rhs, EntityId s,
                          ENABLE_IF((traits::is_invocable<RHS, point_type>::value))) {
        SetEntity(lhs, rhs(dynamic_cast<M const*>(lhs.GetMesh())->point(s)), s);
    }
    template <typename U, int IFORM, int... DOF, typename RHS>
    static void Fill(Field<M, U, IFORM, DOF...>& lhs, RHS const& rhs) {
        M const& m = *dynamic_cast<M const*>(lhs.GetMesh());
        traits::foreach (lhs.data(), [&](auto& a, auto&&... subs) {
            a = getValue(m, rhs, IdxShift{0, 0, 0}, std::forward<decltype(subs)>(subs)...);
        });
    }

    template <typename U, int IFORM, int... DOF, typename RHS>
    static void Fill(Field<M, U, IFORM, DOF...>& lhs, RHS const& rhs, EntityRange r) {
        M const& m = *dynamic_cast<M const*>(lhs.GetMesh());

        if (r.isUndefined()) {
            traits::foreach (lhs.data(), [&](auto& a, int n0, auto&&... subs) {
                int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][n0] |
                          (reduction(tags::multiplication(), std::forward<decltype(subs)>(subs)...) << 3);

                int n = (IFORM == VERTEX || IFORM == VOLUME)
                            ? (reduction(tags::addition(), std::forward<decltype(subs)>(subs)...))
                            : n0;
                a = [&](index_type x, index_type y, index_type z) {
                    EntityId s;
                    s.w = tag;
                    s.x = x;
                    s.y = y;
                    s.z = z;
                    return calculus::getValue((getValue(m, rhs, IdxShift{0, 0, 0}, s)), n);
                };
            });
        } else if (!r.isNull()) {
            r.foreach ([&](EntityId s) {
                lhs[s] = calculus::getValue(getValue(m, rhs, IdxShift{0, 0, 0}, s),
                                            (IFORM == VERTEX || IFORM == VOLUME) ? (s.w >> 3) : (s.w & 0b111));
            });
        }
    }
};

//********************************************************************************************************************************

/**
 * A radial basis function (RBF) is a real-valued function whose value
 * depends
 * only
 * on the distance from the origin, so that \f$\phi(\mathbf{x}) =
 * \phi(\|\mathbf{x}\|)\f$;
 * or alternatively on the distance from some other point c, called a
 * center,
 * so that
 * \f$\phi(\mathbf{x}, \mathbf{c}) = \phi(\|\mathbf{x}-\mathbf{c}\|)\f$.
 */
//
//    Real RBF(mesh_type const& m, point_type const& x0, point_type const& x1, vector_type const& a)  {
//        vector_type r;
//        r = (x1 - x0) / a;
//        // @NOTE this is not  an exact  RBF
//        return (1.0 - std::abs(r[0])) * (1.0 - std::abs(r[1])) * (1.0 - std::abs(r[2]));
//    }
//
//    Real RBF(mesh_type const& m, point_type const& x0, point_type const& x1, Real const& a)  {
//        return (1.0 - m.distance(x1, x0) / a);
//    }

//    template <int DOF, typename... U>
//     void Assign(mesh_type const& m, Field<mesh_type, U...>& f, EntityId
//    s,
//                       nTuple<U, DOF> const& v)  {
//        for (int i = 0; i < DOF; ++i) { f[EntityIdCoder::sw(s, i)] = v[i]; }
//    }

////    template <typename... U>
////     void assign(mesh_type const& m, Field<U...>& f,
////                       EntityId s, nTuple<U, 3> const& v) {
////        for (int i = 0; i < DOF; ++i) { f[EntityIdCoder::sw(s, i)] = v[EntityIdCoder::sub_index(s)]; }
////    }
////
////    template <typename V, int DOF, int... I, typename U>
////     void assign(mesh_type const& m, Field<mesh_type, V, FACE, DOF, I...>& f,
////                       EntityId s, nTuple<U, 3> const& v) {
////        for (int i = 0; i < DOF; ++i) { f[EntityIdCoder::sw(s, i)] = v[EntityIdCoder::sub_index(s)]; }
////    }
////
////    template <typename V, int DOF, int... I, typename U>
////     void assign(mesh_type const& m, Field<mesh_type, V, VOLUME, DOF, I...>& f,
////                       EntityId s, nTuple<U, DOF> const& v) {
////        for (int i = 0; i < DOF; ++i) { f[EntityIdCoder::sw(s, i)] = v[i]; }
//    }
//
//    template <typename V, int IFORM, int DOF, int... I, typename U>
//     void Assign(mesh_type const& m, Field<mesh_type, V, IFORM, DOF, I...>& f,
//                       EntityId s, U const& v) {
//        for (int i = 0; i < DOF; ++i) { f[EntityIdCoder::sw(s, i)] = v; }
//    }

// template <typename TV, typename M, int IFORM, int DOF>
// constexpr Real   calculator<Field<TV, M, IFORM, DOF>>::m_p_curl_factor[3];
}  // namespace simpla { {

#endif /* FDM_H_ */
