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

#include <simpla/utilities/Array.h>
#include <simpla/utilities/EntityId.h>
#include <simpla/utilities/FancyStream.h>
#include <simpla/utilities/macro.h>
#include <simpla/utilities/sp_def.h>
#include <simpla/utilities/type_traits.h>
#include "Calculus.h"
namespace simpla {
template <typename TM, typename TV, int IFORM, int DOF>
class Field;

/**
 * @ingroup diff_scheme
 * finite volume
 */
template <typename TM>
struct calculator {
    typedef TM mesh_type;
    typedef calculator<mesh_type> this_type;

    //**********************************************************************************************
    // for element-wise arithmetic operation
    template <typename TOP, typename... T, int... I>
    auto _invoke_helper(mesh_type const& m, Expression<TOP, T...> const& expr, int n, IdxShift S,
                        int_sequence<I...>) const {
        return expr.m_op_(getValue(m, std::get<I>(expr.m_args_), n, S)...);
    }

    template <typename TOP, typename... T, int... I>
    auto eval(mesh_type const& m, Expression<TOP, T...> const& expr, int n, IdxShift S, int_sequence<I...>) const {
        return _invoke_helper(m, expr, n, S, make_int_sequence<sizeof...(I)>());
    }

    template <typename TOP, typename... T>
    auto getValue(mesh_type const& m, Expression<TOP, T...> const& expr, int n, IdxShift S) const {
        return eval(m, expr, n, S, int_sequence<traits::iform<T>::value...>());
    }

    template <typename M, typename V, int I, int D>
    auto getValue(mesh_type const& m, Field<M, V, I, D> const& f, int n, IdxShift S) const {
        return f[n / D][n % D](S);
    };
    template <typename M, typename V, int I, int D>
    auto getValue(mesh_type const& m, Field<M, V, I, D>& f, int n, IdxShift S) const {
        return f[n / D][n % D](S);
    };

    template <typename T>
    T const& getValue(mesh_type const& m, T const& v, int n, IdxShift S,
                      ENABLE_IF((std::is_arithmetic<T>::value))) const {
        return v;
    }
    template <typename T>
    T const& getValue(mesh_type const& m, T const* v, int n, IdxShift S,
                      ENABLE_IF((std::is_arithmetic<T>::value))) const {
        return v[n];
    }
    auto Volume(mesh_type const& m, int n, IdxShift S) const { return getValue(m, m.m_volume_, n, S); }
    auto IVolume(mesh_type const& m, int n, IdxShift S) const { return getValue(m, m.m_inv_volume_, n, S); }
    auto DVolume(mesh_type const& m, int n, IdxShift S) const { return getValue(m, m.m_dual_volume_, n, S); }
    auto IDVolume(mesh_type const& m, int n, IdxShift S) const { return getValue(m, m.m_inv_dual_volume_, n, S); }

    template <typename TExpr>
    auto getV(mesh_type const& m, TExpr const& expr, int n, IdxShift S) const {
        return getValue(m, expr, n, S) * Volume(m, n, S);
    }
    template <typename TExpr>
    auto getDualV(mesh_type const& m, TExpr const& expr, int n, IdxShift S) const {
        return getValue(m, expr, n, S) * DVolume(m, n, S);
    }

    //******************************************************************************
    // Exterior algebra
    //******************************************************************************

    //! grad<0>

    template <typename TExpr>
    auto eval(mesh_type const& m, Expression<tags::_exterior_derivative, TExpr> const& expr, int n, IdxShift S,
              int_sequence<VERTEX>) const {
        auto const& l = std::get<0>(expr.m_args_);
        IdxShift D{0, 0, 0};
        D[n] = 1;
        return (getV(m, l, n, S + D) - getV(m, l, n, S)) * IVolume(m, n, S);
    }

    //! curl<1>

    template <typename TExpr>
    auto eval(mesh_type const& m, Expression<tags::_exterior_derivative, TExpr> const& expr, int n, IdxShift S,
              int_sequence<EDGE>) const {
        auto const& l = std::get<0>(expr.m_args_);

        IdxShift X{0, 0, 0};
        IdxShift Y{0, 0, 0};

        IdxShift Z{0, 0, 0};
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return ((getV(m, l, (n + 1) % 3, S) -        //
                 getV(m, l, (n + 1) % 3, S - Y)) +   //
                (getV(m, l, (n + 2) % 3, S) -        //
                 getV(m, l, (n + 2) % 3, S - Z))) *  //
               IVolume(m, n, S);
    }

    //! div<2>
    template <typename TExpr>
    auto eval(mesh_type const& m, Expression<tags::_exterior_derivative, TExpr> const& expr, int n, IdxShift S,
              int_sequence<FACE>) const {
        auto const& l = std::get<0>(expr.m_args_);

        IdxShift X{1, 0, 0};
        IdxShift Y{0, 1, 0};
        IdxShift Z{0, 0, 1};

        return ((getV(m, l, 0, S + X) - getV(m, l, 0, S)) +  //
                (getV(m, l, 1, S + Y) - getV(m, l, 1, S)) +  //
                (getV(m, l, 2, S + Z) - getV(m, l, 2, S))    //
                ) *
               IVolume(m, 7, S);
    }

    //! curl<2>
    template <typename TExpr>
    auto eval(mesh_type const& m, Expression<tags::_codifferential_derivative, TExpr> const& expr, int n, IdxShift S,
              int_sequence<FACE>) const {
        auto const& l = std::get<0>(expr.m_args_);

        IdxShift X{0, 0, 0};
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};

        X[n] = 1;
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return ((getV(m, l, (n + 1) % 3, S + Y) -  //
                 getV(m, l, (n + 1) % 3, S)) +     //
                (getV(m, l, (n + 2) % 3, S + Z) -  //
                 getV(m, l, (n + 2) % 3, S))) *    //
               (-IVolume(m, n, S));
    }

    //! div<1>

    template <typename TExpr>
    auto eval(mesh_type const& m, Expression<tags::_codifferential_derivative, TExpr> const& expr, int n, IdxShift S,
              int_sequence<EDGE>) const {
        auto const& l = std::get<0>(expr.m_args_);

        IdxShift X{0, 0};
        IdxShift Y{1, 0};
        IdxShift Z{2, 0};

        return ((getDualV(m, l, 0, S) - getDualV(m, l, 0, S /*- X*/)) +  //
                (getDualV(m, l, 1, S) - getDualV(m, l, 1, S /*- Y*/)) +  //
                (getDualV(m, l, 2, S) - getDualV(m, l, 2, S /*- Z*/))) *
               (-IDVolume(m, n, S));
        ;
    }

    //! grad<3>

    template <typename TExpr>
    auto eval(mesh_type const& m, Expression<tags::_codifferential_derivative, TExpr> const& expr, int n, IdxShift S,
              int_sequence<VOLUME>) const {
        auto const& l = std::get<0>(expr.m_args_);

        IdxShift D{0, 0, 0};
        D[n] = 1;

        return (getV(m, l, n, S) - getV(m, l, n, S - D)) * (-IVolume(m, n, S));
    }
    //
    //    template<typename T>
    //      auto // traits::value_type_t
    //    <Expression<tags::_codifferential_derivative, T>>
    //    GetValue(mesh_type const &m,
    //    Expression<tags::_codifferential_derivative, T> const &expr,
    //              EntityId const &s)
    //    {
    //        _assert(traits::GetIFORM<T>::value != VOLUME &&
    //        traits::GetIFORM<T>::value != VERTEX,
    //                      "ERROR: grad VERTEX/VOLUME Field  ");
    //    };
    //! *Form<IR> => Form<N-IL>

    template <typename TExpr>
    auto eval(mesh_type const& m, Expression<tags::_hodge_star, TExpr> const& expr, int n, IdxShift S,
              int_sequence<VERTEX>) const {
        auto const& l = std::get<0>(expr.m_args_);

        return (                                              //
                   getV(m, l, n, S + IdxShift{-1, -1, -1}) +  //
                   getV(m, l, n, S + IdxShift{-1, -1, +1}) +  //
                   getV(m, l, n, S + IdxShift{-1, +1, -1}) +  //
                   getV(m, l, n, S + IdxShift{-1, +1, +1}) +  //
                   getV(m, l, n, S + IdxShift{+1, -1, -1}) +  //
                   getV(m, l, n, S + IdxShift{+1, -1, +1}) +  //
                   getV(m, l, n, S + IdxShift{+1, +1, -1}) +  //
                   getV(m, l, n, S + IdxShift{+1, +1, +1})    //
                   ) *
               IDVolume(m, n, S) * 0.125;
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
    //        return (get_v(m, std::PopPatch<0>(expr.m_args_), s + EntityIdCoder::DI(I)) -
    //                get_v(m, std::PopPatch<0>(expr.m_args_), s - EntityIdCoder::DI(I))
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
    //        return (get_v(m, std::PopPatch<0>(expr.m_args_), s + EntityIdCoder::DI(I)) -
    //                get_v(m, std::PopPatch<0>(expr.m_args_), s - EntityIdCoder::DI(I))
    //               ) * m.inv_dual_volume(s) * m_p_curl_factor_[(I + 3 -
    //               EntityIdCoder::sub_index(s)) % 3];
    //    }

    ////***************************************************************************************************
    //
    ////! map_to
    //    template<typename T, size_t I>
    //      T
    //    _map_to(mesh_type const &m, T const &r, EntityId const &s,
    //    int_sequence<VERTEX, I>,
    //          st::is_primary_t<T> *_p = nullptr) const { return r; }
    //
    //    template<typename TF, size_t I>
    //      traits::value_type_t<TF>
    //    _map_to(mesh_type const &m, TF const &expr, EntityId const &s,
    //    int_sequence<I, I>,
    //          std::enable_if_t<!st::is_primary<TF>::value>
    //          *_p = nullptr) const { return GetValue(m, expr, s); }

    template <typename TExpr, int I>
    auto _map_to(mesh_type const& m, TExpr const& l, int n, IdxShift S, std::index_sequence<I, I>) const {
        return getValue(m, l, n, S);
    };

    template <typename TExpr>
    auto _map_to(mesh_type const& m, TExpr const& l, int n, IdxShift S, std::index_sequence<VERTEX, EDGE>) const {
        IdxShift D{0, 0, 0};
        D[n] = 1;
        return (getValue(m, l, 0, S) + getValue(m, l, 0, S + D)) * 0.5;
    }

    template <typename TExpr>
    auto _map_to(mesh_type const& m, TExpr const& expr, int n, IdxShift S, std::index_sequence<VERTEX, FACE>) const {
        IdxShift X{0, 0, 0};
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        X[n] = 1;
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;
        return (                                    //
                   getValue(m, expr, 0, S) +        //
                   getValue(m, expr, 0, S + Y) +    //
                   getValue(m, expr, 0, S + Z) +    //
                   getValue(m, expr, 0, S + Y + Z)  //
                   ) *
               0.25;
    }

    template <typename TExpr>
    auto _map_to(mesh_type const& m, TExpr const& l, int n, IdxShift S, std::index_sequence<VERTEX, VOLUME>) const {
        return (                                                  //
                   getValue(m, l, 0, S + IdxShift{-1, -1, +1}) +  //
                   getValue(m, l, 0, S + IdxShift{-1, +1, -1}) +  //
                   getValue(m, l, 0, S + IdxShift{-1, -1, -1}) +  //
                   getValue(m, l, 0, S + IdxShift{-1, +1, +1}) +  //
                   getValue(m, l, 0, S + IdxShift{+1, -1, -1}) +  //
                   getValue(m, l, 0, S + IdxShift{+1, -1, +1}) +  //
                   getValue(m, l, 0, S + IdxShift{+1, +1, -1}) +  //
                   getValue(m, l, 0, S + IdxShift{+1, +1, +1})    //
                   ) *
               0.125;
    }

    template <typename TExpr>
    auto _map_to(mesh_type const& m, TExpr const& l, int n, IdxShift S, std::index_sequence<EDGE, VERTEX>) const {
        IdxShift D{0, 0, 0};
        D[n] = 1;
        return (getValue(m, l, n, S - D) + getValue(m, l, n, S)) * 0.5;
    }

    template <typename TExpr>
    auto _map_to(mesh_type const& m, TExpr const& expr, int n, IdxShift S, std::index_sequence<FACE, VERTEX>) const {
        IdxShift X{0, 0, 0};
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        X[n] = 1;
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (getValue(m, expr, n, S - Y - Z) +  //
                getValue(m, expr, n, S - Y + Z) +  //
                getValue(m, expr, n, S + Y - Z) +  //
                getValue(m, expr, n, S + Y + Z)    //
                ) *
               0.25;
    }

    template <typename TExpr>
    auto _map_to(mesh_type const& m, TExpr const& expr, int n, IdxShift S, std::index_sequence<VOLUME, VERTEX>) const {
        auto const& l = expr;

        IdxShift X{0, 0, 0};
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        X[n] = 1;
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (                                                  //
                   getValue(m, l, n, S + IdxShift{-1, -1, -1}) +  //
                   getValue(m, l, n, S + IdxShift{-1, -1, +1}) +  //
                   getValue(m, l, n, S + IdxShift{-1, +1, -1}) +  //
                   getValue(m, l, n, S + IdxShift{-1, +1, +1}) +  //
                   getValue(m, l, n, S + IdxShift{+1, -1, -1}) +  //
                   getValue(m, l, n, S + IdxShift{+1, -1, +1}) +  //
                   getValue(m, l, n, S + IdxShift{+1, +1, -1}) +  //
                   getValue(m, l, n, S + IdxShift{+1, +1, +1})    //
                   ) *
               0.125;
    }

    template <typename TExpr>
    auto _map_to(mesh_type const& m, TExpr const& expr, int n, IdxShift S, std::index_sequence<VOLUME, FACE>) const {
        IdxShift D{0, 0, 0};
        D[n] = 1;

        return (getValue(m, expr, n, S - D) + getValue(m, expr, n, S)) * 0.5;
    }

    template <typename TExpr>
    auto _map_to(mesh_type const& m, TExpr const& expr, int n, IdxShift S, std::index_sequence<VOLUME, EDGE>) const {
        auto const& l = expr;

        IdxShift X{0, 0, 0};
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        X[n] = 1;
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (                                   //
                   getValue(m, l, n, S - Y - Z) +  //
                   getValue(m, l, n, S - Y + Z) +  //
                   getValue(m, l, n, S + Y - Z) +  //
                   getValue(m, l, n, S + Y + Z)    //
                   ) *
               0.25;
    }

    template <typename TExpr>
    auto _map_to(mesh_type const& m, TExpr const& expr, int n, IdxShift S, std::index_sequence<FACE, VOLUME>) const {
        IdxShift D{0, 0, 0};
        D[n] = 1;

        return (getValue(m, expr, n, S) + getValue(m, expr, n, S + D)) * 0.5;
    }

    template <typename TExpr>
    auto _map_to(mesh_type const& m, TExpr const& expr, int n, IdxShift S, std::index_sequence<EDGE, VOLUME>) const {
        //        auto const &l = expr;
        //
        //        auto X = EntityIdCoder::DI(0, s);
        //        auto Y = EntityIdCoder::DI(1, s);
        //        auto Z = EntityIdCoder::DI(2, s);

        IdxShift X{0, 0, 0};
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        X[n] = 1;
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (getValue(m, expr, n, S - Y - Z) +  //
                getValue(m, expr, n, S - Y + Z) +  //
                getValue(m, expr, n, S + Y - Z) +  //
                getValue(m, expr, n, S + Y + Z)    //
                ) *
               0.25;
    }

    template <typename TExpr, int IL, int IR>
    auto eval(mesh_type const& m, Expression<simpla::tags::_map_to<IL>, TExpr> const& expr, int n, IdxShift S,
              int_sequence<IR>) const {
        return _map_to(m, std::get<0>(expr.m_args_), n, S, std::index_sequence<IL, IR>());
    }
    //***************************************************************************************************
    //
    //! Form<IL> ^ Form<IR> => Form<IR+IL>
    template <typename... TExpr, int IL, int IR>
    auto eval(mesh_type const& m, Expression<simpla::tags::_wedge, TExpr...> const& expr, int n, IdxShift S,
              int_sequence<IL, IR>) const {
        return m.inner_product(_map_to(m, std::get<0>(expr.m_args_), n, S, std::index_sequence<IL, IR + IL>()),
                               _map_to(m, std::get<1>(expr.m_args_), n, S, std::index_sequence<IR, IR + IL>()));
    }

    template <typename... TExpr>
    auto eval(mesh_type const& m, Expression<simpla::tags::_wedge, TExpr...> const& expr, int n, IdxShift S,
              int_sequence<EDGE, EDGE>) const {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        IdxShift X{0, 0, 0};
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        X[n] = 1;
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (getValue(m, l, n, S - Y) +   //
                getValue(m, l, n, S + Y)) *  //
               (getValue(m, l, n, S - Z) +   //
                getValue(m, l, n, S + Z)     //
                ) *
               0.25;
    }

    template <typename... TExpr, int I>
    auto eval(mesh_type const& m, Expression<simpla::tags::_wedge, TExpr...> const& expr, int n, IdxShift S,
              int_sequence<I, I>)  //
    {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        IdxShift X{0, 0, 0};
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        X[n] = 1;
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;
        return getValue(m, l, n, S + Z) * getValue(m, r, n, S + Y) -
               getValue(m, l, n, S + Z) * getValue(m, r, n, S + Y);
    }

    template <typename... TExpr, int I>
    auto eval(mesh_type const& m, Expression<simpla::tags::_dot, TExpr...> const& expr, int n, IdxShift S,
              int_sequence<I, I>)  //
    {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        IdxShift X{1, 0, 0};
        IdxShift Y{0, 1, 0};
        IdxShift Z{0, 0, 1};

        return getValue(m, l, n, S + X) * getValue(m, r, n, S + X) +
               getValue(m, l, n, S + Y) * getValue(m, r, n, S + Y) +
               getValue(m, l, n, S + Z) * getValue(m, r, n, S + Z);
    }
    template <typename... TExpr, int I, int K>
    auto eval(mesh_type const& m, Expression<tags::_cross, TExpr...> const& expr, int n, IdxShift S,
              int_sequence<I, K>)  //
    {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        IdxShift X{0, 0, 0};
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        X[n] = 1;
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;
        return getValue(m, l, n, S + Y) * getValue(m, r, n, S + Z) -
               getValue(m, l, n, S + Z) * getValue(m, r, n, S + Y);
    }
    //    template<typename TExpr, int I>  auto
    //    getValue(mesh_type const &m, TExpr const &expr, EntityId const &s,
    //         expression_tag<tags::divides, I, VERTEX>) //
    //    AUTO_RETURN((GetValue(m, std::PopPatch<0>(expr.m_args_), s) /
    //                 _map_to(m, std::PopPatch<1>(expr.m_args_), s,
    //                 int_sequence<VERTEX, I>())))

    //    template<typename TExpr, int I>  auto
    //    getValue(mesh_type const &m, TExpr const &expr, EntityId const &s,
    //         expression_tag<tags::multiplies, I, VERTEX>) //
    //    AUTO_RETURN((GetValue(m, std::PopPatch<0>(expr.m_args_), s) *
    //                 _map_to(m, std::PopPatch<1>(expr.m_args_), s,
    //                 int_sequence<VERTEX, I>())))

    ///*********************************************************************************************
    /// @name general_algebra General algebra
    /// @{
    //
    //    template <typename V, int I, int D>
    //    auto getValue(mesh_type const& m, Field<mesh_type, V, I, D> const& f, int n) const {
    //        return f[n / D][n % D];
    //    };
    //    template <typename V, int I, int D>
    //    auto getValue(mesh_type const& m, Field<mesh_type, V, I, D>& f, int n) const {
    //        return f[n / D][n % D];
    //    };
    //
    //    template <typename T>
    //    T const& getValue(mesh_type const& mesh_type, T const& v, int n, ENABLE_IF((std::is_arithmetic<T>::value)))
    //    const {
    //        return v;
    //    }
    //
    //    template <typename TOP, typename... T>
    //    auto getValue(mesh_type const& m, Expression<TOP, T...> const& expr, int n) const {
    //        return eval(m, expr, n, IdxShift{0, 0, 0}, expression_tag<TOP, traits::iform<T>::value...>());
    //    }
    //
    template <typename V, int I, int D>
    V const& getValue(mesh_type const& m, Field<TM, V, I, D> const& f, EntityId s) const {
        return f[EntityIdCoder::m_id_to_sub_index_[s.w % 0b111]][(s.w >> 3) % D](s.x, s.y, s.z);
    };
    template <typename V, int I, int D>
    V& getValue(mesh_type const& m, Field<TM, V, I, D>& f, EntityId s) const {
        return f[EntityIdCoder::m_id_to_sub_index_[s.w % 0b111]][(s.w >> 3) % D](s.x, s.y, s.z);
    };

    ///*********************************************************************************************
    /**
     * @ingroup interpolate
     * @brief basic linear interpolate
     */
    template <typename TD, typename TIDX>
    auto gather_impl_(mesh_type const& m, TD const& f, TIDX const& idx) const {
        EntityId X = (EntityIdCoder::_DI);
        EntityId Y = (EntityIdCoder::_DJ);
        EntityId Z = (EntityIdCoder::_DK);

        point_type r;  //= std::PopPatch<1>(idx);
        EntityId s;    //= std::PopPatch<0>(idx);

        return getValue(m, f, ((s + X) + Y) + Z) * (r[0]) * (r[1]) * (r[2]) +    //
               getValue(m, f, (s + X) + Y) * (r[0]) * (r[1]) * (1.0 - r[2]) +    //
               getValue(m, f, (s + X) + Z) * (r[0]) * (1.0 - r[1]) * (r[2]) +    //
               getValue(m, f, (s + X)) * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]) +  //
               getValue(m, f, (s + Y) + Z) * (1.0 - r[0]) * (r[1]) * (r[2]) +    //
               getValue(m, f, (s + Y)) * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]) +  //
               getValue(m, f, s + Z) * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]) +    //
               getValue(m, f, s) * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
    }

   public:
    template <typename TF>
    constexpr auto gather(mesh_type const& m, TF const& f, point_type const& r,
                          ENABLE_IF((traits::iform<TF>::value == VERTEX))) const {
        return gather_impl_(m, f, m.point_global_to_local(r, 0));
    }

    template <typename TF>
    constexpr auto gather(mesh_type const& m, TF const& f, point_type const& r,
                          ENABLE_IF((traits::iform<TF>::value == EDGE))) const {
        return traits::field_value_t<TF>{gather_impl_(m, f, m.point_global_to_local(r, 1)),
                                         gather_impl_(m, f, m.point_global_to_local(r, 2)),
                                         gather_impl_(m, f, m.point_global_to_local(r, 4))};
    }

    template <typename TF>
    constexpr auto gather(mesh_type const& m, TF const& f, point_type const& r,
                          ENABLE_IF((traits::iform<TF>::value == FACE))) const {
        return traits::field_value_t<TF>{gather_impl_(m, f, m.point_global_to_local(r, 6)),
                                         gather_impl_(m, f, m.point_global_to_local(r, 5)),
                                         gather_impl_(m, f, m.point_global_to_local(r, 3))};
    }

    template <typename TF>
    constexpr auto gather(mesh_type const& m, TF const& f, point_type const& x,
                          ENABLE_IF((traits::iform<TF>::value == VOLUME))) const {
        return gather_impl_(m, f, m.point_global_to_local(x, 7));
    }

    template <typename TF, typename IDX, typename TV>
    void scatter_impl_(mesh_type const& m, TF& f, IDX const& idx, TV const& v) const {
        EntityId X = (EntityIdCoder::_DI);
        EntityId Y = (EntityIdCoder::_DJ);
        EntityId Z = (EntityIdCoder::_DK);

        point_type r = std::get<1>(idx);
        EntityId s = std::get<0>(idx);

        getValue(m, f, ((s + X) + Y) + Z) += v * (r[0]) * (r[1]) * (r[2]);
        getValue(m, f, (s + X) + Y) += v * (r[0]) * (r[1]) * (1.0 - r[2]);
        getValue(m, f, (s + X) + Z) += v * (r[0]) * (1.0 - r[1]) * (r[2]);
        getValue(m, f, s + X) += v * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
        getValue(m, f, (s + Y) + Z) += v * (1.0 - r[0]) * (r[1]) * (r[2]);
        getValue(m, f, s + Y) += v * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]);
        getValue(m, f, s + Z) += v * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]);
        getValue(m, f, s) += v * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
    }

    template <typename TF, typename TX, typename TV>
    void scatter_(mesh_type const& m, int_const<VERTEX>, TF& f, TX const& x, TV const& u) const {
        scatter_impl_(m, f, m.point_global_to_local(x, 0), u);
    }

    template <typename TF, typename TX, typename TV>
    void scatter_(mesh_type const& m, int_const<EDGE>, TF& f, TX const& x, TV const& u) const {
        scatter_impl_(m, f, m.point_global_to_local(x, 1), u[0]);
        scatter_impl_(m, f, m.point_global_to_local(x, 2), u[1]);
        scatter_impl_(m, f, m.point_global_to_local(x, 4), u[2]);
    }

    template <typename TF, typename TX, typename TV>
    void scatter_(mesh_type const& m, int_const<FACE>, TF& f, TX const& x, TV const& u) const {
        scatter_impl_(m, f, m.point_global_to_local(x, 6), u[0]);
        scatter_impl_(m, f, m.point_global_to_local(x, 5), u[1]);
        scatter_impl_(m, f, m.point_global_to_local(x, 3), u[2]);
    }

    template <typename TF, typename TX, typename TV>
    void scatter_(mesh_type const& m, int_const<VOLUME>, TF& f, TX const& x, TV const& u) const {
        scatter_impl_(m, f, m.point_global_to_local(x, 7), u);
    }

   public:
    template <typename TF, typename... Args>
    void scatter(mesh_type const& m, TF& f, Args&&... args) const {
        scatter_(m, traits::iform<TF>(), f, std::forward<Args>(args)...);
    }

    template <typename TV>
    auto sample_(mesh_type const& m, EntityId s, TV& v) const {
        return v;
    }

    template <typename TV, int N>
    auto sample_(mesh_type const& m, EntityId s, nTuple<TV, N> const& v) const {
        return v[s.w % N];
    }

    template <typename TFun>
    auto getValue(mesh_type const& m, TFun const& fun, int n, IdxShift S,
                  ENABLE_IF(simpla::concept::is_callable<TFun(EntityId)>::value)) const {
        return [&](index_tuple const& idx) {
            EntityId s;
            s.w = n;
            s.x = idx[0] + S[0];
            s.y = idx[1] + S[1];
            s.z = idx[2] + S[2];
            return sample(m, s, fun(s));
        };
    }

    template <typename TFun>
    auto getValue(mesh_type const& m, TFun const& fun, int n, IdxShift S,
                  ENABLE_IF(simpla::concept::is_callable<TFun(point_type const&)>::value)) const {
        return [&](index_tuple const& idx) {
            EntityId s;
            s.w = n;
            s.x = idx[0] + S[0];
            s.y = idx[1] + S[1];
            s.z = idx[2] + S[2];
            return sample(m, s, fun(m.point(s)));
        };
    }

    template <typename TV>
    auto sample(mesh_type const& m, EntityId s, TV const& v) const {
        return sample_(m, s, v);
    }
    //
    //    template <int IFORM, typename TExpr>
    //    auto getValue(std::integral_constant<int, IFORM> const&, mesh_type const& m, TExpr const& expr, index_type i,
    //                  index_type j, index_type k, unsigned int n, unsigned int d) const {
    //        return getValue(m, expr, EntityIdCoder::Pack<IFORM>(i, j, k, n, d));
    //    }
    //    template <typename TField, typename TOP, typename... Args>
    //    void foreach_(mesh_type const& m, TField& self, Range<EntityId> const& r, TOP const& op, Args&&... args) const
    //    {
    //        r.foreach ([&](EntityId s) const { op(getValue(m, self, s), getValue(m, std::forward<Args>(args), s)...);
    //        });
    //    }
    //    template <typename... Args>
    //    void foreach (Args&&... args) const {
    //        foreach_(std::forward<Args>(args)...);
    //    }
    //    template <typename TField, typename... Args>
    //    void foreach (mesh_type const& m, TField & self, mesh::MeshZoneTag const& tag, Args && ... args) const {
    //        foreach_(m, self, m.range(tag, traits::iform<TField>::value, traits::dof<TField>::value),
    //                 std::forward<Args>(args)...);
    //    }
    //    template <typename TField, typename... Args>
    //    void foreach (mesh_type const& m, TField & self, Args && ... args) const {
    //        foreach_(m, self, m.range(SP_ES_ALL, traits::iform<TField>::value, traits::dof<TField>::value),
    //                 std::forward<Args>(args)...);
    //    }
};
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
//    Real RBF(mesh_type const& m, point_type const& x0, point_type const& x1, vector_type const& a) const {
//        vector_type r;
//        r = (x1 - x0) / a;
//        // @NOTE this is not  an exact  RBF
//        return (1.0 - std::abs(r[0])) * (1.0 - std::abs(r[1])) * (1.0 - std::abs(r[2]));
//    }
//
//    Real RBF(mesh_type const& m, point_type const& x0, point_type const& x1, Real const& a) const {
//        return (1.0 - m.distance(x1, x0) / a);
//    }

//    template <int DOF, typename... U>
//     void Assign(mesh_type const& m, Field<mesh_type, U...>& f, EntityId
//    s,
//                       nTuple<U, DOF> const& v) const {
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

// template <typename TV, typename TM, int IFORM, int DOF>
// constexpr Real   calculator<Field<TV, TM, IFORM, DOF>>::m_p_curl_factor[3];
}  // namespace simpla { {

#endif /* FDM_H_ */
