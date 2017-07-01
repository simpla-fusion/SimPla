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
#include <simpla/engine/SPObject.h>
#include <simpla/utilities/macro.h>
#include <simpla/utilities/type_traits.h>
#include "Calculus.h"
namespace simpla {
template <typename TM, typename TV, int, int...>
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
    template <typename TOP, typename... T, size_t... I>
    static auto _invoke_helper(Expression<TOP, T...> const& expr, mesh_type const& m, int tag, IdxShift S,
                               std::index_sequence<I...>) {
        return expr.m_op_(getValue(std::get<I>(expr.m_args_), m, tag, S)...);
    }

    template <typename TOP, typename... T, int... I>
    static auto eval(Expression<TOP, T...> const& expr, mesh_type const& m, int tag, IdxShift S,
                     std::integer_sequence<int, I...>) {
        return _invoke_helper(expr, m, tag, S, std::make_index_sequence<sizeof...(I)>());
    }

    template <typename TOP, typename... T>
    static auto getValue(Expression<TOP, T...> const& expr, mesh_type const& m, int tag,
                         IdxShift S = IdxShift{0, 0, 0}) {
        return eval(expr, m, tag, S, std::integer_sequence<int, traits::iform<T>::value...>());
    }

    template <typename M, typename V, int I, int... D>
    static auto getValue(Field<M, V, I, D...> const& f, mesh_type const& m, int tag, IdxShift S) {
        return f[EntityIdCoder::m_id_to_sub_index_[(tag & 0b111)]][(tag >> 3)](S);
    };
    template <typename M, typename V, int I, int... D>
    static auto getValue(Field<M, V, I, D...>& f, mesh_type const& m, int tag, IdxShift S) {
        return f[EntityIdCoder::m_id_to_sub_index_[tag & 0b111]][(tag >> 3)](S);
    };

    template <typename TFun>
    static auto getValue(TFun const& f, mesh_type const& m, int tag, IdxShift S = IdxShift{0, 0, 0},
                         ENABLE_IF((concept::is_callable<TFun(simpla::EntityId)>::value))) {
        return [&](index_tuple const& idx) {
            EntityId s;
            s.w = static_cast<int16_t>(tag);
            s.x = static_cast<int16_t>(idx[0] + S[0]);
            s.y = static_cast<int16_t>(idx[1] + S[1]);
            s.z = static_cast<int16_t>(idx[2] + S[2]);
            return f(s);
        };
    };

    template <typename T>
    static T const& getValue(T const& v, mesh_type const& m, int tag, IdxShift S = IdxShift{0, 0, 0},
                             ENABLE_IF((std::is_arithmetic<T>::value))) {
        return v;
    }
    template <typename T>
    static T const& getValue(T const* v, mesh_type const& m, int tag, IdxShift S = IdxShift{0, 0, 0},
                             ENABLE_IF((std::is_arithmetic<T>::value))) {
        return v[(((tag & 0b111) == 0) || ((tag & 0b111) == 7)) ? (tag >> 3)
                                                                : EntityIdCoder::m_id_to_sub_index_[tag & 0b111]];
    }
    //    static auto Volume(mesh_type const& m, int tag, IdxShift S) {
    //        return getValue(m, m.m_volume_, ((tag & 0b111) << 3) | 0b111, S);
    //    }
    //    static auto IVolume(mesh_type const& m, int tag, IdxShift S) {
    //        return getValue(m, m.m_inv_volume_, ((tag & 0b111) << 3) | 0b111, S);
    //    }
    //    static auto DVolume(mesh_type const& m, int tag, IdxShift S) {
    //        return getValue(m, m.m_dual_volume_, ((tag & 0b111) << 3) | 0b111, S);
    //    }
    //    static auto IDVolume(mesh_type const& m, int tag, IdxShift S) {
    //        return getValue(m, m.m_inv_dual_volume_, ((tag & 0b111) << 3) | 0b111, S);
    //    }
    //
    //
    //    template <typename TExpr>
    //    static auto getV_(mesh_type const& m, std::integral_constant<int, VERTEX>, TExpr const& expr, int tag,
    //    IdxShift S) {
    //        return getValue(m, expr * m.m_vertex_volume_, tag, S);
    //    }
    //    template <typename TExpr>
    //    static auto getV_(mesh_type const& m, std::integral_constant<int, EDGE>, TExpr const& expr, int tag, IdxShift
    //    S) {
    //        return getValue(m, expr * m.m_edge_volume_, tag, S);
    //    }
    //    template <typename TExpr>
    //    static auto getV_(mesh_type const& m, std::integral_constant<int, FACE>, TExpr const& expr, int tag, IdxShift
    //    S) {
    //        return getValue(m, expr * m.m_face_volume_, tag, S);
    //    }
    //    template <typename TExpr>
    //    static auto getV_(mesh_type const& m, std::integral_constant<int, VOLUME>, TExpr const& expr, int tag,
    //    IdxShift S) {
    //        return getValue(m, expr * m.m_volume_volume_, tag, S);
    //    }
    template <typename TExpr>
    static auto getV(TExpr const& expr, mesh_type const& m, int tag, IdxShift S,
                     ENABLE_IF((traits::iform<TExpr>::value == VERTEX))) {
        return getValue(expr, m, tag, S) * getValue(m.m_vertex_volume_, m, tag, S);
    }
    template <typename TExpr>
    static auto getV(TExpr const& expr, mesh_type const& m, int tag, IdxShift S,
                     ENABLE_IF((traits::iform<TExpr>::value == EDGE))) {
        return getValue(expr, m, tag, S) * getValue(m.m_edge_volume_, m, tag, S);
    }
    template <typename TExpr>
    static auto getV(TExpr const& expr, mesh_type const& m, int tag, IdxShift S,
                     ENABLE_IF((traits::iform<TExpr>::value == FACE))) {
        return getValue(expr, m, tag, S) * getValue(m.m_face_volume_, m, tag, S);
    }
    template <typename TExpr>
    static auto getV(TExpr const& expr, mesh_type const& m, int tag, IdxShift S,
                     ENABLE_IF((traits::iform<TExpr>::value == VOLUME))) {
        return getValue(expr, m, tag, S) * getValue(m.m_volume_volume_, m, tag, S);
    }

    template <typename TExpr>
    static auto getDualV(TExpr const& expr, mesh_type const& m, int tag, IdxShift S,
                         ENABLE_IF((traits::iform<TExpr>::value == VERTEX))) {
        return getValue(expr, m, tag, S) * getValue(m.m_vertex_dual_volume_, m, tag, S);
    }
    template <typename TExpr>
    static auto getDualV(TExpr const& expr, mesh_type const& m, int tag, IdxShift S,
                         ENABLE_IF((traits::iform<TExpr>::value == EDGE))) {
        return getValue(expr, m, tag, S) * getValue(m.m_edge_dual_volume_, m, tag, S);
    }
    template <typename TExpr>
    static auto getDualV(TExpr const& expr, mesh_type const& m, int tag, IdxShift S,
                         ENABLE_IF((traits::iform<TExpr>::value == FACE))) {
        return getValue(expr, m, tag, S) * getValue(m.m_face_dual_volume_, m, tag, S);
    }
    template <typename TExpr>
    static auto getDualV(TExpr const& expr, mesh_type const& m, int tag, IdxShift S,
                         ENABLE_IF((traits::iform<TExpr>::value == VOLUME))) {
        return getValue(expr, m, tag, S) * getValue(m.m_volume_dual_volume_, m, tag, S);
    }
    //
    //
    //
    //
    //    template <typename TExpr>
    //    static auto getDualV( TExpr const& expr,mesh_type const& m, int tag, IdxShift S) {
    //        return getDualV_(m, std::integral_constant<int, traits::iform<TExpr>::value>(), expr, tag, S);
    //        //        return getValue( expr,m, tag, S) * Volume(m, tag, S);
    //    }

    //******************************************************************************
    // Exterior algebra
    //******************************************************************************

    //! grad<0>

    template <typename TExpr>
    static auto eval(Expression<tags::_exterior_derivative, TExpr> const& expr, mesh_type const& m, int tag, IdxShift S,
                     std::integer_sequence<int, VERTEX>) {
        auto const& l = std::get<0>(expr.m_args_);
        IdxShift D{0, 0, 0};
        D[EntityIdCoder::m_id_to_sub_index_[tag % 0b111]] = 1;
        return (getV(l, m, tag & (~0b111), S + D) - getV(l, m, tag & (~0b111), S)) *
               getValue(m.m_edge_inv_volume_, m, tag, S);
    }

    //! curl<1>

    template <typename TExpr>
    static auto eval(Expression<tags::_exterior_derivative, TExpr> const& expr, mesh_type const& m, int tag, IdxShift S,
                     std::integer_sequence<int, EDGE>) {
        auto const& l = std::get<0>(expr.m_args_);

        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};

        int n = EntityIdCoder::m_id_to_sub_index_[tag];

        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return ((getV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][(n + 1) % 3], S + Z) -
                 getV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][(n + 1) % 3], S)) -
                (getV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][(n + 2) % 3], S + Y) -
                 getV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][(n + 2) % 3], S))) *
               getValue(m.m_face_inv_volume_, m, tag, S);
    }

    //! div<2>
    template <typename TExpr>
    static auto eval(Expression<tags::_exterior_derivative, TExpr> const& expr, mesh_type const& m, int tag, IdxShift S,
                     std::integer_sequence<int, FACE>) {
        auto const& l = std::get<0>(expr.m_args_);

        IdxShift X{1, 0, 0};
        IdxShift Y{0, 1, 0};
        IdxShift Z{0, 0, 1};

        return ((getV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[FACE][0], S + X) -
                 getV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[FACE][0], S)) +  //
                (getV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[FACE][1], S + Y) -
                 getV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[FACE][1], S)) +  //
                (getV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[FACE][2], S + Z) -
                 getV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[FACE][2], S))  //
                ) *
               getValue(m.m_edge_inv_volume_, m, tag, S);
    }

    //! curl<2>
    template <typename TExpr>
    static auto eval(Expression<tags::_codifferential_derivative, TExpr> const& expr, mesh_type const& m, int tag,
                     IdxShift S, std::integer_sequence<int, FACE>) {
        auto const& l = std::get<0>(expr.m_args_);
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};

        int n = EntityIdCoder::m_id_to_sub_index_[tag % 0b111];

        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return ((getDualV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[FACE][(n + 1) % 3], S) -
                 getDualV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[FACE][(n + 1) % 3], S - Z)) -
                (getDualV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[FACE][(n + 2) % 3], S) -
                 getDualV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[FACE][(n + 2) % 3], S - Y))) *
               (-getValue(m.m_edge_inv_dual_volume_, m, tag, S));
    }

    //! div<1>

    template <typename TExpr>
    static auto eval(Expression<tags::_codifferential_derivative, TExpr> const& expr, mesh_type const& m, int tag,
                     IdxShift S, std::integer_sequence<int, EDGE>) {
        auto const& l = std::get<0>(expr.m_args_);

        IdxShift X{1, 0, 0};
        IdxShift Y{0, 1, 0};
        IdxShift Z{0, 0, 1};

        return ((getDualV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][0], S) -
                 getDualV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][0], S - X)) +  //
                (getDualV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][1], S) -
                 getDualV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][1], S - Y)) +  //
                (getDualV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][2], S) -
                 getDualV(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][2], S - Z))) *
               (-getValue(m.m_vertex_inv_dual_volume_, m, tag, S));

        ;
    }

    //! grad<3>

    template <typename TExpr>
    static auto eval(Expression<tags::_codifferential_derivative, TExpr> const& expr, mesh_type const& m, int tag,
                     IdxShift S, std::integer_sequence<int, VOLUME>) {
        auto const& l = std::get<0>(expr.m_args_);
        IdxShift D{0, 0, 0};
        D[EntityIdCoder::m_id_to_sub_index_[tag & 0b111]] = 1;

        return (getV(l, m, tag | 0b111, S) - getV(l, m, tag | 0b111, S - D)) *
               (-getValue(m.m_face_inv_volume_, m, tag, S));
    }

    //! *Form<IR> => Form<N-IL>

    template <typename TExpr>
    static auto eval(Expression<tags::_hodge_star, TExpr> const& expr, mesh_type const& m, int tag, IdxShift S,
                     std::integer_sequence<int, VERTEX>) {
        auto const& l = std::get<0>(expr.m_args_);
        auto id = tag & (~0b111);
        return (                                            //
                   getV(l, m, id, S) +                      //
                   getV(l, m, id, S + IdxShift{0, 0, 1}) +  //
                   getV(l, m, id, S + IdxShift{0, 1, 0}) +  //
                   getV(l, m, id, S + IdxShift{0, 1, 1}) +  //
                   getV(l, m, id, S + IdxShift{1, 0, 0}) +  //
                   getV(l, m, id, S + IdxShift{1, 0, 1}) +  //
                   getV(l, m, id, S + IdxShift{1, 1, 0}) +  //
                   getV(l, m, id, S + IdxShift{1, 1, 1})    //
                   ) *
               getValue(m.m_volume_inv_volume_, m, tag, S) * 0.125;
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
    //        return (get_v(m, std::Pop<0>(expr.m_args_), s + EntityIdCoder::DI(I)) -
    //                get_v(m, std::Pop<0>(expr.m_args_), s - EntityIdCoder::DI(I))
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
    //        return (get_v(m, std::Pop<0>(expr.m_args_), s + EntityIdCoder::DI(I)) -
    //                get_v(m, std::Pop<0>(expr.m_args_), s - EntityIdCoder::DI(I))
    //               ) * m.inv_dual_volume(s) * m_p_curl_factor_[(I + 3 -
    //               EntityIdCoder::sub_index(s)) % 3];
    //    }

    ////***************************************************************************************************
    //
    ////! map_to

    template <typename TExpr, int I>
    static auto _map_to(TExpr const& expr, mesh_type const& m, int tag, IdxShift S, std::index_sequence<I, I>) {
        return getValue(expr, m, tag, S);
    };

    template <typename TExpr>
    static auto _map_to(TExpr const& expr, mesh_type const& m, int tag, IdxShift S, std::index_sequence<VERTEX, EDGE>) {
        int n = EntityIdCoder::m_id_to_sub_index_[tag & 0b111];
        int id = n << 3;
        IdxShift D{0, 0, 0};
        D[n] = 1;

        return (getValue(expr, m, id, S) + getValue(expr, m, id, S + D)) * 0.5;
    }

    template <typename TExpr>
    static auto _map_to(TExpr const& expr, mesh_type const& m, int tag, IdxShift S, std::index_sequence<EDGE, VERTEX>) {
        IdxShift D{0, 0, 0};
        int n = (tag >> 3) % 3;
        int id = EntityIdCoder::m_sub_index_to_id_[EDGE][n];
        D[n] = 1;
        return (getValue(expr, m, id, S - D) + getValue(expr, m, id, S)) * 0.5;
    }

    template <typename TExpr>
    static auto _map_to(TExpr const& expr, mesh_type const& m, int tag, IdxShift S, std::index_sequence<VERTEX, FACE>) {
        int n = EntityIdCoder::m_id_to_sub_index_[tag % 0b111];
        int id = n << 3;
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;
        return (                                     //
                   getValue(expr, m, id, S) +        //
                   getValue(expr, m, id, S + Y) +    //
                   getValue(expr, m, id, S + Z) +    //
                   getValue(expr, m, id, S + Y + Z)  //
                   ) *
               0.25;
    }

    template <typename TExpr>
    static auto _map_to(TExpr const& expr, mesh_type const& m, int tag, IdxShift S, std::index_sequence<FACE, VERTEX>) {
        int n = (tag >> 3) % 3;
        int id = EntityIdCoder::m_sub_index_to_id_[FACE][n];

        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};

        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (getValue(expr, m, id, S - Y - Z) +  //
                getValue(expr, m, id, S - Y) +      //
                getValue(expr, m, id, S - Z) +      //
                getValue(expr, m, id, S)            //
                ) *
               0.25;
    }

    template <typename TExpr>
    static auto _map_to(TExpr const& expr, mesh_type const& m, int tag, IdxShift S,
                        std::index_sequence<VERTEX, VOLUME>) {
        int id = tag & (~0b111);
        return (                                                   //
                   getValue(expr, m, id, S + IdxShift{0, 0, 0}) +  //
                   getValue(expr, m, id, S + IdxShift{0, 0, 1}) +  //
                   getValue(expr, m, id, S + IdxShift{0, 1, 0}) +  //
                   getValue(expr, m, id, S + IdxShift{0, 1, 1}) +  //
                   getValue(expr, m, id, S + IdxShift{1, 0, 0}) +  //
                   getValue(expr, m, id, S + IdxShift{1, 0, 1}) +  //
                   getValue(expr, m, id, S + IdxShift{1, 1, 0}) +  //
                   getValue(expr, m, id, S + IdxShift{1, 1, 1})    //
                   ) *
               0.125;
    }

    template <typename TExpr>
    static auto _map_to(TExpr const& expr, mesh_type const& m, int tag, IdxShift S,
                        std::index_sequence<VOLUME, VERTEX>) {
        int id = tag | 0b111;
        return (                                                   //
                   getValue(expr, m, id, S - IdxShift{1, 1, 1}) +  //
                   getValue(expr, m, id, S - IdxShift{1, 1, 0}) +  //
                   getValue(expr, m, id, S - IdxShift{1, 0, 1}) +  //
                   getValue(expr, m, id, S - IdxShift{1, 0, 0}) +  //
                   getValue(expr, m, id, S - IdxShift{0, 1, 1}) +  //
                   getValue(expr, m, id, S - IdxShift{0, 1, 0}) +  //
                   getValue(expr, m, id, S - IdxShift{0, 0, 1}) +  //
                   getValue(expr, m, id, S)                        //
                   ) *
               0.125;
    }

    template <typename TExpr>
    static auto _map_to(TExpr const& expr, mesh_type const& m, int tag, IdxShift S, std::index_sequence<VOLUME, FACE>) {
        int n = EntityIdCoder::m_id_to_sub_index_[tag];
        int id = (n << 3) | 0b111;
        IdxShift D{0, 0, 0};
        D[n] = 1;
        return (getValue(expr, m, id, S - D) + getValue(expr, m, id, S)) * 0.5;
    }

    template <typename TExpr>
    static auto _map_to(TExpr const& expr, mesh_type const& m, int tag, IdxShift S, std::index_sequence<FACE, VOLUME>) {
        int n = (tag >> 3) % 3;
        int id = EntityIdCoder::m_sub_index_to_id_[FACE][n];
        IdxShift D{0, 0, 0};
        D[n] = 1;

        return (getValue(expr, m, id, S) + getValue(expr, m, id, S + D)) * 0.5;
    }

    template <typename TExpr>
    static auto _map_to(TExpr const& expr, mesh_type const& m, int tag, IdxShift S, std::index_sequence<VOLUME, EDGE>) {
        int n = EntityIdCoder::m_id_to_sub_index_[tag & 0b111];
        int id = (n << 3) | 0b111;
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (                                       //
                   getValue(expr, m, id, S - Y) +      //
                   getValue(expr, m, id, S - Z) +      //
                   getValue(expr, m, id, S - Y - Z) +  //
                   getValue(expr, m, id, S)            //
                   ) *
               0.25;
    }

    template <typename TExpr>
    static auto _map_to(TExpr const& expr, mesh_type const& m, int tag, IdxShift S, std::index_sequence<EDGE, VOLUME>) {
        int n = (tag >> 3) % 3;
        int id = EntityIdCoder::m_sub_index_to_id_[EDGE][n];

        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};

        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (getValue(expr, m, id, S) +        //
                getValue(expr, m, id, S + Z) +    //
                getValue(expr, m, id, S + Y) +    //
                getValue(expr, m, id, S + Y + Z)  //
                ) *
               0.25;
    }

    template <typename TExpr, int ISrc, int IDest>
    static auto eval(Expression<simpla::tags::_map_to<IDest>, TExpr> const& expr, mesh_type const& m, int tag,
                     IdxShift S, std::integer_sequence<int, ISrc>) {
        return _map_to(std::get<0>(expr.m_args_), m, tag, S, std::index_sequence<ISrc, IDest>());
    }
    //***************************************************************************************************
    //
    //! Form<IL> ^ Form<IR> => Form<IR+IL>
    template <typename... TExpr, int IL, int IR>
    static auto eval(Expression<simpla::tags::_wedge, TExpr...> const& expr, mesh_type const& m, int tag, IdxShift S,
                     std::integer_sequence<int, IL, IR>) {
        return m.inner_product(_map_to(std::get<0>(expr.m_args_), m, tag, S, std::index_sequence<IL, IR + IL>()),
                               _map_to(std::get<1>(expr.m_args_), m, tag, S, std::index_sequence<IR, IR + IL>()));
    }

    template <typename... TExpr>
    static auto eval(Expression<simpla::tags::_wedge, TExpr...> const& expr, mesh_type const& m, int tag, IdxShift S,
                     std::integer_sequence<int, EDGE, EDGE>) {
        int n = EntityIdCoder::m_id_to_sub_index_[tag];

        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (getValue(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][(n + 2) % 3], S) +  //
                getValue(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][(n + 2) % 3],
                         S + Y)) *                                                                            //
               (getValue(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][(n + 1) % 3], S) +  //
                getValue(l, m, (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][(n + 1) % 3],
                         S + Z)) *  //
               0.25;
    }

    template <typename... TExpr>
    static auto eval(Expression<simpla::tags::_wedge, TExpr...> const& expr, mesh_type const& m, int tag, IdxShift S,
                     std::integer_sequence<int, FACE, FACE>) {
        int n = EntityIdCoder::m_id_to_sub_index_[tag];

        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;
        return getValue(l, m, n, S + Z) * getValue(r, m, n, S + Y) -
               getValue(l, m, n, S + Z) * getValue(r, m, n, S + Y);
    }

    template <typename... TExpr>
    static auto eval(Expression<simpla::tags::_dot, TExpr...> const& expr, mesh_type const& m, int tag, IdxShift S,
                     std::integer_sequence<int, VERTEX, VERTEX>) {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        return getValue(l, m, (0 << 3), S) * getValue(r, m, (0 << 3), S) +
               getValue(l, m, (1 << 3), S) * getValue(r, m, (1 << 3), S) +
               getValue(l, m, (2 << 3), S) * getValue(r, m, (2 << 3), S);
    }

    template <typename... TExpr>
    static auto eval(Expression<simpla::tags::_dot, TExpr...> const& expr, mesh_type const& m, int tag, IdxShift S,
                     std::integer_sequence<int, VOLUME, VOLUME>) {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        return getValue(l, m, (0 << 3) | 0b111, S) * getValue(r, m, (0 << 3) | 0b111, S) +
               getValue(l, m, (1 << 3) | 0b111, S) * getValue(r, m, (1 << 3) | 0b111, S) +
               getValue(l, m, (2 << 3) | 0b111, S) * getValue(r, m, (2 << 3) | 0b111, S);
    }

    template <typename... TExpr>
    static auto eval(Expression<simpla::tags::_dot, TExpr...> const& expr, mesh_type const& m, int tag, IdxShift S,
                     std::integer_sequence<int, EDGE, EDGE>) {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        return eval(dot_v(map_to<VERTEX>(l), map_to<VERTEX>(r)), m, tag, S, std::integer_sequence<int, VERTEX>());
    }
    template <typename... TExpr>
    static auto eval(Expression<simpla::tags::_dot, TExpr...> const& expr, mesh_type const& m, int tag, IdxShift S,
                     std::integer_sequence<int, FACE, FACE>) {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);
        return eval(dot_v(map_to<VERTEX>(l), map_to<VERTEX>(r)), m, tag, S, std::integer_sequence<int, VERTEX>());
    }
    template <typename... TExpr>
    static auto eval(Expression<simpla::tags::_cross, TExpr...> const& expr, mesh_type const& m, int tag, IdxShift S,
                     std::integer_sequence<int, VERTEX, VERTEX>) {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        int n = tag >> 3;
        return getValue(l, m, ((n + 1) % 3) << 3, S) * getValue(r, m, ((n + 2) % 3) << 3, S) -
               getValue(l, m, ((n + 2) % 3) << 3, S) * getValue(r, m, ((n + 1) % 3) << 3, S);
    }

    template <typename... TExpr>
    static auto eval(Expression<simpla::tags::_cross, TExpr...> const& expr, mesh_type const& m, int tag, IdxShift S,
                     std::integer_sequence<int, VOLUME, VOLUME>) {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);
        int n = (tag >> 3) % 3;
        return getValue(l, m, 0b111 | (((n + 1) % 3) << 3), S) * getValue(r, m, 0b111 | (((n + 2) % 3) << 3), S) -
               getValue(l, m, 0b111 | (((n + 2) % 3) << 3), S) * getValue(r, m, 0b111 | (((n + 1) % 3) << 3), S);
    }

    ///*********************************************************************************************
    /// @name general_algebra General algebra
    /// @{

    template <typename V, int I, int... D>
    static V const& getValue(Field<TM, V, I, D...> const& f, mesh_type const& m, EntityId s) {
        return f[EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]][(s.w >> 3)](s.x, s.y, s.z);
    };
    template <typename V, int I, int... D>
    static V& getValue(Field<TM, V, I, D...>& f, mesh_type const& m, EntityId s) {
        return f[EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]][(s.w >> 3)](s.x, s.y, s.z);
    };

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

        point_type r;  //= std::Pop<1>(idx);
        EntityId s;    //= std::Pop<0>(idx);

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
    //    static auto getValue(mesh_type const& m, TFun const& fun, int tag, IdxShift S,
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
    //    static auto getValue(mesh_type const& m, TFun const& fun, int tag, IdxShift S,
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
    //    static auto  getValue(std::integral_constant<int, IFORM> const&,  TExpr const& expr,mesh_type const& m,
    //    index_type i,
    //                  index_type j, index_type k, unsigned int n, unsigned int d)  {
    //        return getValue( expr,m, EntityIdCoder::Pack<IFORM>(i, j, k, n, d));
    //    }
    //    template <typename TField, typename TOP, typename... Args>
    //    void foreach_(mesh_type const& m, TField& self, Range<EntityId> const& r, TOP const& op, Args&&... args) const
    //    {
    //        r.foreach ([&](EntityId s)  { op(getValue(m, self, s), getValue(m, std::forward<Args>(args), s)...);
    //        });
    //    }
    //    template <typename... Args>
    //    void foreach (Args&&... args)  {
    //        foreach_(std::forward<Args>(args)...);
    //    }
    //    template <typename TField, typename... Args>
    //    void foreach (mesh_type const& m, TField & self, mesh::MeshZoneTag const& tag, Args && ... args)  {
    //        foreach_(m, self, m.range(tag, traits::iform<TField>::value, traits::dof<TField>::value),
    //                 std::forward<Args>(args)...);
    //    }
    //    template <typename TField, typename... Args>
    //    void foreach (mesh_type const& m, TField & self, Args && ... args)  {
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

// template <typename TV, typename TM, int IFORM, int DOF>
// constexpr Real   calculator<Field<TV, TM, IFORM, DOF>>::m_p_curl_factor[3];
}  // namespace simpla { {

#endif /* FDM_H_ */
