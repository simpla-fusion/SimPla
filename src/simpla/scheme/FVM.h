/**
 * @file  calculate_fvm.h
 *
 *  Created on: 2014-9-23
 *      Author: salmon
 */

#ifndef FVM_H_
#define FVM_H_

#include <complex>
#include <cstddef>
#include <type_traits>

#include "simpla/algebra/Array.h"
#include "simpla/algebra/Calculus.h"
#include "simpla/algebra/EntityId.h"
#include "simpla/algebra/ExpressionTemplate.h"
#include "simpla/engine/Engine.h"
#include "simpla/utilities/type_traits.h"

namespace simpla {
namespace scheme {

namespace st = simpla::traits;

template <typename THost>
struct FVM {
    SP_ENGINE_POLICY_HEAD(FVM);

    typedef THost domain_type;
    static constexpr unsigned int NDIMS = 3;

    template <typename TM, typename TV, int... N>
    decltype(auto) getArray(Field<TM, TV, N...> const& v, IdxShift S, int tag) const {
        return st::recursive_index(v[EntityIdCoder::m_id_to_sub_index_[tag & 0b111]], tag << 3)(S);
    }

    template <size_t... I, typename TOP, typename... Args>
    decltype(auto) _invoke_helper(std::index_sequence<I...> _, Expression<TOP, Args...> const& expr, IdxShift S,
                                  int tag) const {
        return expr.m_op_(getArray(std::get<I>(expr.m_args_), S, tag)...);
    }

    template <int... I, typename TOP, typename... Args>
    decltype(auto) eval(std::integer_sequence<int, I...> _, Expression<TOP, Args...> const& expr, IdxShift S,
                        int tag) const {
        return _invoke_helper(std::make_index_sequence<sizeof...(I)>(), expr, S, tag);
    }

    template <typename TOP, typename... Args>
    decltype(auto) getArray(Expression<TOP, Args...> const& expr, IdxShift S, int tag) const {
        return eval(std::integer_sequence<int, st::iform<Args>::value...>(), expr, S, tag);
    }

   private:
    template <int N, typename TExpr>
    auto const& _getArray(std::integral_constant<int, N> _, TExpr const& expr, IdxShift S, int tag) const {
        return expr;
    }

    template <typename TExpr>
    auto _getArray(std::integral_constant<int, 0b00010> _, TExpr const& expr, IdxShift S, int tag) const {
        int n = ((tag & 0b111) == 0 || (tag & 0b111) == 0b111) ? (tag << 3)
                                                               : EntityIdCoder::m_id_to_sub_index_[tag & 0b111];
        return [=](index_type x, index_type y, index_type z) {
            return st::recursive_index(expr(m_host_->GetMesh()->local_coordinates(x + S[0], y + S[1], z + S[2], tag)),
                                       n);
        };
    }

    template <typename TExpr>
    auto _getArray(std::integral_constant<int, 0b00100> _, TExpr const& expr, IdxShift S, int tag) const {
        int n = ((tag & 0b111) == 0 || (tag & 0b111) == 0b111) ? (tag << 3)
                                                               : EntityIdCoder::m_id_to_sub_index_[tag & 0b111];
        return [=](index_type x, index_type y, index_type z) {
            return st::recursive_index(expr(x + S[0], y + S[1], z + S[2]), n);
        };
    }

    template <typename TExpr>
    auto _getArray(std::integral_constant<int, 0b01000> _, TExpr const& expr, IdxShift S, int tag) const {
        int n = ((tag & 0b111) == 0 || (tag & 0b111) == 0b111) ? (tag << 3)
                                                               : EntityIdCoder::m_id_to_sub_index_[tag & 0b111];
        return [=](index_type x, index_type y, index_type z) {
            return st::recursive_index(expr(x + S[0], y + S[1], z + S[2], tag), n);
        };
    }

   public:
    template <typename TExpr>
    auto getArray(TExpr const& expr, IdxShift S, int tag) const {
        return _getArray(
            std::integral_constant<
                int,
                (st::is_invocable<TExpr, point_type const&>::value /*                     */ ? 0b0010 : 0) |
                    (st::is_invocable<TExpr, index_type, index_type, index_type>::value /**/ ? 0b0100 : 0) |
                    (st::is_invocable<TExpr, index_type, index_type, index_type, int>::value ? 0b1000 : 0)>(),
            expr, S, tag);
    }

    template <typename M, typename U, int IFORM, int... N, typename RHS>
    void Calculate(Field<M, U, IFORM, N...>& lhs, RHS const& rhs) const {
        st::foreach (lhs.Get(),  //
                     [&](auto& a, int n0, auto&&... subs) {
                         auto tag = static_cast<int16_t>(
                             EntityIdCoder::m_sub_index_to_id_[IFORM][n0] |
                             (st::recursive_calculate_shift<1, N...>(0, std::forward<decltype(subs)>(subs)...) << 3));
                         a = getArray((rhs), IdxShift{0, 0, 0}, tag);
                     });
    }

    template <typename M, typename U, int IFORM, int... N, typename RHS>
    void Calculate(Field<M, U, IFORM, N...>& lhs, RHS const& rhs, Range<EntityId> const& r) const {
        if (r.isNull()) { return; }

        st::foreach (lhs.Get(),  //
                     [&](auto& a, int n0, auto&&... subs) {
                         auto tag = static_cast<int16_t>(
                             EntityIdCoder::m_sub_index_to_id_[IFORM][n0] |
                             (st::recursive_calculate_shift<1, N...>(0, std::forward<decltype(subs)>(subs)...) << 3));

                         int n = ((tag & 0b111) == 0 || (tag & 0b111) == 0b111)
                                     ? (tag << 3)
                                     : EntityIdCoder::m_id_to_sub_index_[tag & 0b111];

                         r.foreach ([&](EntityId s) {
                             if (s.w == tag) {
                                 a(s.x, s.y, s.z) = st::recursive_index(
                                     calculus::getValue(getArray((rhs), IdxShift{0, 0, 0}, tag), s.x, s.y, s.z), n);
                             }
                         });
                     });
    }

    auto _getV(std::integral_constant<int, VERTEX> _, IdxShift S, int tag) const {
        return getArray(m_host_->m_vertex_volume_, S, tag);
    }

    auto _getV(std::integral_constant<int, EDGE> _, IdxShift S, int tag) const {
        return getArray(m_host_->m_edge_volume_, S, tag);
    }

    auto _getV(std::integral_constant<int, FACE> _, IdxShift S, int tag) const {
        return getArray(m_host_->m_face_volume_, S, tag);
    }

    auto _getV(std::integral_constant<int, VOLUME> _, IdxShift S, int tag) const {
        return getArray(m_host_->m_volume_volume_, S, tag);
    }

    auto _getDualV(std::integral_constant<int, VERTEX> _, IdxShift S, int tag) const {
        return getArray(m_host_->m_vertex_dual_volume_, S, tag);
    }

    auto _getDualV(std::integral_constant<int, EDGE> _, IdxShift S, int tag) const {
        return getArray(m_host_->m_edge_dual_volume_, S, tag);
    }

    auto _getDualV(std::integral_constant<int, FACE> _, IdxShift S, int tag) const {
        return getArray(m_host_->m_face_dual_volume_, S, tag);
    }

    auto _getDualV(std::integral_constant<int, VOLUME> _, IdxShift S, int tag) const {
        return getArray(m_host_->m_volume_dual_volume_, S, tag);
    }

    template <typename TExpr>
    auto getDualV(TExpr const& expr, IdxShift S, int tag) const {
        return getArray(expr, S, tag) * _getDualV(std::integral_constant<int, st::iform<TExpr>::value>(), S, tag);
    }
    template <typename TExpr>
    auto getV(TExpr const& expr, IdxShift S, int tag) const {
        return getArray(expr, S, tag) * _getV(std::integral_constant<int, st::iform<TExpr>::value>(), S, tag);
    }

    //******************************************************************************
    // Exterior algebra
    //******************************************************************************

    //! grad<0>

    template <typename TExpr>
    auto eval(std::integer_sequence<int, VERTEX> _, Expression<tags::exterior_derivative, TExpr> const& expr,
              IdxShift S, int tag) const {
        auto const& l = std::get<0>(expr.m_args_);
        IdxShift D{0, 0, 0};
        int n = EntityIdCoder::m_id_to_sub_index_[tag & 0b111];
        D[n] = 1;
        return (getV(l, S + D, tag & (~0b111)) - getV(l, S, tag & (~0b111))) *
               getArray(m_host_->m_edge_inv_volume_, S, tag);
    }

    //! curl<1>

    template <typename TExpr>
    auto eval(std::integer_sequence<int, EDGE> _, Expression<tags::exterior_derivative, TExpr> const& expr, IdxShift S,
              int tag) const {
        auto const& l = std::get<0>(expr.m_args_);

        int n = EntityIdCoder::m_id_to_sub_index_[tag & 0b111];
        int IX = (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][(n + 0) % 3];
        int IY = (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][(n + 1) % 3];
        int IZ = (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][(n + 2) % 3];

        IdxShift SY{0, 0, 0};
        IdxShift SZ{0, 0, 0};
        SY[(n + 1) % 3] = 1;
        SZ[(n + 2) % 3] = 1;

        return ((getV(l, S + SZ, IY) - getV(l, S, IY)) - (getV(l, S + SY, IZ) - getV(l, S, IZ))) *
               getArray(m_host_->m_face_inv_volume_, S, tag);
    }

    //! div<2>
    template <typename TExpr>
    auto eval(std::integer_sequence<int, FACE> _, Expression<tags::exterior_derivative, TExpr> const& expr, IdxShift S,
              int tag) const {
        auto const& l = std::get<0>(expr.m_args_);

        IdxShift SX{1, 0, 0};
        IdxShift SY{0, 1, 0};
        IdxShift SZ{0, 0, 1};

        int IX = (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[FACE][0];
        int IY = (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[FACE][1];
        int IZ = (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[FACE][2];

        return ((getV(l, S + SX, IX) - getV(l, S, IX)) + (getV(l, S + SY, IY) - getV(l, S, IY)) +
                (getV(l, S + SZ, IZ) - getV(l, S, IZ))) *
               getArray(m_host_->m_volume_inv_volume_, S, tag);
    }

    //! curl<2>
    template <typename TExpr>
    auto eval(std::integer_sequence<int, FACE> _, Expression<tags::codifferential_derivative, TExpr> const& expr,
              IdxShift S, int tag) const {
        auto const& l = std::get<0>(expr.m_args_);

        int n = EntityIdCoder::m_id_to_sub_index_[tag & 0b111];

        int IX = (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[FACE][(n + 0) % 3];
        int IY = (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[FACE][(n + 1) % 3];
        int IZ = (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[FACE][(n + 2) % 3];
        IdxShift SY{0, 0, 0};
        IdxShift SZ{0, 0, 0};
        SY[(n + 1) % 3] = 1;
        SZ[(n + 2) % 3] = 1;

        return ((getDualV(l, S, IY) - getDualV(l, S - SZ, IY)) - (getDualV(l, S, IZ) - getDualV(l, S - SY, IZ))) *
               (-getArray(m_host_->m_edge_inv_dual_volume_, S, tag));
    }

    //! div<1>

    template <typename TExpr>
    auto eval(std::integer_sequence<int, EDGE> _, Expression<tags::codifferential_derivative, TExpr> const& expr,
              IdxShift S, int tag) const {
        auto const& l = std::get<0>(expr.m_args_);
        int n = EntityIdCoder::m_id_to_sub_index_[tag & 0b111];

        IdxShift SX{1, 0, 0};
        IdxShift SY{0, 1, 0};
        IdxShift SZ{0, 0, 1};

        int IX = (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][0];
        int IY = (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][1];
        int IZ = (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][2];

        return ((getDualV(l, S, IX) - getDualV(l, S - SX, IX)) +  //
                (getDualV(l, S, IY) - getDualV(l, S - SY, IY)) +  //
                (getDualV(l, S, IZ) - getDualV(l, S - SZ, IZ))) *
               (-getArray(m_host_->m_vertex_inv_dual_volume_, S, tag));

        ;
    }

    //! grad<3>

    template <typename TExpr>
    auto eval(std::integer_sequence<int, VOLUME> _, Expression<tags::codifferential_derivative, TExpr> const& expr,
              IdxShift S, int tag) const {
        auto const& l = std::get<0>(expr.m_args_);
        int n = EntityIdCoder::m_id_to_sub_index_[tag & 0b111];
        IdxShift SD{0, 0, 0};
        SD[n] = 1;
        int ID = (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[VOLUME][n];

        return (getV(l, S, ID) - getV(l, S - SD, ID)) * (-getArray(m_host_->m_face_inv_volume_, S, tag));
    }

    //! *Form<IR> => Form<N-IL>

    template <typename TExpr>
    auto eval(std::integer_sequence<int, VERTEX> _, Expression<tags::hodge_star, TExpr> const& expr, IdxShift S,
              int tag) const {
        auto const& l = std::get<0>(expr.m_args_);
        int I0 = tag & (~0b111);
        return (getV(l, S + IdxShift{0, 0, 0}, I0) + getV(l, S + IdxShift{0, 0, 1}, I0) +
                getV(l, S + IdxShift{0, 1, 0}, I0) + getV(l, S + IdxShift{0, 1, 1}, I0) +
                getV(l, S + IdxShift{1, 0, 0}, I0) + getV(l, S + IdxShift{1, 0, 1}, I0) +
                getV(l, S + IdxShift{1, 1, 0}, I0) + getV(l, S + IdxShift{1, 1, 1}, I0)) *
               getArray(m_host_->m_volume_inv_volume_, S, tag) * 0.125;
    };
    ////***************************************************************************************************
    //! p_curl<1>
    //     constexpr Real m_p_curl_factor_[3] = {0, 1, -1};
    //    template<typename TOP, typename T>   st::value_type_t
    //    <Expression<TOP, T>>
    //    GetValue(mesh_type const &Expression<TOP, T> const &expr,
    //    EntityId const &s,
    //    ENABLE_IF((std::is_same<TOP, tags::p_exterior_derivative < 0>>
    //                      ::value && st::GetIFORM<T>::value == EDGE))
    //    )
    //    {
    //        return (get_v(std::Serialize<0>(expr.m_args_), s + EntityIdCoder::DI(I)) -
    //                get_v(std::Serialize<0>(expr.m_args_), s - EntityIdCoder::DI(I))
    //               ) * m_host_->inv_volume(s) * m_p_curl_factor_[(I + 3 -
    //               EntityIdCoder::sub_index(s)) % 3];
    //    }
    //
    //
    //    template<typename T, size_t I>
    //      st::value_type_t
    //    <Expression<tags::p_codifferential_derivative < I>, T>>
    //    GetValue(
    //    mesh_type const &m,
    //    Expression<tags::p_codifferential_derivative < I>, T
    //    > const &expr,
    //    EntityId const &s,
    //    ENABLE_IF(st::GetIFORM<T>::value == FACE)
    //    )
    //    {
    //
    //        return (get_v(std::Serialize<0>(expr.m_args_), s + EntityIdCoder::DI(I)) -
    //                get_v(std::Serialize<0>(expr.m_args_), s - EntityIdCoder::DI(I))
    //               ) * m_host_->inv_dual_volume(s) * m_p_curl_factor_[(I + 3 -
    //               EntityIdCoder::sub_index(s)) % 3];
    //    }

    ////***************************************************************************************************
    ////! map_to

    template <int I, typename TExpr>
    auto _map_to(std::index_sequence<I, I> _, TExpr const& expr, IdxShift S, int tag) const {
        return getArray(expr, S, tag);
    };

    template <typename TExpr>
    auto _map_to(std::index_sequence<VERTEX, EDGE> _, TExpr const& expr, IdxShift S, int tag) const {
        int n = EntityIdCoder::m_id_to_sub_index_[tag & 0b111];
        int IX = (((tag << 3) * 3) + n) >> 3;
        IdxShift SX{0, 0, 0};
        SX[n] = 1;

        return (getArray(expr, S, IX) + getArray(expr, S + SX, IX)) * 0.5;
    }

    template <typename TExpr>
    auto _map_to(std::index_sequence<EDGE, VERTEX> _, TExpr const& expr, IdxShift S, int tag) const {
        int n = (tag << 3) % 3;
        int IX = (((tag << 3) / 3) >> 3) | EntityIdCoder::m_sub_index_to_id_[EDGE][n];
        IdxShift SX{0, 0, 0};
        SX[n] = 1;
        return (getArray(expr, S - SX, IX) + getArray(expr, S, IX)) * 0.5;
    }

    template <typename TExpr>
    auto _map_to(std::index_sequence<VERTEX, FACE> _, TExpr const& expr, IdxShift S, int tag) const {
        int n = EntityIdCoder::m_id_to_sub_index_[tag & 0b111];
        int IX = (((tag << 3) * 3 + n) >> 3);
        IdxShift SY{0, 0, 0};
        IdxShift SZ{0, 0, 0};
        SY[(n + 1) % 3] = 1;
        SZ[(n + 2) % 3] = 1;
        return (getArray(expr, S, IX) + getArray(expr, S + SY, IX) + getArray(expr, S + SZ, IX) +
                getArray(expr, S + SY + SZ, IX)) *
               0.25;
    }

    template <typename TExpr>
    auto _map_to(std::index_sequence<FACE, VERTEX> _, TExpr const& expr, IdxShift S, int tag) const {
        int n = (tag << 3) % 3;
        int IX = (((tag << 3) / 3) >> 3) | EntityIdCoder::m_sub_index_to_id_[FACE][(n + 0) % 3];
        IdxShift SY{0, 0, 0};
        IdxShift SZ{0, 0, 0};
        SY[(n + 1) % 3] = 1;
        SZ[(n + 2) % 3] = 1;

        return (getArray(expr, S - SY - SZ, IX) + getArray(expr, S - SY, IX) + getArray(expr, S - SZ, IX) +
                getArray(expr, S, IX)) *
               0.25;
    }

    template <typename TExpr>
    auto _map_to(std::index_sequence<VERTEX, VOLUME> _, TExpr const& expr, IdxShift S, int tag) const {
        tag = tag & (~0b111);
        return (getArray(expr, S + IdxShift{0, 0, 0}, tag) + getArray(expr, S + IdxShift{0, 0, 1}, tag) +
                getArray(expr, S + IdxShift{0, 1, 0}, tag) + getArray(expr, S + IdxShift{0, 1, 1}, tag) +
                getArray(expr, S + IdxShift{1, 0, 0}, tag) + getArray(expr, S + IdxShift{1, 0, 1}, tag) +
                getArray(expr, S + IdxShift{1, 1, 0}, tag) + getArray(expr, S + IdxShift{1, 1, 1}, tag)) *
               0.125;
    }

    template <typename TExpr>
    auto _map_to(std::index_sequence<VOLUME, VERTEX> _, TExpr const& expr, IdxShift S, int tag) const {
        tag = tag | (0b111);
        return (getArray(expr, S - IdxShift{1, 1, 1}, tag) + getArray(expr, S - IdxShift{1, 1, 0}, tag) +
                getArray(expr, S - IdxShift{1, 0, 1}, tag) + getArray(expr, S - IdxShift{1, 0, 0}, tag) +
                getArray(expr, S - IdxShift{0, 1, 1}, tag) + getArray(expr, S - IdxShift{0, 1, 0}, tag) +
                getArray(expr, S - IdxShift{0, 0, 1}, tag) + getArray(expr, S, tag)) *
               0.125;
    }

    template <typename TExpr>
    auto _map_to(std::index_sequence<VOLUME, FACE> _, TExpr const& expr, IdxShift S, int tag) const {
        int n = EntityIdCoder::m_id_to_sub_index_[tag & 0b111];
        int IX = 0b111 | (((n << 3) * 3 + n) >> 3);

        IdxShift SD{0, 0, 0};
        SD[n] = 1;
        return (getArray(expr, S - SD, IX) + getArray(expr, S, IX)) * 0.5;
    }

    template <typename TExpr>
    auto _map_to(std::index_sequence<FACE, VOLUME> _, TExpr const& expr, IdxShift S, int tag) const {
        int n = (tag << 3) % 3;
        int IX = (((tag << 3) / 3) >> 3) | EntityIdCoder::m_sub_index_to_id_[FACE][n];
        IdxShift SX{0, 0, 0};
        SX[n] = 1;
        return (getArray(expr, S, IX) + getArray(expr, S + SX, IX)) * 0.5;
    }

    template <typename TExpr>
    auto _map_to(std::index_sequence<VOLUME, EDGE> _, TExpr const& expr, IdxShift S, int tag) const {
        int n = EntityIdCoder::m_id_to_sub_index_[tag & 0b111];
        int IX = 0b111 | (((n >> 3) * 3 + n) << 3);

        IdxShift SY{0, 0, 0};
        IdxShift SZ{0, 0, 0};
        SY[(n + 1) % 3] = 1;
        SZ[(n + 2) % 3] = 1;

        return (getArray(expr, S - SY, IX) + getArray(expr, S - SZ, IX) + getArray(expr, S - SY - SZ, IX) +
                getArray(expr, S, IX)) *
               0.25;
    }

    template <typename TExpr>
    auto _map_to(std::index_sequence<EDGE, VOLUME> _, TExpr const& expr, IdxShift S, int tag) const {
        int n = (tag << 3) % 3;
        int IX = (((tag << 3) / 3) >> 3) | EntityIdCoder::m_sub_index_to_id_[EDGE][n];

        IdxShift SY{0, 0, 0};
        IdxShift SZ{0, 0, 0};
        SY[(n + 1) % 3] = 1;
        SZ[(n + 2) % 3] = 1;

        return (getArray(expr, S, IX) + getArray(expr, S + SZ, IX) + getArray(expr, S + SY, IX) +
                getArray(expr, S + SY + SZ, IX)) *
               0.25;
    }

    template <typename TExpr, int ISrc, int IDest>
    auto eval(std::integer_sequence<int, ISrc> _, Expression<tags::map_to<IDest>, TExpr> const& expr, IdxShift S,
              int tag) const {
        return _map_to(std::index_sequence<ISrc, IDest>(), std::get<0>(expr.m_args_), S, tag);
    }
    //***************************************************************************************************
    //
    //! Form<IL> ^ Form<IR> => Form<IR+IL>
    template <typename... TExpr, int IL, int IR>
    auto eval(std::integer_sequence<int, IL, IR> _, Expression<tags::wedge, TExpr...> const& expr, IdxShift S,
              int tag) const {
        FIXME;
        return m_host_->inner_product(_map_to(std::index_sequence<IL, IR + IL>(), std::get<0>(expr.m_args_), S, tag),
                                      _map_to(std::index_sequence<IR, IR + IL>(), std::get<1>(expr.m_args_), S, tag));
    }

    template <typename... TExpr>
    auto eval(std::integer_sequence<int, EDGE, EDGE> _, Expression<tags::wedge, TExpr...> const& expr, IdxShift S,
              int tag) const {
        // FIXME: only correct for Cartesian coordinates
        FIXME;

        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);
        int n = EntityIdCoder::m_id_to_sub_index_[tag & 0b111];

        int IX = (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][(n + 0) % 3];
        int IY = (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][(n + 1) % 3];
        int IZ = (tag & (~0b111)) | EntityIdCoder::m_sub_index_to_id_[EDGE][(n + 2) % 3];

        IdxShift SY{0, 0, 0};
        IdxShift SZ{0, 0, 0};
        SY[(n + 1) % 3] = 1;
        SZ[(n + 2) % 3] = 1;

        return (getArray(l, S, IY) + getArray(l, S + SZ, IY)) * (getArray(r, S, IY) + getArray(r, S + SY, IZ)) * 0.25 -
               (getArray(r, S, IY) + getArray(r, S + SZ, IY)) * (getArray(l, S, IY) + getArray(l, S + SY, IZ)) * 0.25;
    }

    template <typename... TExpr>
    auto eval(std::integer_sequence<int, EDGE, FACE> _, Expression<tags::wedge, TExpr...> const& expr, IdxShift S,
              int tag) const {
        FIXME;
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        int IX = (~0b111) | (((tag << 3) * 3 + 0) >> 3);
        int IY = (~0b111) | (((tag << 3) * 3 + 1) >> 3);
        int IZ = (~0b111) | (((tag << 3) * 3 + 2) >> 3);

        return _map_to(std::integer_sequence<int, EDGE, VOLUME>(), l, S, IX) *
                   _map_to(std::integer_sequence<int, FACE, VOLUME>(), r, S, IX) +
               _map_to(std::integer_sequence<int, EDGE, VOLUME>(), l, S, IY) *
                   _map_to(std::integer_sequence<int, FACE, VOLUME>(), r, S, IY) +
               _map_to(std::integer_sequence<int, EDGE, VOLUME>(), l, S, IZ) *
                   _map_to(std::integer_sequence<int, FACE, VOLUME>(), r, S, IZ);
    }

    template <typename... TExpr>
    auto eval(std::integer_sequence<int, FACE, EDGE> _, Expression<tags::wedge, TExpr...> const& expr, IdxShift S,
              int tag) const {
        FIXME;
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        int IX = (~0b111) | (((tag << 3) * 3 + 0) >> 3);
        int IY = (~0b111) | (((tag << 3) * 3 + 1) >> 3);
        int IZ = (~0b111) | (((tag << 3) * 3 + 2) >> 3);

        return _map_to(std::integer_sequence<int, FACE, VOLUME>(), l, S, IX) *
                   _map_to(std::integer_sequence<int, EDGE, VOLUME>(), r, S, IX) +
               _map_to(std::integer_sequence<int, FACE, VOLUME>(), l, S, IY) *
                   _map_to(std::integer_sequence<int, EDGE, VOLUME>(), r, S, IY) +
               _map_to(std::integer_sequence<int, FACE, VOLUME>(), l, S, IZ) *
                   _map_to(std::integer_sequence<int, EDGE, VOLUME>(), r, S, IZ);
    }

    template <typename... TExpr>
    auto eval(std::integer_sequence<int, EDGE, EDGE> _, Expression<tags::dot, TExpr...> const& expr, IdxShift S,
              int tag) const {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);
        return getArray(wedge(l, hodgestar(r)), S, tag);
    }
    template <typename... TExpr>
    auto eval(std::integer_sequence<int, FACE, FACE> _, Expression<tags::dot, TExpr...> const& expr, IdxShift S,
              int tag) const {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);
        return getArray(wedge(l, hodgestar(r)), S, tag);
    }

    template <typename... TExpr>
    auto eval(std::integer_sequence<int, VERTEX, VERTEX> _, Expression<tags::dot, TExpr...> const& expr, IdxShift S,
              int tag) const {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        int IX = ((tag << 3) * 3 + 0) >> 3;
        int IY = ((tag << 3) * 3 + 1) >> 3;
        int IZ = ((tag << 3) * 3 + 2) >> 3;

        return getArray(l, S, IX) * getArray(r, S, IX) + getArray(l, S, IY) * getArray(r, S, IY) +
               getArray(l, S, IZ) * getArray(r, S, IZ);
    }

    template <typename... TExpr>
    auto eval(std::integer_sequence<int, VOLUME, VOLUME> _, Expression<tags::dot, TExpr...> const& expr, IdxShift S,
              int tag) const {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);
        int IX = (((tag << 3) * 3 + 0) >> 3) | 0b111;
        int IY = (((tag << 3) * 3 + 1) >> 3) | 0b111;
        int IZ = (((tag << 3) * 3 + 2) >> 3) | 0b111;

        return getArray(l, S, IX) * getArray(r, S, IX) + getArray(l, S, IY) * getArray(r, S, IY) +
               getArray(l, S, IZ) * getArray(r, S, IZ);
    }

    template <typename... TExpr>
    auto eval(std::integer_sequence<int, VERTEX, VERTEX> _, Expression<tags::cross, TExpr...> const& expr, IdxShift S,
              int tag) const {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);
        int n = (tag << 3) % 3;

        int IX = ((((tag << 3) / 3) * 3 + (n + 0) % 3) >> 3);
        int IY = ((((tag << 3) / 3) * 3 + (n + 1) % 3) >> 3);
        int IZ = ((((tag << 3) / 3) * 3 + (n + 2) % 3) >> 3);

        return getArray(l, S, IY) * getArray(r, S, IZ) - getArray(l, S, IZ) * getArray(r, S, IY);
    }

    template <typename... TExpr>
    auto eval(std::integer_sequence<int, VOLUME, VOLUME> _, Expression<tags::cross, TExpr...> const& expr, IdxShift S,
              int tag) const {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);
        int n = (tag << 3) % 3;

        int IX = ((((tag << 3) / 3) * 3 + (n + 0) % 3) >> 3) | 0b111;
        int IY = ((((tag << 3) / 3) * 3 + (n + 1) % 3) >> 3) | 0b111;
        int IZ = ((((tag << 3) / 3) * 3 + (n + 2) % 3) >> 3) | 0b111;

        return getArray(l, S, IY) * getArray(r, S, IZ) - getArray(l, S, IZ) * getArray(r, S, IY);
    }
};  // class FVM
//    //**********************************************************************************************
//    // for element-wise arithmetic operation
//
//    template <typename T, typename... Args>
//    T& GetEntity(T& rhs, int tag, int tag) const {
//        return rhs;
//    }
//
//    template <typename... E, typename... Args>
//    auto GetEntity(Expression<E...> const& rhs, int tag, int tag) const {
//        return GetEntity(calculus::getArray(rhs, std::forward<Args>(args)...), tag);
//    }
//
//    template <typename V, int N0, int... N, typename... Args>
//    auto const& GetEntity(nTuple<V, N0, N...> const& rhs, int tag, int tag) const {
//        return calculus::getArray(
//            st::recursive_index<N...>(rhs[EntityIdCoder::m_id_to_sub_index_[tag & 0b111]], tag >> 3),
//            std::forward<Args>(args)...);
//    }
//    template <typename V, int N0, int... N, typename... Args>
//    auto& GetEntity(nTuple<V, N0, N...>& rhs, int tag, int tag) const {
//        return calculus::getArray(
//            st::recursive_index<N...>(rhs[EntityIdCoder::m_id_to_sub_index_[tag & 0b111]], tag >> 3),
//            std::forward<Args>(args)...);
//    }
//
//    template <typename EXPR, typename... Args>
//    auto GetEntity(EXPR const& expr, int tag, int tag) const {
//        return GetEntity(expr(m_host_->local_coordinate(x, y, z, tag)), tag);
//    }
//
//    template <typename EXPR>
//    auto GetEntity(EXPR const& rhs, EntityId s,
//                   ENABLE_IF((!st::is_invocable<EXPR, EntityId>::value))) const {
//        return GetEntity(rhs, static_cast<int>(s.w & 0b111), static_cast<index_type>(s.x),
//        static_cast<index_type>(s.y),
//                         static_cast<index_type>(s.z));
//    }
//
//    template <int IFORM, typename EXPR>
//    auto GetEntity(EXPR const& rhs, EntityId s,
//                   ENABLE_IF((st::is_invocable<EXPR, EntityId>::value))) const {
//        return GetEntity<IFORM>(rhs(s), static_cast<int>(s.w & 0b111));
//    }
//
//    template <typename U, int IFORM, int... N, typename RHS>
//    void Fill(Field<THost, U, IFORM, N...>& lhs, RHS const& rhs, ENABLE_IF((std::is_arithmetic<RHS>::value)))
//    const {
//        lhs.Get() = rhs;
//        st::foreach (lhs.Get(), [&](auto& a, auto&&... subs) {
//            a = getArray((rhs), IdxShift{0, 0, 0}, std::forward<decltype(subs)>(subs)...);
//        });
//    }
//
//    template <typename U, int... N, typename RHS>
//    void Fill(Field<THost, U, VERTEX, N...>& lhs, nTuple<RHS, N...> const& rhs) const {
//        lhs.Get()[0] = rhs;
//    }
//
//    template <typename U, int... N, typename RHS>
//    void Fill(Field<THost, U, VOLUME, N...>& lhs, nTuple<RHS, N...> const& rhs) const {
//        lhs.Get()[0] = rhs;
//    }
//
//    template <typename U, int... N, typename RHS>
//    void Fill(Field<THost, U, EDGE, N...>& lhs, nTuple<RHS, 3, N...> const& rhs) const {
//        lhs.Get() = rhs;
//    }
//
//    template <typename U, int... N, typename RHS>
//    void Fill(Field<THost, U, FACE, N...>& lhs, nTuple<RHS, 3, N...> const& rhs) const {
//        lhs.Get() = rhs;
//    }
//
//    template <typename U, int... NL, typename V, int... NR>
//    void Fill(Field<THost, U, NL...>& lhs, Field<THost, V, NR...> const& rhs) const {
//        lhs.Get() = rhs;
//    }

//    ///*********************************************************************************************
//    /// @name general_algebra General algebra
//    /// @{
//
//    //    template <typename V, int I, int... D>
//    //     V const& getArray( Field<M, V, I, D...> const& f, EntityId s)const{
//    //        return f[EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]][(s.w >> 3)](s.x, s.y, s.z);
//    //    };
//    //    template <typename V, int I, int... D>
//    //     V& getArray( Field<M, V, I, D...>& f, EntityId s)const{
//    //        return f[EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]][(s.w >> 3)](s.x, s.y, s.z);
//    //    };
//
//    ///*********************************************************************************************
//    /**
//     * @ingroup interpolate
//     * @brief basic linear interpolate
//     */
//    template <typename TD, typename TIDX>
//     auto gather_impl_( TD const& f, TIDX const& idx)const{
//        EntityId X = (EntityIdCoder::_DI);
//        EntityId Y = (EntityIdCoder::_DJ);
//        EntityId Z = (EntityIdCoder::_DK);
//
//        point_type r;  //= std::Serialize<1>(idx);
//        EntityId s;    //= std::Serialize<0>(idx);
//
//        return getArray(f, ((s + X) + Y) + Z) * (r[0]) * (r[1]) * (r[2]) +    //
//               getArray(f, (s + X) + Y) * (r[0]) * (r[1]) * (1.0 - r[2]) +    //
//               getArray(f, (s + X) + Z) * (r[0]) * (1.0 - r[1]) * (r[2]) +    //
//               getArray(f, (s + X)) * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]) +  //
//               getArray(f, (s + Y) + Z) * (1.0 - r[0]) * (r[1]) * (r[2]) +    //
//               getArray(f, (s + Y)) * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]) +  //
//               getArray(f, s + Z) * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]) +    //
//               getArray(f, s) * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
//    }
//
//   public:
//    template <typename TF>
//    constexpr  auto gather( TF const& f, point_type const& r,
//                                 ENABLE_IF((st::iform<TF>::value == VERTEX)))const{
//        return gather_impl_(f, m_host_->point_global_to_local(r, 0));
//    }
//
//    template <typename TF>
//    constexpr  auto gather( TF const& f, point_type const& r,
//                                 ENABLE_IF((st::iform<TF>::value == EDGE)))const{
//        return TF::field_value_type{gather_impl_(f, m_host_->point_global_to_local(r, 1)),
//                                    gather_impl_(f, m_host_->point_global_to_local(r, 2)),
//                                    gather_impl_(f, m_host_->point_global_to_local(r, 4))};
//    }
//
//    template <typename TF>
//    constexpr  auto gather( TF const& f, point_type const& r,
//                                 ENABLE_IF((st::iform<TF>::value == FACE)))const{
//        return TF::field_value_type{gather_impl_(f, m_host_->point_global_to_local(r, 6)),
//                                    gather_impl_(f, m_host_->point_global_to_local(r, 5)),
//                                    gather_impl_(f, m_host_->point_global_to_local(r, 3))};
//    }
//
//    template <typename TF>
//    constexpr  auto gather( TF const& f, point_type const& x,
//                                 ENABLE_IF((st::iform<TF>::value == VOLUME)))const{
//        return gather_impl_(f, m_host_->point_global_to_local(x, 7));
//    }
//
//    template <typename TF, typename IDX, typename TV>
//     void scatter_impl_( TF& f, IDX const& idx, TV const& v)const{
//        EntityId X = (EntityIdCoder::_DI);
//        EntityId Y = (EntityIdCoder::_DJ);
//        EntityId Z = (EntityIdCoder::_DK);
//
//        point_type r = std::get<1>(idx);
//        EntityId s = std::get<0>(idx);
//
//        getArray(f, ((s + X) + Y) + Z) += v * (r[0]) * (r[1]) * (r[2]);
//        getArray(f, (s + X) + Y) += v * (r[0]) * (r[1]) * (1.0 - r[2]);
//        getArray(f, (s + X) + Z) += v * (r[0]) * (1.0 - r[1]) * (r[2]);
//        getArray(f, s + X) += v * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
//        getArray(f, (s + Y) + Z) += v * (1.0 - r[0]) * (r[1]) * (r[2]);
//        getArray(f, s + Y) += v * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]);
//        getArray(f, s + Z) += v * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]);
//        getArray(f, s) += v * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
//    }
//
//    template <typename TF, typename TX, typename TV>
//     void scatter_( std::integral_constant<int, VERTEX>_, TF& f, TX const& x, TV const& u)
//    {
//        scatter_impl_(f, m_host_->point_global_to_local(x, 0), u);
//    }
//
//    template <typename TF, typename TX, typename TV>
//     void scatter_( std::integral_constant<int, EDGE>_, TF& f, TX const& x, TV const& u)const{
//        scatter_impl_(f, m_host_->point_global_to_local(x, 1), u[0]);
//        scatter_impl_(f, m_host_->point_global_to_local(x, 2), u[1]);
//        scatter_impl_(f, m_host_->point_global_to_local(x, 4), u[2]);
//    }
//
//    template <typename TF, typename TX, typename TV>
//     void scatter_( std::integral_constant<int, FACE>_, TF& f, TX const& x, TV const& u)const{
//        scatter_impl_(f, m_host_->point_global_to_local(x, 6), u[0]);
//        scatter_impl_(f, m_host_->point_global_to_local(x, 5), u[1]);
//        scatter_impl_(f, m_host_->point_global_to_local(x, 3), u[2]);
//    }
//
//    template <typename TF, typename TX, typename TV>
//     void scatter_( std::integral_constant<int, VOLUME>_, TF& f, TX const& x, TV const& u)
//    {
//        scatter_impl_(f, m_host_->point_global_to_local(x, 7), u);
//    }
//
//   public:
//    template <typename TF, typename... Args>
//     void scatter( TF& f, int tag)const{
//        scatter_(st::iform<TF>(), f, std::forward<Args>(args)...);
//    }
//
//    template <typename TV>
//     auto sample_( EntityId s, TV& v)const{
//        return v;
//    }
//
//    template <typename TV, int N>
//     auto sample_( EntityId s, nTuple<TV, N> const& v)const{
//        return v[((s.w & 0b111) == 0 || (s.w & 0b111) == 7) ? (s.w >> 3) % N
//                                                            : EntityIdCoder::m_id_to_sub_index_[s.w & 0b11]];
//    }
//
//    template <typename TV>
//     auto sample( EntityId s, TV const& v)const{
//        return sample_(s, v);
//    }

//    template <typename TFun>
//     auto getArray( TFun const& fun,  IdxShift S, int tag,
//                         ENABLE_IF(simpla::concept::is_callable<TFun(simpla::EntityId)>::value))const{
//        return [&](index_tuple const& idx)const{
//            EntityId s;
//            s.w = _cast<int16_t>(tag);
//            s.x = _cast<int16_t>(idx[0] + S[0]);
//            s.y = _cast<int16_t>(idx[1] + S[1]);
//            s.z = _cast<int16_t>(idx[2] + S[2]);
//            return sample(s, fun(s));
//        };
//    }
//
//    template <typename TFun>
//     auto getArray( TFun const& fun,  IdxShift S, int tag,
//                         ENABLE_IF(simpla::concept::is_callable<TFun(point_type const&)>::value))const{
//        return [&](index_tuple const& idx)const{
//            EntityId s;
//            s.w = _cast<int16_t>(tag);
//            s.x = _cast<int16_t>(idx[0] + S[0]);
//            s.y = _cast<int16_t>(idx[1] + S[1]);
//            s.z = _cast<int16_t>(idx[2] + S[2]);
//            return sample(tag, fun(m_host_->point(s)));
//        };
//    }
//
//    template <int IFORM, typename TExpr>
//     auto  getArray(std::integral_constant<int, IFORM> const&,  TExpr const& expr,
//    m,
//    index_type i,
//                  index_type j, index_type k, unsigned int n, unsigned int d)  {
//        return getArray( expr,EntityIdCoder::Serialize<IFORM>(i, j, k, n, d));
//    }
//    template <typename TField, typename TOP, typename... Args>
//    void foreach_( TField& self, Range<EntityId> const& r, TOP const& op, int tag) const
//    {
//        r.foreach ([&](EntityId s)  { op(getArray(self, s), getArray(std::forward<Others>(others), s)...);
//        });
//    }
//    template <typename... Args>
//    void foreach (int tag)  {
//        foreach_(std::forward<Others>(others)...);
//    }
//    template <typename TField, typename... Args>
//    void foreach ( TField & self, mesh::MeshZoneTag const& tag, Args && ... args)  {
//        foreach_(self, m_host_->range(tag, st::iform<TField>::value,
//        st::dof<TField>::value),
//                tag);
//    }
//    template <typename TField, typename... Args>
//    void foreach ( TField & self, Args && ... args)  {
//        foreach_(self, m_host_->range(SP_ES_ALL, st::iform<TField>::value,
//        st::dof<TField>::value),
//                tag);
//    }

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
//    Real RBF( point_type const& x0, point_type const& x1, vector_type const& a)  {
//        vector_type r;
//        r = (x1 - x0) / a;
//        // @NOTE this is not  an exact  RBF
//        return (1.0 - std::abs(r[0])) * (1.0 - std::abs(r[1])) * (1.0 - std::abs(r[2]));
//    }
//
//    Real RBF( point_type const& x0, point_type const& x1, Real const& a)  {
//        return (1.0 - m_host_->distance(x1, x0) / a);
//    }

//    template <int DOF, typename... U>
//     void Assign( Field<mesh_type, U...>& f, EntityId
//    s,
//                       nTuple<U, DOF> const& v)  {
//        for (int i = 0; i < DOF; ++i)const{ f[EntityIdCoder::sw(s, i)] = v[i]; }
//    }

////    template <typename... U>
////     void assign( Field<U...>& f,
////                       EntityId s, nTuple<U, 3> const& v)const{
////        for (int i = 0; i < DOF; ++i)const{ f[EntityIdCoder::sw(s, i)] = v[EntityIdCoder::sub_index(s)]; }
////    }
////
////    template <typename V, int DOF, int... I, typename U>
////     void assign( Field<domain_type, V, FACE, DOF, I...>& f,
////                       EntityId s, nTuple<U, 3> const& v)const{
////        for (int i = 0; i < DOF; ++i)const{ f[EntityIdCoder::sw(s, i)] = v[EntityIdCoder::sub_index(s)]; }
////    }
////
////    template <typename V, int DOF, int... I, typename U>
////     void assign( Field<domain_type, V, VOLUME, DOF, I...>& f,
////                       EntityId s, nTuple<U, DOF> const& v)const{
////        for (int i = 0; i < DOF; ++i)const{ f[EntityIdCoder::sw(s, i)] = v[i]; }
//    }
//
//    template <typename V, int IFORM, int DOF, int... I, typename U>
//     void Assign( Field<mesh_type, V, IFORM, DOF, I...>& f,
//                       EntityId s, U const& v)const{
//        for (int i = 0; i < DOF; ++i)const{ f[EntityIdCoder::sw(s, i)] = v; }
//    }

// template <typename TV, typename M, int IFORM, int DOF>
// constexpr Real   calculator<Field<TV, M, IFORM, DOF>>::m_p_curl_factor[3];
}  // namespace scheme{
}  // namespace simpla { {

#endif /* FDM_H_ */
