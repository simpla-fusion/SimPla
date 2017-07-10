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

#include <simpla/algebra/EntityId.h>
#include <simpla/engine/Domain.h>
#include <simpla/utilities/type_traits.h>
#include "Array.h"
#include "Calculus.h"
#include "ExpressionTemplate.h"
namespace simpla {

template <typename THost>
struct FVM {
    DOMAIN_POLICY_HEAD(FVM);

    typedef THost domain_type;
    typedef FVM<domain_type> this_type;
    static constexpr unsigned int NDIMS = 3;
    template <typename V>
    using array = Array<V, ZSFC<NDIMS>>;

    //**********************************************************************************************
    // for element-wise arithmetic operation

    template <int IFORM, typename U, int... N, typename... Args>
    decltype(auto) GetEntity(nTuple<array<U>, N...>& lhs, Args&&... args) const {
        return calculus::getValue(lhs, std::forward<Args>(args)...);
    }

    template <int IFORM, typename... E>
    auto GetEntity(Expression<E...> const& rhs, EntityId s) const {
        return getValue(rhs, IdxShift{0, 0, 0}, EntityIdCoder::m_id_to_sub_index_[s.w & 0b111], s.x, s.y, s.z);
    }

    template <int IFORM, typename U, int... N>
    decltype(auto) GetEntity(nTuple<array<U>, N...>& lhs, EntityId s) const {
        return traits::recursive_index(lhs[EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]], s.w >> 3)(s.x, s.y, s.z);
    }
    template <int IFORM, typename U, int... N>
    decltype(auto) GetEntity(nTuple<array<U>, N...> const& lhs, EntityId s) const {
        return traits::recursive_index(lhs[EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]], s.w >> 3)(s.x, s.y, s.z);
    }

    template <int IFORM, typename RHS>
    auto GetEntity(RHS const& rhs, EntityId s, ENABLE_IF((std::is_arithmetic<RHS>::value))) const {
        return rhs;
    }
    template <int IFORM, typename V, int N1>
    auto GetEntity(nTuple<V, N1> const& rhs, EntityId s) const {
        return rhs[(IFORM == VERTEX || IFORM == VOLUME) ? (s.w >> 3) : EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]];
    }

    template <int IFORM, typename RHS>
    decltype(auto) GetEntity(RHS const& rhs, EntityId s,
                             ENABLE_IF((traits::is_invocable<RHS, EntityId>::value))) const {
        return GetEntity(rhs(s), s);
    }
    template <int IFORM, typename RHS>
    decltype(auto) GetEntity(RHS const& rhs, EntityId s,
                             ENABLE_IF((traits::is_invocable<RHS, point_type>::value))) const {
        return GetEntity(rhs(m_host_->point(s)), s);
    }

    template <int IFORM, typename U, int... N, typename RHS, typename... Args>
    void SetEntity(nTuple<array<U>, N...>& lhs, RHS&& rhs, Args&&... args) const {
        GetEntity<IFORM>(lhs, std::forward<Args>(args)...) =
            GetEntity<IFORM>(std::forward<RHS>(rhs), std::forward<Args>(args)...);
    }
    //    template <typename U, int IFORM, int... DOF, typename RHS>
    //     void Fill( nTuple<array<U>,N...>& lhs, RHS const& rhs)const{
    //        traits::foreach (lhs.Get(), [&](auto& a, auto&&... subs)const{
    //            a = getValue(rhs, IdxShift{0, 0, 0}, std::forward<decltype(subs)>(subs)...);
    //        });
    //    }

    template <typename OtherMesh, typename U, int IFORM, int... N, typename RHS>
    void Fill(Field<OtherMesh, U, IFORM, N...>& lhs, RHS&& rhs) const {
        //        traits::foreach (lhs.Get(), [&](auto& a, int n0, auto&&... subs) {
        //            int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][n0] |
        //                      (reduction_v(tags::multiplication(), 1, std::forward<decltype(subs)>(subs * DOF)...) <<
        //                      3);
        //
        //            int n = (IFORM == VERTEX || IFORM == VOLUME)
        //                        ? (reduction_v(tags::addition(), 0, std::forward<decltype(subs)>(subs)...))
        //                        : n0;
        //            a = [&](index_type x, index_type y, index_type z) {
        //                EntityId s;
        //                s.w = tag;
        //                s.x = x;
        //                s.y = y;
        //                s.z = z;
        //                return calculus::getValue((getValue(rhs, IdxShift{0, 0, 0}, s)), n);
        //            };
        //        });

        //        else if (!r.isNull()) {
        //            r.foreach ([&](EntityId s) {
        //                lhs[s] = calculus::getValue(getValue(rhs, IdxShift{0, 0, 0}, s),
        //                                            (IFORM == VERTEX || IFORM == VOLUME) ? (s.w >> 3) : (s.w &
        //                                            0b111));
        //            });
        //        }
    }

    template <size_t... I, typename TOP, typename... Args, typename... Others>
    auto _invoke_helper(std::index_sequence<I...>, Expression<TOP, Args...> const& expr, IdxShift S,
                        Others&&... others) const {
        return expr.m_op_(getValue(std::get<I>(expr.m_args_), S, std::forward<Others>(others)...)...);
    }

    template <int... I, typename TOP, typename... Args, typename... Others>
    auto eval(std::integer_sequence<int, I...>, Expression<TOP, Args...> const& expr, IdxShift S,
              Others&&... others) const {
        return _invoke_helper(std::make_index_sequence<sizeof...(I)>(), expr, S, std::forward<Others>(others)...);
    }

    template <typename TOP, typename... Args, typename... Others>
    auto getValue(Expression<TOP, Args...> const& expr, IdxShift S, Others&&... others) const {
        return eval(std::integer_sequence<int, traits::iform<Args>::value...>(), expr, S,
                    std::forward<Others>(others)...);
    }

    template <typename T, typename... Args>
    auto getValue(T const& v, IdxShift S, Args&&... args) const {
        return calculus::getValue(v, std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto _getV(std::integral_constant<int, VERTEX>, Args&&... args) const {
        return getValue(m_host_->m_vertex_volume_, std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto _getV(std::integral_constant<int, EDGE>, Args&&... args) const {
        return getValue(m_host_->m_edge_volume_, std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto _getV(std::integral_constant<int, FACE>, Args&&... args) const {
        return getValue(m_host_->m_face_volume_, std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto _getV(std::integral_constant<int, VOLUME>, Args&&... args) const {
        return getValue(m_host_->m_volume_volume_, std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto _getDualV(std::integral_constant<int, VERTEX>, Args&&... args) const {
        return getValue(m_host_->m_vertex_dual_volume_, std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto _getDualV(std::integral_constant<int, EDGE>, Args&&... args) const {
        return getValue(m_host_->m_edge_dual_volume_, std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto _getDualV(std::integral_constant<int, FACE>, Args&&... args) const {
        return getValue(m_host_->m_face_dual_volume_, std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto _getDualV(std::integral_constant<int, VOLUME>, Args&&... args) const {
        return getValue(m_host_->m_volume_dual_volume_, std::forward<Args>(args)...);
    }

    template <typename TExpr, typename... Args>
    auto getDualV(TExpr const& expr, IdxShift S, int n, Args&&... args) const {
        return getValue(expr, S, n, std::forward<Args>(args)...) *
               _getDualV(std::integral_constant<int, traits::iform<TExpr>::value>(), S, n);
    }
    template <typename TExpr, typename... Args>
    auto getV(TExpr const& expr, IdxShift S, int n, Args&&... args) const {
        return getValue(expr, S, n, std::forward<Args>(args)...) *
               _getV(std::integral_constant<int, traits::iform<TExpr>::value>(), S, n);
    }

    //******************************************************************************
    // Exterior algebra
    //******************************************************************************

    //! grad<0>

    template <typename TExpr, typename... Others>
    auto eval(std::integer_sequence<int, VERTEX>, Expression<tags::exterior_derivative, TExpr> const& expr, IdxShift S,
              int n, Others&&... others) const {
        auto const& l = std::get<0>(expr.m_args_);
        IdxShift D{0, 0, 0};
        D[n] = 1;
        return (getV(l, S + D, n, std::forward<Others>(others)...) - getV(l, S, n, std::forward<Others>(others)...)) *
               getValue(m_host_->m_edge_inv_volume_, S, n);
    }

    //! curl<1>

    template <typename TExpr, typename... Others>
    auto eval(std::integer_sequence<int, EDGE>, Expression<tags::exterior_derivative, TExpr> const& expr, IdxShift S,
              int n, Others&&... others) const {
        auto const& l = std::get<0>(expr.m_args_);

        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};

        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return ((getV(l, S + Z, (n + 1) % 3, std::forward<Others>(others)...) -
                 getV(l, S, (n + 1) % 3, std::forward<Others>(others)...)) -
                (getV(l, S + Y, (n + 2) % 3, std::forward<Others>(others)...) -
                 getV(l, S, (n + 2) % 3, std::forward<Others>(others)...))) *
               getValue(m_host_->m_face_inv_volume_, S, n, std::forward<Others>(others)...);
    }

    //! div<2>
    template <typename TExpr, typename... Others>
    auto eval(std::integer_sequence<int, FACE>, Expression<tags::exterior_derivative, TExpr> const& expr, IdxShift S,
              int n, Others&&... others) const {
        auto const& l = std::get<0>(expr.m_args_);

        IdxShift X{1, 0, 0};
        IdxShift Y{0, 1, 0};
        IdxShift Z{0, 0, 1};

        return ((getV(l, S + X, 0, std::forward<Others>(others)...) - getV(l, S, 0, std::forward<Others>(others)...)) +
                (getV(l, S + Y, 1, std::forward<Others>(others)...) - getV(l, S, 1, std::forward<Others>(others)...)) +
                (getV(l, S + Z, 2, std::forward<Others>(others)...) - getV(l, S, 2, std::forward<Others>(others)...))) *
               getValue(m_host_->m_volume_inv_volume_, S, 0, std::forward<Others>(others)...);
    }

    //! curl<2>
    template <typename TExpr, typename... Others>
    auto eval(std::integer_sequence<int, FACE>, Expression<tags::codifferential_derivative, TExpr> const& expr,
              IdxShift S, int n, Others&&... others) const {
        auto const& l = std::get<0>(expr.m_args_);
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};

        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return ((getDualV(l, S, (n + 1) % 3, std::forward<Others>(others)...) -
                 getDualV(l, S - Z, (n + 1) % 3, std::forward<Others>(others)...)) -
                (getDualV(l, S, (n + 2) % 3, std::forward<Others>(others)...) -
                 getDualV(l, S - Y, (n + 2) % 3, std::forward<Others>(others)...))) *
               (-getValue(m_host_->m_edge_inv_dual_volume_, S, n, std::forward<Others>(others)...));
    }

    //! div<1>

    template <typename TExpr, typename... Others>
    auto eval(std::integer_sequence<int, EDGE>, Expression<tags::codifferential_derivative, TExpr> const& expr,
              IdxShift S, int n, Others&&... others) const {
        auto const& l = std::get<0>(expr.m_args_);

        IdxShift X{1, 0, 0};
        IdxShift Y{0, 1, 0};
        IdxShift Z{0, 0, 1};

        return ((getDualV(l, S, 0, std::forward<Others>(others)...) -
                 getDualV(l, S - X, 0, std::forward<Others>(others)...)) +  //
                (getDualV(l, S, 1, std::forward<Others>(others)...) -
                 getDualV(l, S - Y, 1, std::forward<Others>(others)...)) +  //
                (getDualV(l, S, 2, std::forward<Others>(others)...) -
                 getDualV(l, S - Z, 2, std::forward<Others>(others)...))) *
               (-getValue(m_host_->m_vertex_inv_dual_volume_, S, 0, std::forward<Others>(others)...));

        ;
    }

    //! grad<3>

    template <typename TExpr, typename... Others>
    auto eval(std::integer_sequence<int, VOLUME>, Expression<tags::codifferential_derivative, TExpr> const& expr,
              IdxShift S, int n, Others&&... others) const {
        auto const& l = std::get<0>(expr.m_args_);
        IdxShift D{0, 0, 0};
        D[n] = 1;

        return (getV(l, S, 0, std::forward<Others>(others)...) - getV(l, S - D, 0, std::forward<Others>(others)...)) *
               (-getValue(m_host_->m_face_inv_volume_, S, n, std::forward<Others>(others)...));
    }

    //! *Form<IR> => Form<N-IL>

    template <typename TExpr, typename... Others>
    auto eval(std::integer_sequence<int, VERTEX>, Expression<tags::hodge_star, TExpr> const& expr, IdxShift S, int n,
              Others&&... others) const {
        auto const& l = std::get<0>(expr.m_args_);
        return (getV(l, S + IdxShift{0, 0, 0}, n, std::forward<Others>(others)...) +
                getV(l, S + IdxShift{0, 0, 1}, n, std::forward<Others>(others)...) +
                getV(l, S + IdxShift{0, 1, 0}, n, std::forward<Others>(others)...) +
                getV(l, S + IdxShift{0, 1, 1}, n, std::forward<Others>(others)...) +
                getV(l, S + IdxShift{1, 0, 0}, n, std::forward<Others>(others)...) +
                getV(l, S + IdxShift{1, 0, 1}, n, std::forward<Others>(others)...) +
                getV(l, S + IdxShift{1, 1, 0}, n, std::forward<Others>(others)...) +
                getV(l, S + IdxShift{1, 1, 1}, n, std::forward<Others>(others)...)) *
               getValue(m_host_->m_volume_inv_volume_, n, std::forward<Others>(others)...) * 0.125;
    };
    ////***************************************************************************************************
    //! p_curl<1>
    //     constexpr Real m_p_curl_factor_[3] = {0, 1, -1};
    //    template<typename TOP, typename T>   traits::value_type_t
    //    <Expression<TOP, T>>
    //    GetValue(domain_type const &Expression<TOP, T> const &expr,
    //    EntityId const &s,
    //    ENABLE_IF((std::is_same<TOP, tags::p_exterior_derivative < 0>>
    //                      ::value && traits::GetIFORM<T>::value == EDGE))
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
    //      traits::value_type_t
    //    <Expression<tags::p_codifferential_derivative < I>, T>>
    //    GetValue(
    //    domain_type const &m,
    //    Expression<tags::p_codifferential_derivative < I>, T
    //    > const &expr,
    //    EntityId const &s,
    //    ENABLE_IF(traits::GetIFORM<T>::value == FACE)
    //    )
    //    {
    //
    //        return (get_v(std::Serialize<0>(expr.m_args_), s + EntityIdCoder::DI(I)) -
    //                get_v(std::Serialize<0>(expr.m_args_), s - EntityIdCoder::DI(I))
    //               ) * m_host_->inv_dual_volume(s) * m_p_curl_factor_[(I + 3 -
    //               EntityIdCoder::sub_index(s)) % 3];
    //    }

    ////***************************************************************************************************
    //
    ////! map_to

    template <int I, typename TExpr, typename... Others>
    auto _map_to(std::index_sequence<I, I>, TExpr const& expr, IdxShift S, Others&&... others) const {
        return getValue(expr, S, std::forward<Others>(others)...);
    };

    template <typename TExpr, typename... Others>
    auto _map_to(std::index_sequence<VERTEX, EDGE>, TExpr const& expr, IdxShift S, int n, Others&&... others) const {
        IdxShift D{0, 0, 0};
        D[n] = 1;

        return (getValue(expr, S, 0, std::forward<Others>(others)...) +
                getValue(expr, S + D, 0, std::forward<Others>(others)...)) *
               0.5;
    }

    template <typename TExpr, typename... Others>
    auto _map_to(std::index_sequence<EDGE, VERTEX>, TExpr const& expr, IdxShift S, int n0, int n,
                 Others&&... others) const {
        IdxShift D{0, 0, 0};

        D[n] = 1;
        return (getValue(expr, S - D, n, std::forward<Others>(others)...) +
                getValue(expr, S, n, std::forward<Others>(others)...)) *
               0.5;
    }

    template <typename TExpr, typename... Others>
    auto _map_to(std::index_sequence<VERTEX, FACE>, TExpr const& expr, IdxShift S, int n, Others&&... others) const {
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;
        return (getValue(expr, S, 0, std::forward<Others>(others)...) +
                getValue(expr, S + Y, 0, std::forward<Others>(others)...) +
                getValue(expr, S + Z, 0, std::forward<Others>(others)...) +
                getValue(expr, S + Y + Z, 0, std::forward<Others>(others)...)) *
               0.25;
    }

    template <typename TExpr, typename... Others>
    auto _map_to(std::index_sequence<FACE, VERTEX>, TExpr const& expr, IdxShift S, int n0, int n,
                 Others&&... others) const {
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};

        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (getValue(expr, S - Y - Z, n, std::forward<Others>(others)...) +
                getValue(expr, S - Y, n, std::forward<Others>(others)...) +
                getValue(expr, S - Z, n, std::forward<Others>(others)...) +
                getValue(expr, S, n, std::forward<Others>(others)...)) *
               0.25;
    }

    template <typename TExpr, typename... Others>
    auto _map_to(std::index_sequence<VERTEX, VOLUME>, TExpr const& expr, IdxShift S, Others&&... others) const {
        return (getValue(expr, S + IdxShift{0, 0, 0}, std::forward<Others>(others)...) +
                getValue(expr, S + IdxShift{0, 0, 1}, std::forward<Others>(others)...) +
                getValue(expr, S + IdxShift{0, 1, 0}, std::forward<Others>(others)...) +
                getValue(expr, S + IdxShift{0, 1, 1}, std::forward<Others>(others)...) +
                getValue(expr, S + IdxShift{1, 0, 0}, std::forward<Others>(others)...) +
                getValue(expr, S + IdxShift{1, 0, 1}, std::forward<Others>(others)...) +
                getValue(expr, S + IdxShift{1, 1, 0}, std::forward<Others>(others)...) +
                getValue(expr, S + IdxShift{1, 1, 1}, std::forward<Others>(others)...)) *
               0.125;
    }

    template <typename TExpr, typename... Others>
    auto _map_to(std::index_sequence<VOLUME, VERTEX>, TExpr const& expr, IdxShift S, Others&&... others) const {
        return (getValue(expr, S - IdxShift{1, 1, 1}, std::forward<Others>(others)...) +
                getValue(expr, S - IdxShift{1, 1, 0}, std::forward<Others>(others)...) +
                getValue(expr, S - IdxShift{1, 0, 1}, std::forward<Others>(others)...) +
                getValue(expr, S - IdxShift{1, 0, 0}, std::forward<Others>(others)...) +
                getValue(expr, S - IdxShift{0, 1, 1}, std::forward<Others>(others)...) +
                getValue(expr, S - IdxShift{0, 1, 0}, std::forward<Others>(others)...) +
                getValue(expr, S - IdxShift{0, 0, 1}, std::forward<Others>(others)...) +
                getValue(expr, S, std::forward<Others>(others)...)) *
               0.125;
    }

    template <typename TExpr, typename... Others>
    auto _map_to(std::index_sequence<VOLUME, FACE>, TExpr const& expr, IdxShift S, int n, Others&&... others) const {
        IdxShift D{0, 0, 0};
        D[n] = 1;
        return (getValue(expr, S - D, std::forward<Others>(others)...) +
                getValue(expr, S, std::forward<Others>(others)...)) *
               0.5;
    }

    template <typename TExpr, typename... Others>
    auto _map_to(TExpr const& expr, IdxShift S, int n0, int n, Others&&... others,
                 std::index_sequence<FACE, VOLUME>) const {
        IdxShift D{0, 0, 0};
        D[n] = 1;

        return (getValue(expr, S, n, std::forward<Others>(others)...) +
                getValue(expr, S + D, n, std::forward<Others>(others)...)) *
               0.5;
    }

    template <typename TExpr, typename... Others>
    auto _map_to(std::index_sequence<VOLUME, EDGE>, TExpr const& expr, IdxShift S, int n, Others&&... others) const {
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (getValue(expr, S - Y, std::forward<Others>(others)...) +
                getValue(expr, S - Z, std::forward<Others>(others)...) +
                getValue(expr, S - Y - Z, std::forward<Others>(others)...) +
                getValue(expr, S, std::forward<Others>(others)...)) *
               0.25;
    }

    template <typename TExpr, typename... Others>
    auto _map_to(std::index_sequence<EDGE, VOLUME>, TExpr const& expr, IdxShift S, int n0, int n,
                 Others&&... others) const {
        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};

        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (getValue(expr, S, n, std::forward<Others>(others)...) +
                getValue(expr, S + Z, n, std::forward<Others>(others)...) +
                getValue(expr, S + Y, n, std::forward<Others>(others)...) +
                getValue(expr, S + Y + Z, n, std::forward<Others>(others)...)) *
               0.25;
    }

    template <typename TExpr, int ISrc, int IDest, typename... Others>
    auto eval(std::integer_sequence<int, ISrc>, Expression<tags::map_to<IDest>, TExpr> const& expr, IdxShift S,
              Others&&... others) const {
        return _map_to(std::index_sequence<ISrc, IDest>(), std::get<0>(expr.m_args_), S,
                       std::forward<Others>(others)...);
    }
    //***************************************************************************************************
    //
    //! Form<IL> ^ Form<IR> => Form<IR+IL>
    template <typename... TExpr, int IL, int IR, typename... Others>
    auto eval(std::integer_sequence<int, IL, IR>, Expression<tags::wedge, TExpr...> const& expr, IdxShift S,
              Others&&... others) const {
        return m_host_->inner_product(
            _map_to(std::index_sequence<IL, IR + IL>(), std::get<0>(expr.m_args_), S, std::forward<Others>(others)...),
            _map_to(std::index_sequence<IR, IR + IL>(), std::get<1>(expr.m_args_), S, std::forward<Others>(others)...));
    }

    template <typename... TExpr, typename... Others>
    auto eval(std::integer_sequence<int, EDGE, EDGE>, Expression<tags::wedge, TExpr...> const& expr, IdxShift S, int n,
              Others&&... others) const {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;

        return (getValue(l, S, (n + 2) % 3, std::forward<Others>(others)...) +
                getValue(l, S + Y, (n + 2) % 3, std::forward<Others>(others)...)) *
               (getValue(l, S, (n + 1) % 3, std::forward<Others>(others)...) +
                getValue(l, S + Z, (n + 1) % 3, std::forward<Others>(others)...)) *
               0.25;
    }

    template <typename... TExpr, typename... Others>
    auto eval(std::integer_sequence<int, FACE, FACE>, Expression<tags::wedge, TExpr...> const& expr, IdxShift S, int n,
              Others&&... others) const {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        IdxShift Y{0, 0, 0};
        IdxShift Z{0, 0, 0};
        Y[(n + 1) % 3] = 1;
        Z[(n + 2) % 3] = 1;
        return getValue(l, S + Z, n, std::forward<Others>(others)...) *
                   getValue(r, S + Y, n, std::forward<Others>(others)...) -
               getValue(l, S + Z, n, std::forward<Others>(others)...) *
                   getValue(r, S + Y, n, std::forward<Others>(others)...);
    }

    template <typename... TExpr, typename... Others>
    auto eval(std::integer_sequence<int, VERTEX, VERTEX>, Expression<tags::dot, TExpr...> const& expr, IdxShift S,
              int n0, Others&&... others) const {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        return getValue(l, S, 0, std::forward<Others>(others)...) * getValue(r, S, 0, std::forward<Others>(others)...) +
               getValue(l, S, 1, std::forward<Others>(others)...) * getValue(r, S, 1, std::forward<Others>(others)...) +
               getValue(l, S, 2, std::forward<Others>(others)...) * getValue(r, S, 2, std::forward<Others>(others)...);
    }

    template <typename... TExpr, typename... Others>
    auto eval(std::integer_sequence<int, VOLUME, VOLUME>, Expression<tags::dot, TExpr...> const& expr, IdxShift S,
              int n0, Others&&... others) const {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        return getValue(l, S, 0, 0, std::forward<Others>(others)...) *
                   getValue(r, S, 0, 0, std::forward<Others>(others)...) +
               getValue(l, S, 0, 1, std::forward<Others>(others)...) *
                   getValue(r, S, 0, 1, std::forward<Others>(others)...) +
               getValue(l, S, 0, 2, std::forward<Others>(others)...) *
                   getValue(r, S, 0, 2, std::forward<Others>(others)...);
    }

    template <typename... TExpr, typename... Others>
    auto eval(std::integer_sequence<int, EDGE, EDGE>, Expression<tags::dot, TExpr...> const& expr, IdxShift S,
              Others&&... others) const {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        return eval(std::integer_sequence<int, VERTEX>(), dot_v(map_to<VERTEX>(l), map_to<VERTEX>(r)), S,
                    std::forward<Others>(others)...);
    }
    template <typename... TExpr, typename... Others>
    auto eval(std::integer_sequence<int, FACE, FACE>, Expression<tags::dot, TExpr...> const& expr, IdxShift S,
              Others&&... others) const {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);
        return eval(std::integer_sequence<int, VERTEX>(), dot_v(map_to<VERTEX>(l), map_to<VERTEX>(r)), S,
                    std::forward<Others>(others)...);
    }
    template <typename... TExpr, typename... Others>
    auto eval(std::integer_sequence<int, VERTEX, VERTEX>, Expression<tags::cross, TExpr...> const& expr, IdxShift S,
              int n0, int n, Others&&... others) const {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        return getValue(l, S, 0, ((n + 1) % 3), std::forward<Others>(others)...) *
                   getValue(r, S, 0, ((n + 2) % 3), std::forward<Others>(others)...) -
               getValue(l, S, 0, ((n + 2) % 3), std::forward<Others>(others)...) *
                   getValue(r, S, 0, ((n + 1) % 3), std::forward<Others>(others)...);
    }

    template <typename... TExpr, typename... Others>
    auto eval(std::integer_sequence<int, VOLUME, VOLUME>, Expression<tags::cross, TExpr...> const& expr, IdxShift S,
              int n0, int n, Others&&... others) const {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);
        return getValue(l, S, 0, (n + 1) % 3, std::forward<Others>(others)...) *
                   getValue(r, S, 0, (n + 2) % 3, std::forward<Others>(others)...) -
               getValue(l, S, 0, (n + 2) % 3, std::forward<Others>(others)...) *
                   getValue(r, S, 0, (n + 1) % 3, std::forward<Others>(others)...);
    }

    //    ///*********************************************************************************************
    //    /// @name general_algebra General algebra
    //    /// @{
    //
    //    //    template <typename V, int I, int... D>
    //    //     V const& getValue( Field<M, V, I, D...> const& f, EntityId s)const{
    //    //        return f[EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]][(s.w >> 3)](s.x, s.y, s.z);
    //    //    };
    //    //    template <typename V, int I, int... D>
    //    //     V& getValue( Field<M, V, I, D...>& f, EntityId s)const{
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
    //        return getValue(f, ((s + X) + Y) + Z) * (r[0]) * (r[1]) * (r[2]) +    //
    //               getValue(f, (s + X) + Y) * (r[0]) * (r[1]) * (1.0 - r[2]) +    //
    //               getValue(f, (s + X) + Z) * (r[0]) * (1.0 - r[1]) * (r[2]) +    //
    //               getValue(f, (s + X)) * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]) +  //
    //               getValue(f, (s + Y) + Z) * (1.0 - r[0]) * (r[1]) * (r[2]) +    //
    //               getValue(f, (s + Y)) * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]) +  //
    //               getValue(f, s + Z) * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]) +    //
    //               getValue(f, s) * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
    //    }
    //
    //   public:
    //    template <typename TF>
    //    constexpr  auto gather( TF const& f, point_type const& r,
    //                                 ENABLE_IF((traits::iform<TF>::value == VERTEX)))const{
    //        return gather_impl_(f, m_host_->point_global_to_local(r, 0));
    //    }
    //
    //    template <typename TF>
    //    constexpr  auto gather( TF const& f, point_type const& r,
    //                                 ENABLE_IF((traits::iform<TF>::value == EDGE)))const{
    //        return TF::field_value_type{gather_impl_(f, m_host_->point_global_to_local(r, 1)),
    //                                    gather_impl_(f, m_host_->point_global_to_local(r, 2)),
    //                                    gather_impl_(f, m_host_->point_global_to_local(r, 4))};
    //    }
    //
    //    template <typename TF>
    //    constexpr  auto gather( TF const& f, point_type const& r,
    //                                 ENABLE_IF((traits::iform<TF>::value == FACE)))const{
    //        return TF::field_value_type{gather_impl_(f, m_host_->point_global_to_local(r, 6)),
    //                                    gather_impl_(f, m_host_->point_global_to_local(r, 5)),
    //                                    gather_impl_(f, m_host_->point_global_to_local(r, 3))};
    //    }
    //
    //    template <typename TF>
    //    constexpr  auto gather( TF const& f, point_type const& x,
    //                                 ENABLE_IF((traits::iform<TF>::value == VOLUME)))const{
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
    //        getValue(f, ((s + X) + Y) + Z) += v * (r[0]) * (r[1]) * (r[2]);
    //        getValue(f, (s + X) + Y) += v * (r[0]) * (r[1]) * (1.0 - r[2]);
    //        getValue(f, (s + X) + Z) += v * (r[0]) * (1.0 - r[1]) * (r[2]);
    //        getValue(f, s + X) += v * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
    //        getValue(f, (s + Y) + Z) += v * (1.0 - r[0]) * (r[1]) * (r[2]);
    //        getValue(f, s + Y) += v * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]);
    //        getValue(f, s + Z) += v * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]);
    //        getValue(f, s) += v * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
    //    }
    //
    //    template <typename TF, typename TX, typename TV>
    //     void scatter_( std::integral_constant<int, VERTEX>, TF& f, TX const& x, TV const& u)
    //    {
    //        scatter_impl_(f, m_host_->point_global_to_local(x, 0), u);
    //    }
    //
    //    template <typename TF, typename TX, typename TV>
    //     void scatter_( std::integral_constant<int, EDGE>, TF& f, TX const& x, TV const& u)const{
    //        scatter_impl_(f, m_host_->point_global_to_local(x, 1), u[0]);
    //        scatter_impl_(f, m_host_->point_global_to_local(x, 2), u[1]);
    //        scatter_impl_(f, m_host_->point_global_to_local(x, 4), u[2]);
    //    }
    //
    //    template <typename TF, typename TX, typename TV>
    //     void scatter_( std::integral_constant<int, FACE>, TF& f, TX const& x, TV const& u)const{
    //        scatter_impl_(f, m_host_->point_global_to_local(x, 6), u[0]);
    //        scatter_impl_(f, m_host_->point_global_to_local(x, 5), u[1]);
    //        scatter_impl_(f, m_host_->point_global_to_local(x, 3), u[2]);
    //    }
    //
    //    template <typename TF, typename TX, typename TV>
    //     void scatter_( std::integral_constant<int, VOLUME>, TF& f, TX const& x, TV const& u)
    //    {
    //        scatter_impl_(f, m_host_->point_global_to_local(x, 7), u);
    //    }
    //
    //   public:
    //    template <typename TF, typename... Args>
    //     void scatter( TF& f, Args&&... args)const{
    //        scatter_(traits::iform<TF>(), f, std::forward<Args>(args)...);
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
    //     auto getValue( TFun const& fun,  IdxShift S, int n, Others&&... others,
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
    //     auto getValue( TFun const& fun,  IdxShift S, int n, Others&&... others,
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
    //     auto  getValue(std::integral_constant<int, IFORM> const&,  TExpr const& expr,
    //    m,
    //    index_type i,
    //                  index_type j, index_type k, unsigned int n, unsigned int d)  {
    //        return getValue( expr,EntityIdCoder::Serialize<IFORM>(i, j, k, n, d));
    //    }
    //    template <typename TField, typename TOP, typename... Args>
    //    void foreach_( TField& self, Range<EntityId> const& r, TOP const& op, Args&&... args) const
    //    {
    //        r.foreach ([&](EntityId s)  { op(getValue(self, s), getValue(std::forward<Others>(others), s)...);
    //        });
    //    }
    //    template <typename... Args>
    //    void foreach (Args&&... args)  {
    //        foreach_(std::forward<Others>(others)...);
    //    }
    //    template <typename TField, typename... Args>
    //    void foreach ( TField & self, mesh::MeshZoneTag const& tag, Args && ... args)  {
    //        foreach_(self, m_host_->range(tag, traits::iform<TField>::value, traits::dof<TField>::value),
    //                 std::forward<Others>(others)...);
    //    }
    //    template <typename TField, typename... Args>
    //    void foreach ( TField & self, Args && ... args)  {
    //        foreach_(self, m_host_->range(SP_ES_ALL, traits::iform<TField>::value, traits::dof<TField>::value),
    //                 std::forward<Others>(others)...);
    //    }
};

template <typename THost>
void FVM<THost>::InitialCondition(Real time_now) {}
template <typename THost>
void FVM<THost>::BoundaryCondition(Real time_now, Real time_dt) {}
template <typename THost>
void FVM<THost>::Advance(Real time_now, Real time_dt) {}
template <typename THost>
std::shared_ptr<simpla::data::DataTable> FVM<THost>::Serialize() const {
    return std::make_shared<simpla::data::DataTable>();
}
template <typename THost>
void FVM<THost>::Deserialize(std::shared_ptr<simpla::data::DataTable> const& cfg) {}
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
//     void Assign( Field<domain_type, U...>& f, EntityId
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
//     void Assign( Field<domain_type, V, IFORM, DOF, I...>& f,
//                       EntityId s, U const& v)const{
//        for (int i = 0; i < DOF; ++i)const{ f[EntityIdCoder::sw(s, i)] = v; }
//    }

// template <typename TV, typename M, int IFORM, int DOF>
// constexpr Real   calculator<Field<TV, M, IFORM, DOF>>::m_p_curl_factor[3];
}  // namespace simpla { {

#endif /* FDM_H_ */
