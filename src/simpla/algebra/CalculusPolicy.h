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

#include <simpla/utilities/EntityId.h>
#include <simpla/utilities/FancyStream.h>
#include <simpla/utilities/macro.h>
#include <simpla/utilities/sp_def.h>
#include <simpla/utilities/type_traits.h>
#include "Calculus.h"
#include "Field.h"
namespace simpla {
namespace calculus {

namespace st = simpla::traits;
template <typename...>
class calculator;

/**
 * @ingroup diff_scheme
 *
 * finite volume
 */
template <typename TM>
struct calculator<TM> {
    typedef TM mesh_type;
    typedef calculator<mesh_type> this_type;

    typedef EntityIdCoder M;

    template <typename TOP, int... I>
    struct expression_tag {};

   private:
    template <typename FExpr>
    static decltype(auto) get_v(mesh_type const& m, FExpr const& f, EntityId const s) {
        return getValue(m, f, s) * m.volume(s);
    }

    template <typename FExpr>
    static decltype(auto) get_d(mesh_type const& m, FExpr const& f, EntityId const s) {
        return getValue(m, f, s) * m.dual_volume(s);
    }

    //******************************************************************************
    // Exterior algebra
    //******************************************************************************

    //! grad<0>
    template <typename TExpr>
    static decltype(auto) eval(mesh_type const& m, TExpr const& expr, EntityId s,
                               expression_tag<tags::_exterior_derivative, VERTEX>) {
        EntityId D = M::delta_index(s);
        return (get_v(m, std::get<0>(expr.m_args_), s + D) - get_v(m, std::get<0>(expr.m_args_), s - D)) *
               m.inv_volume(s);
    }

    //! curl<1>
    template <typename TExpr>
    static decltype(auto) eval(mesh_type const& m, TExpr const& expr, EntityId s,
                               expression_tag<tags::_exterior_derivative, EDGE>) {
        EntityId X = M::delta_index(M::dual(s));
        EntityId Y = M::rotate(X);
        EntityId Z = M::inverse_rotate(X);

        return ((get_v(m, std::get<0>(expr.m_args_), s + Y) - get_v(m, std::get<0>(expr.m_args_), s - Y)) -
                (get_v(m, std::get<0>(expr.m_args_), s + Z) - get_v(m, std::get<0>(expr.m_args_), s - Z))) *
               m.inv_volume(s);
    }

    //! div<1>

    template <typename TExpr>
    static decltype(auto) eval(mesh_type const& m, TExpr const& expr, EntityId s,
                               expression_tag<tags::_codifferential_derivative, EDGE>) {
        return -(get_d(m, std::get<0>(expr.m_args_), s + M::_DI) - get_d(m, std::get<0>(expr.m_args_), s - M::_DI) +
                 get_d(m, std::get<0>(expr.m_args_), s + M::_DJ) - get_d(m, std::get<0>(expr.m_args_), s - M::_DJ) +
                 get_d(m, std::get<0>(expr.m_args_), s + M::_DK) - get_d(m, std::get<0>(expr.m_args_), s - M::_DK)) *
               m.inv_dual_volume(s);
    }

    //! div<2>
    template <typename TExpr>
    static decltype(auto) eval(mesh_type const& m, TExpr const& expr, EntityId s,
                               expression_tag<tags::_exterior_derivative, FACE>) {
        return (get_v(m, std::get<0>(expr.m_args_), s + M::_DI) - get_v(m, std::get<0>(expr.m_args_), s - M::_DI) +
                get_v(m, std::get<0>(expr.m_args_), s + M::_DJ) - get_v(m, std::get<0>(expr.m_args_), s - M::_DJ) +
                get_v(m, std::get<0>(expr.m_args_), s + M::_DK) - get_v(m, std::get<0>(expr.m_args_), s - M::_DK)) *
               m.inv_volume(s);
    }

    //! curl<2>
    template <typename TExpr>
    static decltype(auto) eval(mesh_type const& m, TExpr const& expr, EntityId s,
                               expression_tag<tags::_codifferential_derivative, FACE>) {
        EntityId X = M::delta_index(s);
        EntityId Y = M::rotate(X);
        EntityId Z = M::inverse_rotate(X);

        return -((get_d(m, std::get<0>(expr.m_args_), s + Y) - get_d(m, std::get<0>(expr.m_args_), s - Y)) -
                 (get_d(m, std::get<0>(expr.m_args_), s + Z) - get_d(m, std::get<0>(expr.m_args_), s - Z))) *
               m.inv_dual_volume(s);
    }

    //! grad<3>

    template <typename TExpr>
    static decltype(auto) eval(mesh_type const& m, TExpr const& expr, EntityId s,
                               expression_tag<tags::_codifferential_derivative, VOLUME>) {
        EntityId D = M::delta_index(M::dual(s));

        return -(get_d(m, std::get<0>(expr.m_args_), s + D) - get_d(m, std::get<0>(expr.m_args_), s - D)) *
               m.inv_dual_volume(s);
    }
    //
    //    template<typename T>
    //    static  decltype(auto) // traits::value_type_t
    //    <Expression<tags::_codifferential_derivative, T>>
    //    GetValue(mesh_type const &m,
    //    Expression<tags::_codifferential_derivative, T> const &expr,
    //              EntityId const &s)
    //    {
    //        static_assert(traits::GetIFORM<T>::value != VOLUME &&
    //        traits::GetIFORM<T>::value != VERTEX,
    //                      "ERROR: grad VERTEX/VOLUME Field  ");
    //    };
    //! *Form<IR> => Form<N-IL>

    template <typename TExpr>
    static decltype(auto) eval(mesh_type const& m, TExpr const& expr, EntityId s,
                               expression_tag<tags::_hodge_star, VERTEX>) {
        auto const& l = std::get<0>(expr.m_args_);
        int i = M::iform(s);
        EntityId X = (i == VERTEX || i == VOLUME) ? M::_DI : M::delta_index(M::dual(s));
        EntityId Y = M::rotate(X);
        EntityId Z = M::inverse_rotate(X);

        return (get_v(m, l, ((s - X) - Y) - Z) + get_v(m, l, ((s - X) - Y) + Z) + get_v(m, l, ((s - X) + Y) - Z) +
                get_v(m, l, ((s - X) + Y) + Z) + get_v(m, l, ((s + X) - Y) - Z) + get_v(m, l, ((s + X) - Y) + Z) +
                get_v(m, l, ((s + X) + Y) - Z) + get_v(m, l, ((s + X) + Y) + Z)) *
               m.inv_dual_volume(s) * 0.125;
    };

    ////***************************************************************************************************
    //! p_curl<1>
    static constexpr Real m_p_curl_factor_[3] = {0, 1, -1};

    //    template<typename TOP, typename T> static  traits::value_type_t
    //    <Expression<TOP, T>>
    //    GetValue(mesh_type const &m, Expression<TOP, T> const &expr,
    //    EntityId const &s,
    //    ENABLE_IF((std::is_same<TOP, tags::_p_exterior_derivative < 0>>
    //                      ::value && traits::GetIFORM<T>::value == EDGE))
    //    )
    //    {
    //        return (get_v(m, std::PopPatch<0>(expr.m_args_), s + M::DI(I)) -
    //                get_v(m, std::PopPatch<0>(expr.m_args_), s - M::DI(I))
    //               ) * m.inv_volume(s) * m_p_curl_factor_[(I + 3 -
    //               M::sub_index(s)) % 3];
    //    }
    //
    //
    //    template<typename T, size_t I>
    //    static  traits::value_type_t
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
    //        return (get_v(m, std::PopPatch<0>(expr.m_args_), s + M::DI(I)) -
    //                get_v(m, std::PopPatch<0>(expr.m_args_), s - M::DI(I))
    //               ) * m.inv_dual_volume(s) * m_p_curl_factor_[(I + 3 -
    //               M::sub_index(s)) % 3];
    //    }

    ////***************************************************************************************************
    //
    ////! map_to
    //    template<typename T, size_t I>
    //     static T
    //    _map_to(mesh_type const &m, T const &r, EntityId const &s,
    //    int_sequence<VERTEX, I>,
    //          st::is_primary_t<T> *_p = nullptr) { return r; }
    //
    //    template<typename TF, size_t I>
    //     static traits::value_type_t<TF>
    //    _map_to(mesh_type const &m, TF const &expr, EntityId const &s,
    //    int_sequence<I, I>,
    //          std::enable_if_t<!st::is_primary<TF>::value>
    //          *_p = nullptr) { return GetValue(m, expr, s); }
   private:
    template <typename TExpr, int I>
    static decltype(auto) _map_to(mesh_type const& m, TExpr const& f, EntityId s, std::index_sequence<I, I>) {
        return getValue(m, f, s);
    };

    template <typename TExpr>
    static decltype(auto) _map_to(mesh_type const& m, TExpr const& expr, EntityId s,
                                  std::index_sequence<VERTEX, EDGE>) {
        int n = M::sub_index(s);
        EntityId X = M::delta_index(s);
        auto l = getValue(m, expr, sw(s - X, n));
        auto r = getValue(m, expr, sw(s + X, n));
        return (l + r) * 0.5;
    }

    template <typename TExpr>
    static decltype(auto) _map_to(mesh_type const& m, TExpr const& expr, EntityId s,
                                  std::index_sequence<VERTEX, FACE>) {
        int n = M::sub_index(s);

        auto X = M::delta_index(M::dual(s));
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);

        return (getValue(m, expr, sw(s - Y - Z, n)) + getValue(m, expr, sw(s - Y + Z, n)) +
                getValue(m, expr, sw(s + Y - Z, n)) + getValue(m, expr, sw(s + Y + Z, n))) *
               0.25;
    }

    template <typename TExpr>
    static decltype(auto) _map_to(mesh_type const& m, TExpr const& expr, EntityId s,
                                  std::index_sequence<VERTEX, VOLUME>) {
        auto const& l = expr;

        auto X = M::DI(0, s);
        auto Y = M::DI(1, s);
        auto Z = M::DI(2, s);

        return (getValue(m, l, s - X - Y - Z) + getValue(m, l, s - X - Y + Z) + getValue(m, l, s - X + Y - Z) +
                getValue(m, l, s - X + Y + Z) + getValue(m, l, s + X - Y - Z) + getValue(m, l, s + X - Y + Z) +
                getValue(m, l, s + X + Y - Z) + getValue(m, l, s + X + Y + Z)) *
               0.125;
    }

    template <typename TExpr>
    static decltype(auto) _map_to(mesh_type const& m, TExpr const& expr, EntityId s,
                                  std::index_sequence<EDGE, VERTEX>) {
        EntityId X = M::DI(s.w, s);
        return (getValue(m, expr, sw(s - X, 0)) + getValue(m, expr, sw(s + X, 0))) * 0.5;
    }

    template <typename TExpr>
    static decltype(auto) _map_to(mesh_type const& m, TExpr const& expr, EntityId s,
                                  std::index_sequence<FACE, VERTEX>) {
        EntityId Y = M::DI((s.w + 1) % 3, s);
        EntityId Z = M::DI((s.w + 2) % 3, s);

        return (getValue(m, expr, sw(s - Y - Z, 0)) + getValue(m, expr, sw(s - Y + Z, 0)) +
                getValue(m, expr, sw(s + Y - Z, 0)) + getValue(m, expr, sw(s + Y + Z, 0))) *
               0.25;
    }

    template <typename TExpr>
    static decltype(auto) _map_to(mesh_type const& m, TExpr const& expr, EntityId s,
                                  std::index_sequence<VOLUME, VERTEX>) {
        auto const& l = expr;

        auto X = M::DI(0, s);
        auto Y = M::DI(1, s);
        auto Z = M::DI(2, s);

        return (getValue(m, l, ((s - X - Y - Z))) + getValue(m, l, ((s - X - Y + Z))) +
                getValue(m, l, ((s - X + Y - Z))) + getValue(m, l, ((s - X + Y + Z))) +
                getValue(m, l, ((s + X - Y - Z))) + getValue(m, l, ((s + X - Y + Z))) +
                getValue(m, l, ((s + X + Y - Z))) + getValue(m, l, ((s + X + Y + Z)))) *
               0.125;
    }

    template <typename TExpr>
    static decltype(auto) _map_to(mesh_type const& m, TExpr const& expr, EntityId s,
                                  std::index_sequence<VOLUME, FACE>) {
        auto X = M::delta_index(M::dual(s));

        return (getValue(m, expr, s - X) + getValue(m, expr, s + X)) * 0.5;
    }

    template <typename TExpr>
    static decltype(auto) _map_to(mesh_type const& m, TExpr const& expr, EntityId s,
                                  std::index_sequence<VOLUME, EDGE>) {
        auto const& l = expr;
        auto X = M::delta_index(s);
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);

        return (getValue(m, l, s - Y - Z) + getValue(m, l, s - Y + Z) + getValue(m, l, s + Y - Z) +
                getValue(m, l, s + Y + Z)) *
               0.25;
    }

    template <typename TExpr>
    static decltype(auto) _map_to(mesh_type const& m, TExpr const& expr, EntityId s,
                                  std::index_sequence<FACE, VOLUME>) {
        EntityId X = M::DI(s.w, s);

        return (getValue(m, expr, sw(s - X, 0)) + getValue(m, expr, sw(s + X, 0))) * 0.5;
    }

    template <typename TExpr>
    static decltype(auto) _map_to(mesh_type const& m, TExpr const& expr, EntityId s,
                                  std::index_sequence<EDGE, VOLUME>) {
        //        auto const &l = expr;
        //
        //        auto X = M::DI(0, s);
        //        auto Y = M::DI(1, s);
        //        auto Z = M::DI(2, s);

        EntityId Y = M::DI((s.w + 1) % 3, s);
        EntityId Z = M::DI((s.w + 1) % 3, s);

        return (getValue(m, expr, sw(s - Y - Z, 0)) + getValue(m, expr, sw(s - Y + Z, 0)) +
                getValue(m, expr, sw(s + Y - Z, 0)) + getValue(m, expr, sw(s + Y + Z, 0))) *
               0.25;
    }

    template <typename TExpr, int IL, int IR>
    static decltype(auto) eval(mesh_type const& m, TExpr const& expr, EntityId s,
                               expression_tag<tags::_map_to<IL>, IR>) {
        return _map_to(m, std::get<0>(expr.m_args_), s, std::index_sequence<IL, IR>());
    }

    //    template<int I, typename T>
    //    static  traits::value_type_t <T>
    //    map_to(mesh_type const &m, T const &expr, EntityId const &s)
    //    {
    //        return _map_to(m, expr, s, int_sequence<traits::GetIFORM<T>::value,
    //        I>());
    //    };
    //
    //    template<int I, typename T>
    //    static  traits::value_type_t <T>
    //    GetValue(mesh_type const &m, Expression<tags::_map_to < I>,
    //    T
    //
    //    > const &expr,
    //    EntityId const &s
    //    )
    //    {
    //        return map_to<I>(m, std::PopPatch<0>(expr.m_args_), s);
    //    };

    //***************************************************************************************************
    //
    //! Form<IL> ^ Form<IR> => Form<IR+IL>
    template <typename TExpr, int IL, int IR>
    static decltype(auto) eval(mesh_type const& m, TExpr const& expr, EntityId s,
                               expression_tag<tags::_wedge, IL, IR>) {
        return m.inner_product(_map_to(m, std::get<0>(expr.m_args_), s, std::index_sequence<IL, IR + IL>()),
                               _map_to(m, std::get<1>(expr.m_args_), s, std::index_sequence<IR, IR + IL>()), s);
    }

    template <typename TExpr>
    static decltype(auto) eval(mesh_type const& m, TExpr const& expr, EntityId s,
                               expression_tag<tags::_wedge, EDGE, EDGE>) {
        auto const& l = std::get<0>(expr.m_args_);
        auto const& r = std::get<1>(expr.m_args_);

        auto Y = M::delta_index(M::rotate(M::dual(s)));
        auto Z = M::delta_index(M::inverse_rotate(M::dual(s)));

        return ((getValue(m, l, s - Y) + getValue(m, l, s + Y)) * (getValue(m, l, s - Z) + getValue(m, l, s + Z)) *
                0.25);
    }

    static EntityId sw(EntityId s, u_int16_t w) {
        s.w = w;
        return s;
    }

    template <typename TExpr, int I>
    static decltype(auto) eval(mesh_type const& m, TExpr const& expr, EntityId s,
                               expression_tag<tags::_wedge, I, I>)  //
    {
        return getValue(m, std::get<0>(expr.m_args_), sw(s, (s.w + 1) % 3)) *
                   getValue(m, std::get<1>(expr.m_args_), sw(s, (s.w + 2) % 3)) -
               getValue(m, std::get<0>(expr.m_args_), sw(s, (s.w + 2) % 3)) *
                   getValue(m, std::get<1>(expr.m_args_), sw(s, (s.w + 1) % 3));
    }

    template <typename TExpr, int I>
    static decltype(auto) eval(mesh_type const& m, TExpr const& expr, EntityId s, expression_tag<tags::_dot, I, I>)  //
    {
        return getValue(m, std::get<0>(expr.m_args_), sw(s, 0)) * getValue(m, std::get<1>(expr.m_args_), sw(s, 0)) +
               getValue(m, std::get<0>(expr.m_args_), sw(s, 1)) * getValue(m, std::get<1>(expr.m_args_), sw(s, 1)) +
               getValue(m, std::get<0>(expr.m_args_), sw(s, 2)) * getValue(m, std::get<1>(expr.m_args_), sw(s, 2));
    }
    template <typename TExpr, int I, int K>
    static decltype(auto) eval(mesh_type const& m, TExpr const& expr, EntityId s,
                               expression_tag<tags::_cross, I, K>)  //
    {
        return getValue(m, std::get<0>(expr.m_args_), sw(s, (s.w + 1) % 3)) *
                   getValue(m, std::get<1>(expr.m_args_), sw(s, (s.w + 2) % 3)) -
               getValue(m, std::get<0>(expr.m_args_), sw(s, (s.w + 2) % 3)) *
                   getValue(m, std::get<1>(expr.m_args_), sw(s, (s.w + 1) % 3));
    }
    //    template<typename TExpr, int I> static decltype(auto)
    //    eval(mesh_type const &m, TExpr const &expr, EntityId const &s,
    //         expression_tag<tags::divides, I, VERTEX>) //
    //    AUTO_RETURN((GetValue(m, std::PopPatch<0>(expr.m_args_), s) /
    //                 _map_to(m, std::PopPatch<1>(expr.m_args_), s,
    //                 int_sequence<VERTEX, I>())))

    //    template<typename TExpr, int I> static decltype(auto)
    //    eval(mesh_type const &m, TExpr const &expr, EntityId const &s,
    //         expression_tag<tags::multiplies, I, VERTEX>) //
    //    AUTO_RETURN((GetValue(m, std::PopPatch<0>(expr.m_args_), s) *
    //                 _map_to(m, std::PopPatch<1>(expr.m_args_), s,
    //                 int_sequence<VERTEX, I>())))

    //**********************************************************************************************
    // for element-wise arithmetic operation
    template <typename TExpr, int... I>
    static decltype(auto) _invoke_helper(mesh_type const& m, TExpr const& expr, EntityId s, int_sequence<I...>) {
        return expr.m_op_(getValue(m, std::get<I>(expr.m_args_), s)...);
    }

    template <typename TExpr, typename TOP, int... I>
    static decltype(auto) eval(mesh_type const& m, TExpr const& expr, EntityId s, expression_tag<TOP, I...>) {
        return _invoke_helper(m, expr, s, make_int_sequence<sizeof...(I)>());
    }

    ///*********************************************************************************************
    /**
     * @ingroup interpolate
     * @brief basic linear interpolate
     */
    template <typename TD, typename TIDX>
    static decltype(auto) gather_impl_(mesh_type const& m, TD const& f, TIDX const& idx) {
        EntityId X = (M::_DI);
        EntityId Y = (M::_DJ);
        EntityId Z = (M::_DK);

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
    constexpr static decltype(auto) gather(mesh_type const& m, TF const& f, point_type const& r,
                                           ENABLE_IF((traits::iform<TF>::value == VERTEX))) {
        return gather_impl_(m, f, m.point_global_to_local(r, 0));
    }

    template <typename TF>
    constexpr static decltype(auto) gather(mesh_type const& m, TF const& f, point_type const& r,
                                           ENABLE_IF((traits::iform<TF>::value == EDGE))) {
        return traits::field_value_t<TF>{gather_impl_(m, f, m.point_global_to_local(r, 1)),
                                         gather_impl_(m, f, m.point_global_to_local(r, 2)),
                                         gather_impl_(m, f, m.point_global_to_local(r, 4))};
    }

    template <typename TF>
    constexpr static decltype(auto) gather(mesh_type const& m, TF const& f, point_type const& r,
                                           ENABLE_IF((traits::iform<TF>::value == FACE))) {
        return traits::field_value_t<TF>{gather_impl_(m, f, m.point_global_to_local(r, 6)),
                                         gather_impl_(m, f, m.point_global_to_local(r, 5)),
                                         gather_impl_(m, f, m.point_global_to_local(r, 3))};
    }

    template <typename TF>
    constexpr static decltype(auto) gather(mesh_type const& m, TF const& f, point_type const& x,
                                           ENABLE_IF((traits::iform<TF>::value == VOLUME))) {
        return gather_impl_(m, f, m.point_global_to_local(x, 7));
    }

   private:
    template <typename TF, typename IDX, typename TV>
    static void scatter_impl_(mesh_type const& m, TF& f, IDX const& idx, TV const& v) {
        EntityId X = (M::_DI);
        EntityId Y = (M::_DJ);
        EntityId Z = (M::_DK);

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
    static void scatter_(mesh_type const& m, int_const<VERTEX>, TF& f, TX const& x, TV const& u) {
        scatter_impl_(m, f, m.point_global_to_local(x, 0), u);
    }

    template <typename TF, typename TX, typename TV>
    static void scatter_(mesh_type const& m, int_const<EDGE>, TF& f, TX const& x, TV const& u) {
        scatter_impl_(m, f, m.point_global_to_local(x, 1), u[0]);
        scatter_impl_(m, f, m.point_global_to_local(x, 2), u[1]);
        scatter_impl_(m, f, m.point_global_to_local(x, 4), u[2]);
    }

    template <typename TF, typename TX, typename TV>
    static void scatter_(mesh_type const& m, int_const<FACE>, TF& f, TX const& x, TV const& u) {
        scatter_impl_(m, f, m.point_global_to_local(x, 6), u[0]);
        scatter_impl_(m, f, m.point_global_to_local(x, 5), u[1]);
        scatter_impl_(m, f, m.point_global_to_local(x, 3), u[2]);
    }

    template <typename TF, typename TX, typename TV>
    static void scatter_(mesh_type const& m, int_const<VOLUME>, TF& f, TX const& x, TV const& u) {
        scatter_impl_(m, f, m.point_global_to_local(x, 7), u);
    }

   public:
    template <typename TF, typename... Args>
    static void scatter(mesh_type const& m, TF& f, Args&&... args) {
        scatter_(m, traits::iform<TF>(), f, std::forward<Args>(args)...);
    }

   private:
    template <typename TV>
    static auto sample_(mesh_type const& m, EntityId s, TV& v) {
        return v;
    }

    template <typename TV, int N>
    static auto sample_(mesh_type const& m, EntityId s, nTuple<TV, N> const& v) {
        return v[s.w % N];
    }

    //    template <typename TV, int N>
    //    static auto sample_(mesh_type const& m, EntityId s, nTuple<TV, N> const& v) {
    //        return v[s.w % N];
    //    }
    //
    //    template <typename TV>
    //    static auto sample_(mesh_type const& m, EntityId s, nTuple<TV, 3> const& v) {
    //        return v[M::sub_index(s)];
    //    }
    //
    //    template <typename TV>
    //    static auto sample_(mesh_type const& m, EntityId s, nTuple<TV, 3> const& v) {
    //        return v[M::sub_index(s)];
    //    }
    //
    //    template<typename M,int IFORM,  typename TV>
    //    static   TV sample_(M const & m,int_const< IFORM>, EntityId_type
    //    s,
    //                                       TV const &v) { return v; }

   public:
    //    template<typename M,int IFORM,  typename TV>
    //    static   auto generate(TI const &s, TV const &v)
    //    AUTO_RETURN((sample_(M const & m,int_const< IFORM>(), s, v)))

    template <typename TV>
    static decltype(auto) sample(mesh_type const& m, EntityId s, TV const& v) {
        return sample_(m, s, v);
    }

    ///*********************************************************************************************
    /// @name general_algebra General algebra
    /// @{

    template <typename T>
    static decltype(auto) getValue(mesh_type const& m, T const& v, EntityId s,
                                   ENABLE_IF((std::is_arithmetic<T>::value))) {
        return v;
    }

    //    template <typename U>
    //    static decltype(auto) getValue(mesh_type const& m, U& f, EntityId s,
    //                                   ENABLE_IF(traits::is_primary_field<U>::value)) {
    //        return f.at(s);
    //    };
    //    template <typename... U>
    //    static decltype(auto) getValue(mesh_type const& m, Field<U...> const& f, EntityId s) {
    //        return f.at(s);
    //    };
    //
    //    template <typename... U>
    //    static decltype(auto) getValue(mesh_type const& m, Field<U...> const& f, point_type const& x) {
    //        return gather(m, f, x);
    //    };

    template <typename M, typename V, int I, int D>
    static V const& getValue(mesh_type const& m, Field<M, V, I, D> const& f, EntityId s) {
        return f[EntityIdCoder::SubIndex<I, D>(s)](s.x, s.y, s.z);
    };
    template <typename M, typename V, int I, int D>
    static V& getValue(mesh_type const& m, Field<M, V, I, D>& f, EntityId s) {
        return f[EntityIdCoder::SubIndex<I, D>(s)](s.x, s.y, s.z);
    };

    template <typename TOP, typename... T>
    static auto getValue(mesh_type const& m, Expression<TOP, T...> const& expr, EntityId s) {
        return eval(m, expr, s, expression_tag<TOP, traits::iform<T>::value...>());
    }

    template <typename TFun>
    static auto getValue(mesh_type const& m, TFun const& fun, EntityId s,
                         ENABLE_IF(simpla::concept::is_callable<TFun(EntityId)>::value)) {
        return sample(m, s, fun(s));
    }

    template <typename TFun>
    static auto getValue(mesh_type const& m, TFun const& fun, EntityId s,
                         ENABLE_IF(simpla::concept::is_callable<TFun(point_type const&)>::value)) {
        return sample(m, s, fun(m.point(s)));
    }

    //    template <int IFORM, typename TExpr>
    //    static auto getValue(std::integral_constant<int, IFORM> const&, mesh_type const& m, TExpr const& expr,
    //    index_type i,
    //                         index_type j, index_type k, unsigned int n, unsigned int d) {
    //        return getValue(m, expr, EntityIdCoder::Pack<IFORM>(i, j, k, n, d));
    //    }

    //**********************************************************************************************

    template <typename TField, typename TOP, typename... Args>
    static void foreach_(mesh_type const& m, TField& self, Range<EntityId> const& r, TOP const& op, Args&&... args) {
        r.foreach ([&](EntityId s) { op(getValue(m, self, s), getValue(m, std::forward<Args>(args), s)...); });
    }
    template <typename... Args>
    static void foreach (Args&&... args) {
        foreach_(std::forward<Args>(args)...);
    }

    //    template <typename TField, typename... Args>
    //    static void foreach (mesh_type const& m, TField & self, mesh::MeshZoneTag const& tag, Args && ... args) {
    //        foreach_(m, self, m.range(tag, traits::iform<TField>::value, traits::dof<TField>::value),
    //                 std::forward<Args>(args)...);
    //    }
    //    template <typename TField, typename... Args>
    //    static void foreach (mesh_type const& m, TField & self, Args && ... args) {
    //        foreach_(m, self, m.range(SP_ES_ALL, traits::iform<TField>::value, traits::dof<TField>::value),
    //                 std::forward<Args>(args)...);
    //    }

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

    Real RBF(mesh_type const& m, point_type const& x0, point_type const& x1, vector_type const& a) {
        vector_type r;
        r = (x1 - x0) / a;
        // @NOTE this is not  an exact  RBF
        return (1.0 - std::abs(r[0])) * (1.0 - std::abs(r[1])) * (1.0 - std::abs(r[2]));
    }

    Real RBF(mesh_type const& m, point_type const& x0, point_type const& x1, Real const& a) {
        return (1.0 - m.distance(x1, x0) / a);
    }
};
//    template <int DOF, typename... U>
//    static void Assign(mesh_type const& m, Field<mesh_type, U...>& f, EntityId
//    s,
//                       nTuple<U, DOF> const& v) {
//        for (int i = 0; i < DOF; ++i) { f[M::sw(s, i)] = v[i]; }
//    }

////    template <typename... U>
////    static void assign(mesh_type const& m, Field<U...>& f,
////                       EntityId s, nTuple<U, 3> const& v) {
////        for (int i = 0; i < DOF; ++i) { f[M::sw(s, i)] = v[M::sub_index(s)]; }
////    }
////
////    template <typename V, int DOF, int... I, typename U>
////    static void assign(mesh_type const& m, Field<mesh_type, V, FACE, DOF, I...>& f,
////                       EntityId s, nTuple<U, 3> const& v) {
////        for (int i = 0; i < DOF; ++i) { f[M::sw(s, i)] = v[M::sub_index(s)]; }
////    }
////
////    template <typename V, int DOF, int... I, typename U>
////    static void assign(mesh_type const& m, Field<mesh_type, V, VOLUME, DOF, I...>& f,
////                       EntityId s, nTuple<U, DOF> const& v) {
////        for (int i = 0; i < DOF; ++i) { f[M::sw(s, i)] = v[i]; }
//    }
//
//    template <typename V, int IFORM, int DOF, int... I, typename U>
//    static void Assign(mesh_type const& m, Field<mesh_type, V, IFORM, DOF, I...>& f,
//                       EntityId s, U const& v) {
//        for (int i = 0; i < DOF; ++i) { f[M::sw(s, i)] = v; }
//    }

// template <typename TV, typename TM, int IFORM, int DOF>
// constexpr Real   calculator<Field<TV, TM, IFORM, DOF>>::m_p_curl_factor[3];
}  // namespace calculus
}  // namespace simpla { {

#endif /* FDM_H_ */
