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

#include "../../toolbox/sp_def.h"

#include "simpla/mpl/macro.h"
#include "simpla/mpl/type_traits.h"
#include "simpla/algebra/Algebra.h"
#include "simpla/algebra/Expression.h"
#include "simpla/algebra/Calculus.h"

#include "../ManifoldTraits.h"

namespace simpla { namespace algebra { namespace declare
{
template<typename, typename, size_type ...I> struct Field_;
}}}//namespace simpla { namespace algebra { namespace declare


namespace simpla { namespace manifold { namespace schemes
{
namespace al= simpla::algebra;
namespace algd= simpla::algebra::declare;

namespace algt= simpla::algebra::tags;
namespace at= simpla::algebra::traits;
namespace st= simpla::traits;

/**
 * @ingroup diff_scheme
 *
 * finite volume
 */
template<typename TM>
struct CalculusPolicy
{
    typedef CalculusPolicy<TM> this_type;
    typedef TM mesh_type;
    typedef mesh::MeshEntityIdCoder M;
    typedef mesh::MeshEntityId MeshEntityId;
public:

    CalculusPolicy() {}

    virtual ~CalculusPolicy() {}

public:

//    template<typename ...T>
//    inline static  algebra:: traits::value_type_t<Field<T...> >
//    get_value(m,mesh_type const & m,Field<T...> &expr, MeshEntityId const& s) { return get_value(m,expr, s); }
//
//    template<typename ...T>
//    inline static  algebra:: traits::value_type_t<Field<T...> >
//    get_value(m,mesh_type const & m,Field<T...> const &expr, MeshEntityId const& s) { return get_value(m,expr, s); }
//
//    template<typename T>
//    inline static  T &get_value(m,T &f, MeshEntityId const& s) { return f; }
//
//
//    template<typename T>
//    inline static  T const &get_value(m,T const &f, MeshEntityId const& s) { return f; }



//    template<typename TV, typename OM, size_t IFORM> inline 
//    typename traits::value_type<Field<TV, OM, index_const<IFORM>>>::type &
//    get_value(m,mesh_type const & m,Field<TV, OM, index_const<IFORM>> &f, MeshEntityId const& s) { return f[s]; };

    template<typename U, typename M, size_type...I> inline static at::value_type_t<U>
    get_value(mesh_type const &m, algd::Field_<U, M, I...> const &f, MeshEntityId const &s) { return f[s]; };


private:

    template<typename FExpr> inline static at::value_type_t<FExpr>
    get_v(mesh_type const &m, FExpr const &f, MeshEntityId const s) { return get_value(m, f, s) * m.volume(s); }

    template<typename FExpr> inline static at::value_type_t<FExpr>
    get_d(mesh_type const &m, FExpr const &f, MeshEntityId const s) { return get_value(m, f, s) * m.dual_volume(s); }

public:

    //***************************************************************************************************
    // Exterior algebra
    //***************************************************************************************************

    //! grad<0>
    template<typename T>
    static inline at::value_type_t<algd::Expression<algt::ExteriorDerivative, T>>
    get_value(mesh_type const &m, algd::Expression<algt::ExteriorDerivative, T> const &f,
              MeshEntityId const &s, index_sequence<VERTEX>)
    {
        MeshEntityId D = M::delta_index(s);
        return (get_v(m, std::get<0>(f.m_args_), s + D) - get_v(m, std::get<0>(f.m_args_), s - D)) * m.inv_volume(s);
    }


    //! curl<1>
    template<typename T>
    static inline at::value_type_t<algd::Expression<algt::ExteriorDerivative, T>>
    get_value(mesh_type const &m, algd::Expression<algt::ExteriorDerivative, T> const &expr,
              MeshEntityId const &s, index_sequence<EDGE>)
    {

        MeshEntityId X = M::delta_index(M::dual(s));
        MeshEntityId Y = M::rotate(X);
        MeshEntityId Z = M::inverse_rotate(X);


        return ((get_v(m, std::get<0>(expr.m_args_), s + Y) - get_v(m, std::get<0>(expr.m_args_), s - Y))
                - (get_v(m, std::get<0>(expr.m_args_), s + Z) - get_v(m, std::get<0>(expr.m_args_), s - Z))
               ) * m.inv_volume(s);


    }


    //! div<2>
    template<typename T>
    static inline at::value_type_t<algd::Expression<algt::ExteriorDerivative, T>>
    get_value(mesh_type const &m, algd::Expression<algt::ExteriorDerivative, T> const &expr,
              MeshEntityId const &s, index_sequence<FACE>)
    {
        return (get_v(m, std::get<0>(expr.m_args_), s + M::_DI) - get_v(m, std::get<0>(expr.m_args_), s - M::_DI)
                + get_v(m, std::get<0>(expr.m_args_), s + M::_DJ) - get_v(m, std::get<0>(expr.m_args_), s - M::_DJ)
                + get_v(m, std::get<0>(expr.m_args_), s + M::_DK) - get_v(m, std::get<0>(expr.m_args_), s - M::_DK)
               ) * m.inv_volume(s);

    }


    //! div<1>
    template<typename T>
    static inline at::value_type_t<algd::Expression<algt::CodifferentialDerivative, T>>
    get_value(mesh_type const &m, algd::Expression<algt::CodifferentialDerivative, T> const &expr,
              MeshEntityId const &s, index_sequence<EDGE>)
    {
        return -(get_d(m, std::get<0>(expr.m_args_), s + M::_DI) - get_d(m, std::get<0>(expr.m_args_), s - M::_DI)
                 + get_d(m, std::get<0>(expr.m_args_), s + M::_DJ) - get_d(m, std::get<0>(expr.m_args_), s - M::_DJ)
                 + get_d(m, std::get<0>(expr.m_args_), s + M::_DK) - get_d(m, std::get<0>(expr.m_args_), s - M::_DK)
        ) * m.inv_dual_volume(s);
    }

    //! curl<2>
    template<typename T>
    static inline at::value_type_t<algd::Expression<algt::CodifferentialDerivative, T>>
    get_value(mesh_type const &m, algd::Expression<algt::CodifferentialDerivative, T> const &expr,
              MeshEntityId const &s, index_sequence<FACE>)
    {

        MeshEntityId X = M::delta_index(s);
        MeshEntityId Y = M::rotate(X);
        MeshEntityId Z = M::inverse_rotate(X);

        return -((get_d(m, std::get<0>(expr.m_args_), s + Y) - get_d(m, std::get<0>(expr.m_args_), s - Y))
                 - (get_d(m, std::get<0>(expr.m_args_), s + Z) - get_d(m, std::get<0>(expr.m_args_), s - Z))
        ) * m.inv_dual_volume(s);
    }


    //! grad<3>
    template<typename T>
    static inline at::value_type_t<algd::Expression<algt::CodifferentialDerivative, T>>
    get_value(mesh_type const &m, algd::Expression<algt::CodifferentialDerivative, T> const &expr,
              MeshEntityId const &s, index_sequence<VOLUME>)
    {
        MeshEntityId D = M::delta_index(M::dual(s));
        return -(get_d(m, std::get<0>(expr.m_args_), s + D) - get_d(m, std::get<0>(expr.m_args_), s - D)) *
               m.inv_dual_volume(s);
    }

    //! *Form<IR> => Form<N-IL>


    template<typename T, size_t I>
    static inline at::value_type_t<algd::Expression<algt::HodgeStar, T> >
    get_value(mesh_type const &m, algd::Expression<algt::HodgeStar, T> const &expr,
              MeshEntityId const &s, index_sequence<I>)
    {
        auto const &l = std::get<0>(expr.m_args_);
        int i = M::iform(s);
        MeshEntityId X = (i == VERTEX || i == VOLUME) ? M::_DI : M::delta_index(M::dual(s));
        MeshEntityId Y = M::rotate(X);
        MeshEntityId Z = M::inverse_rotate(X);


        return (
                       get_v(m, l, ((s - X) - Y) - Z) +
                       get_v(m, l, ((s - X) - Y) + Z) +
                       get_v(m, l, ((s - X) + Y) - Z) +
                       get_v(m, l, ((s - X) + Y) + Z) +
                       get_v(m, l, ((s + X) - Y) - Z) +
                       get_v(m, l, ((s + X) - Y) + Z) +
                       get_v(m, l, ((s + X) + Y) - Z) +
                       get_v(m, l, ((s + X) + Y) + Z)

               ) * m.inv_dual_volume(s) * 0.125;


    };

    ////***************************************************************************************************
    //! p_curl<1>
    static constexpr Real m_p_curl_factor_[3] = {0, 1, -1};

    template<typename T, size_t I>
    static inline at::value_type_t<algd::Expression<algt::P_ExteriorDerivative<I>, T>>
    get_value(mesh_type const &m, algd::Expression<algt::P_ExteriorDerivative<I>, T> const &expr,
              MeshEntityId const &s, index_sequence<EDGE>)
    {
        return (get_v(m, std::get<0>(expr.m_args_), s + M::DI(I)) -
                get_v(m, std::get<0>(expr.m_args_), s - M::DI(I))
               ) * m.inv_volume(s) * m_p_curl_factor_[(I + 3 - M::sub_index(s)) % 3];
    }


    template<typename T, size_t I>
    static inline at::value_type_t<algd::Expression<algt::P_CodifferentialDerivative<I>, T>>
    get_value(mesh_type const &m, algd::Expression<algt::P_CodifferentialDerivative<I>, T> const &expr,
              MeshEntityId const &s, index_sequence<FACE>)
    {

        return (get_v(m, std::get<0>(expr.m_args_), s + M::DI(I)) -
                get_v(m, std::get<0>(expr.m_args_), s - M::DI(I))
               ) * m.inv_dual_volume(s) * m_p_curl_factor_[(I + 3 - M::sub_index(s)) % 3];
    }

////***************************************************************************************************
//
////! map_to
//    template<typename T, size_t I>
//    inline static T
//    mapto(mesh_type const &m, T const &r, MeshEntityId const &s, index_sequence<VERTEX, I>,
//          st::is_primary_t<T> *_p = nullptr) { return r; }
//
//    template<typename TF, size_t I>
//    inline static at::value_type_t<TF>
//    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<I, I>,
//          std::enable_if_t<!st::is_primary<TF>::value>
//          *_p = nullptr) { return get_value(m, expr, s); }

    template<typename TV, typename OM, size_t I, int DOF> inline static TV
    mapto(mesh_type const &m, algd::Field_<TV, OM, I, DOF> const &f, MeshEntityId const &s,
          index_sequence<I, I>) { return f[s]; };

    template<typename TF> static inline at::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<VERTEX, EDGE>)
    {
        int n = M::sub_index(s);
        MeshEntityId X = M::delta_index(s);
        auto l = get_value(m, expr, sw(s - X, n));
        auto r = get_value(m, expr, sw(s + X, n));
        return (l + r) * 0.5;
//        return (get_value(m, expr, sw(s - X, n)) + get_value(m, expr, sw(s + X, n))) * 0.5;


    }


    template<typename TF> static inline at::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<VERTEX, FACE>)
    {
        int n = M::sub_index(s);

        auto X = M::delta_index(M::dual(s));
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);

        return (
                       get_value(m, expr, sw(s - Y - Z, n)) +
                       get_value(m, expr, sw(s - Y + Z, n)) +
                       get_value(m, expr, sw(s + Y - Z, n)) +
                       get_value(m, expr, sw(s + Y + Z, n))
               ) * 0.25;
    }


    template<typename TF> static inline at::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<VERTEX, VOLUME>)
    {
        auto const &l = expr;

        auto X = M::DI(0, s);
        auto Y = M::DI(1, s);
        auto Z = M::DI(2, s);

        return (get_value(m, l, s - X - Y - Z) +
                get_value(m, l, s - X - Y + Z) +
                get_value(m, l, s - X + Y - Z) +
                get_value(m, l, s - X + Y + Z) +
                get_value(m, l, s + X - Y - Z) +
                get_value(m, l, s + X - Y + Z) +
                get_value(m, l, s + X + Y - Z) +
                get_value(m, l, s + X + Y + Z)
               ) * 0.125;
    }


    template<typename TF> static inline at::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<EDGE, VERTEX>)
    {
        MeshEntityId X = M::DI(s.w, s);
        return (get_value(m, expr, sw(s - X, 0)) + get_value(m, expr, sw(s + X, 0))) * 0.5;
    }


    template<typename TF> static inline at::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<FACE, VERTEX>)
    {


        MeshEntityId Y = M::DI((s.w + 1) % 3, s);
        MeshEntityId Z = M::DI((s.w + 2) % 3, s);

        return (get_value(m, expr, sw(s - Y - Z, 0)) +
                get_value(m, expr, sw(s - Y + Z, 0)) +
                get_value(m, expr, sw(s + Y - Z, 0)) +
                get_value(m, expr, sw(s + Y + Z, 0))) * 0.25;

//        MeshEntityId X = M::_DI;
//        MeshEntityId Y = M::_DJ;
//        MeshEntityId Z = M::_DK;
//        return nTuple<typename traits::value_type<TF>::type, 3>
//                {
//                        static_cast<typename traits::value_type<TF>::type>(get_value(m, l, (s - Y - Z)) +
//                                                                           get_value(m, l, (s - Y + Z)) +
//                                                                           get_value(m, l, (s + Y - Z)) +
//                                                                           get_value(m, l, (s + Y + Z)) * 0.25),
//                        static_cast<typename traits::value_type<TF>::type>(get_value(m, l, (s - Z - X)) +
//                                                                           get_value(m, l, (s - Z + X)) +
//                                                                           get_value(m, l, (s + Z - X)) +
//                                                                           get_value(m, l, (s + Z + X)) * 0.25),
//                        static_cast<typename traits::value_type<TF>::type>(get_value(m, l, (s - X - Y)) +
//                                                                           get_value(m, l, (s - X + Y)) +
//                                                                           get_value(m, l, (s + X - Y)) +
//                                                                           get_value(m, l, (s + X + Y)) * 0.25)
//                };


    }


    template<typename TF>
    static inline at::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<VOLUME, VERTEX>)
    {
        auto const &l = expr;

        auto X = M::DI(0, s);
        auto Y = M::DI(1, s);
        auto Z = M::DI(2, s);

        return (
                       get_value(m, l, ((s - X - Y - Z))) +
                       get_value(m, l, ((s - X - Y + Z))) +
                       get_value(m, l, ((s - X + Y - Z))) +
                       get_value(m, l, ((s - X + Y + Z))) +
                       get_value(m, l, ((s + X - Y - Z))) +
                       get_value(m, l, ((s + X - Y + Z))) +
                       get_value(m, l, ((s + X + Y - Z))) +
                       get_value(m, l, ((s + X + Y + Z)))
               ) * 0.125;
    }


    template<typename TF>
    static inline at::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<VOLUME, FACE>)
    {
        auto X = M::delta_index(M::dual(s));

        return (get_value(m, expr, s - X) + get_value(m, expr, s + X)) * 0.5;
    }


    template<typename TF>
    static inline at::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<VOLUME, EDGE>)
    {
        auto const &l = expr;
        auto X = M::delta_index(s);
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);

        return (
                       get_value(m, l, s - Y - Z) +
                       get_value(m, l, s - Y + Z) +
                       get_value(m, l, s + Y - Z) +
                       get_value(m, l, s + Y + Z)
               ) * 0.25;
    }


    template<typename TF>
    static inline at::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<FACE, VOLUME>)
    {

        MeshEntityId X = M::DI(s.w, s);

        return (get_value(m, expr, sw(s - X, 0)) + get_value(m, expr, sw(s + X, 0))) * 0.5;


    }


    template<typename TF>
    static inline at::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<EDGE, VOLUME>)
    {
//        auto const &l = expr;
//
//        auto X = M::DI(0, s);
//        auto Y = M::DI(1, s);
//        auto Z = M::DI(2, s);

        MeshEntityId Y = M::DI((s.w + 1) % 3, s);
        MeshEntityId Z = M::DI((s.w + 1) % 3, s);

        return (get_value(m, expr, sw(s - Y - Z, 0)) + get_value(m, expr, sw(s - Y + Z, 0)) +
                get_value(m, expr, sw(s + Y - Z, 0)) + get_value(m, expr, sw(s + Y + Z, 0))) * 0.25;


    }


    //***************************************************************************************************
    //
    //! Form<IL> ^ Form<IR> => Form<IR+IL>
    template<typename ...T, size_t IL, size_t IR>
    static inline at::value_type_t<algd::Expression<algt::Wedge, T...>>
    get_value(mesh_type const &m, algd::Expression<algt::Wedge, T...> const &expr,
              MeshEntityId const &s, index_sequence<IL, IR>)
    {
        return m.inner_product(mapto(m, std::get<0>(expr.m_args_), s, index_sequence<IL, IR + IL>()),
                               mapto(m, std::get<1>(expr.m_args_), s, index_sequence<IR, IR + IL>()),
                               s);

    }


    template<typename TL, typename TR>
    static inline at::value_type_t<algd::Expression<algt::Wedge, TL, TR>>
    get_value(mesh_type const &m, algd::Expression<algt::Wedge, TL, TR> const &expr,
              MeshEntityId const &s, index_sequence<EDGE, EDGE>)
    {
        auto const &l = std::get<0>(expr.m_args_);
        auto const &r = std::get<1>(expr.m_args_);

        auto Y = M::delta_index(M::rotate(M::dual(s)));
        auto Z = M::delta_index(
                M::inverse_rotate(M::dual(s)));

        return ((get_value(m, l, s - Y) + get_value(m, l, s + Y))
                * (get_value(m, l, s - Z) + get_value(m, l, s + Z)) * 0.25);
    }

    static MeshEntityId sw(MeshEntityId s, u_int16_t w)
    {
        s.w = w;
        return s;
    }

    template<typename TL, typename TR, size_t I>
    static inline at::value_type_t<algd::Expression<algt::Cross, TL, TR>>
    get_value(mesh_type const &m, algd::Expression<algt::Cross, TL, TR> const &expr,
              MeshEntityId const &s, index_sequence<I, I>)
    {
        return get_value(m, std::get<0>(expr.m_args_), sw(s, (s.w + 1) % 3)) *
               get_value(m, std::get<1>(expr.m_args_), sw(s, (s.w + 2) % 3)) -
               get_value(m, std::get<0>(expr.m_args_), sw(s, (s.w + 2) % 3)) *
               get_value(m, std::get<1>(expr.m_args_), sw(s, (s.w + 1) % 3));
    }

    template<typename TL, typename TR, size_t I>
    static inline at::value_type_t<algd::Expression<algt::Dot, TL, TR>>
    get_value(mesh_type const &m, algd::Expression<algt::Dot, TL, TR> const &expr,
              MeshEntityId const &s, index_sequence<I, I>)
    {
        return get_value(m, std::get<0>(expr.m_args_), sw(s, 0)) * get_value(m, std::get<1>(expr.m_args_), sw(s, 0)) +
               get_value(m, std::get<0>(expr.m_args_), sw(s, 1)) * get_value(m, std::get<1>(expr.m_args_), sw(s, 1)) +
               get_value(m, std::get<0>(expr.m_args_), sw(s, 2)) * get_value(m, std::get<1>(expr.m_args_), sw(s, 2));
    }


    template<typename TL, typename TR, size_t I>
    static inline at::value_type_t<algd::Expression<algt::divides, TL, TR>>
    get_value(mesh_type const &m, algd::Expression<algt::divides, TL, TR> const &expr,
              MeshEntityId const &s, index_sequence<I, VERTEX>)
    {
        return get_value(m, std::get<0>(expr.m_args_), s) /
               mapto(m, std::get<1>(expr.m_args_), s, index_sequence<VERTEX, I>());
    }

    template<typename TL, typename TR, size_t I>
    static inline at::value_type_t<algd::Expression<algt::multiplies, TL, TR>>
    get_value(mesh_type const &m, algd::Expression<algt::multiplies, TL, TR> const &expr,
              MeshEntityId const &s, index_sequence<I, VERTEX>
    )
    {
        return get_value(m, std::get<0>(expr.m_args_), s) *
               mapto(m, std::get<1>(expr.m_args_), s, index_sequence<VERTEX, I>());
    }


    template<typename ...T, size_t ...I>
    static inline at::value_type_t<algd::Expression<algt::MapTo, T...> >
    get_value(mesh_type const &m, algd::Expression<algt::MapTo, T...> const &expr,
              MeshEntityId const &s, index_sequence<I...>)
    {
        return mapto(m, std::get<0>(expr.m_args_), s, index_sequence<I...>());
    };


    ///***************************************************************************************************
    /// @name general_algebra General algebra
    /// @{
    ///***************************************************************************************************


    template<typename T>
    inline static T const &
    get_value(mesh_type const &m, T const &v, MeshEntityId const &s,
              ENABLE_IF(std::is_arithmetic<T>::value)) { return v; }


//    template<typename T>
//    inline static traits::primary_type_t<T>
//    get_value(mesh_type const &m, T const &v, MeshEntityId const &s, ENABLE_IF(at::is_nTuple<T>::value))
//    {
////        traits::primary_type_t<T> res;
//        res = v;
//        return std::move(res);
//    }

    template<typename TOP, typename ... T>
    inline static at::value_type_t<algd::Expression<TOP, T...> >
    get_value(mesh_type const &m, algd::Expression<TOP, T...> const &expr,
              MeshEntityId const &s)
    {
        return get_value(m, expr, s, at::iform_list_t<T...>());
    }

    //******************************************************************************************************************
    // for element-wise arithmetic operation
    template<typename TOP, typename ...T, size_t ... I> inline static
    at::value_type_t<algd::Expression<TOP, T...> >
    _invoke_helper(mesh_type const &m, algd::Expression<TOP, T...> const &expr,
                   MeshEntityId const &s, index_sequence<I...>)
    {
        return expr.m_op_(mapto(m, std::get<I>(expr.m_args_), s,
                                index_sequence<at::iform<T>::value,
                                        at::iform < algd::Expression < TOP, T...> > ::value > ())...);
    };


    template<typename TOP, typename ... T, size_t ...I>
    inline static at::value_type_t<algd::Expression<TOP, T...>>
    get_value(mesh_type const &m, algd::Expression<TOP, T...> const &expr,
              MeshEntityId const &s, index_sequence<I...>)
    {
        return _invoke_helper(m, expr, s, index_sequence_for<T...>());
    }
    //******************************************************************************************************************


};// struct DiffScheme<TGeo, diff_scheme::tags::finite_volume>

template<typename TM> constexpr Real CalculusPolicy<TM>::m_p_curl_factor_[3];
//template<typename TM> static  Real  FiniteVolume<TM, std::enable_if_t<std::is_base_of<mesh_as::MeshEntityIdCoder, TM>::entity>>::m_p_curl_factor2_[3];
}}}// namespace simpla

#endif /* FDM_H_ */
