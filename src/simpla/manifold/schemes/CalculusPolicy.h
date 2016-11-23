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


#include "../../toolbox/macro.h"
#include "../../sp_def.h"
#include "../../toolbox/type_traits.h"
#include "../../toolbox/ExpressionTemplate.h"
#include "../Calculus.h"
#include "../ManifoldTraits.h"


namespace simpla { namespace manifold { namespace schemes
{
using namespace simpla::mesh;

namespace ct=::simpla::calculus::tags;
namespace st=::simpla::traits;

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
//    inline static  traits::value_type_t<Field<T...> >
//    eval(m,mesh_type const & m,Field<T...> &expr, MeshEntityId const& s) { return eval(m,expr, s); }
//
//    template<typename ...T>
//    inline static  traits::value_type_t<Field<T...> >
//    eval(m,mesh_type const & m,Field<T...> const &expr, MeshEntityId const& s) { return eval(m,expr, s); }
//
//    template<typename T>
//    inline static  T &eval(m,T &f, MeshEntityId const& s) { return f; }
//
//
//    template<typename T>
//    inline static  T const &eval(m,T const &f, MeshEntityId const& s) { return f; }



//    template<typename TV, typename OM, size_t IFORM> inline 
//    typename traits::value_type<Field<TV, OM, index_const<IFORM>>>::type &
//    eval(m,mesh_type const & m,Field<TV, OM, index_const<IFORM>> &f, MeshEntityId const& s) { return f[s]; };

    template<typename TV, typename OM, size_t IFORM, size_type DOF> inline static TV
    eval(mesh_type const &m, Field<TV, OM, index_const<IFORM>, index_const<DOF> > const &f,
         MeshEntityId const &s) { return f[s]; };


private:

    template<typename FExpr> inline static traits::value_type_t<FExpr>
    get_v(mesh_type const &m, FExpr const &f, MeshEntityId const s) { return eval(m, f, s) * m.volume(s); }

    template<typename FExpr> inline static traits::value_type_t<FExpr>
    get_d(mesh_type const &m, FExpr const &f, MeshEntityId const s) { return eval(m, f, s) * m.dual_volume(s); }

public:

    //***************************************************************************************************
    // Exterior algebra
    //***************************************************************************************************

    //! grad<0>
    template<typename T>
    static inline traits::value_type_t<Field<Expression<ct::ExteriorDerivative, T>>>
    eval(mesh_type const &m, Field<Expression<ct::ExteriorDerivative, T>> const &f, MeshEntityId const &s,
         index_sequence<mesh::VERTEX>)
    {
        MeshEntityId D = M::delta_index(s);
        return (get_v(m, std::get<0>(f.args), s + D) - get_v(m, std::get<0>(f.args), s - D)) * m.inv_volume(s);
    }


    //! curl<1>
    template<typename T>
    static inline traits::value_type_t<Field<Expression<ct::ExteriorDerivative, T>>>
    eval(mesh_type const &m, Field<Expression<ct::ExteriorDerivative, T> > const &expr, MeshEntityId const &s,
         index_sequence<EDGE>)
    {

        MeshEntityId X = M::delta_index(M::dual(s));
        MeshEntityId Y = M::rotate(X);
        MeshEntityId Z = M::inverse_rotate(X);


        return ((get_v(m, std::get<0>(expr.args), s + Y) - get_v(m, std::get<0>(expr.args), s - Y))
                - (get_v(m, std::get<0>(expr.args), s + Z) - get_v(m, std::get<0>(expr.args), s - Z))
               ) * m.inv_volume(s);


    }


    //! div<2>
    template<typename T>
    static inline traits::value_type_t<Field<Expression<ct::ExteriorDerivative, T>>>
    eval(mesh_type const &m, Field<Expression<ct::ExteriorDerivative, T> > const &expr, MeshEntityId const &s,
         index_sequence<FACE>)
    {
        return (get_v(m, std::get<0>(expr.args), s + M::_DI) - get_v(m, std::get<0>(expr.args), s - M::_DI)
                + get_v(m, std::get<0>(expr.args), s + M::_DJ) - get_v(m, std::get<0>(expr.args), s - M::_DJ)
                + get_v(m, std::get<0>(expr.args), s + M::_DK) - get_v(m, std::get<0>(expr.args), s - M::_DK)
               ) * m.inv_volume(s);

    }


    //! div<1>
    template<typename T>
    static inline traits::value_type_t<Field<Expression<ct::CodifferentialDerivative, T>>>
    eval(mesh_type const &m, Field<Expression<ct::CodifferentialDerivative, T>> const &expr, MeshEntityId const &s,
         index_sequence<EDGE>)
    {
        return -(get_d(m, std::get<0>(expr.args), s + M::_DI) - get_d(m, std::get<0>(expr.args), s - M::_DI)
                 + get_d(m, std::get<0>(expr.args), s + M::_DJ) - get_d(m, std::get<0>(expr.args), s - M::_DJ)
                 + get_d(m, std::get<0>(expr.args), s + M::_DK) - get_d(m, std::get<0>(expr.args), s - M::_DK)
        ) * m.inv_dual_volume(s);
    }

    //! curl<2>
    template<typename T>
    static inline traits::value_type_t<Field<Expression<ct::CodifferentialDerivative, T>>>
    eval(mesh_type const &m, Field<Expression<ct::CodifferentialDerivative, T>> const &expr, MeshEntityId const &s,
         index_sequence<FACE>)
    {

        MeshEntityId X = M::delta_index(s);
        MeshEntityId Y = M::rotate(X);
        MeshEntityId Z = M::inverse_rotate(X);

        return -((get_d(m, std::get<0>(expr.args), s + Y) - get_d(m, std::get<0>(expr.args), s - Y))
                 - (get_d(m, std::get<0>(expr.args), s + Z) - get_d(m, std::get<0>(expr.args), s - Z))
        ) * m.inv_dual_volume(s);
    }


    //! grad<3>
    template<typename T>
    static inline traits::value_type_t<Field<Expression<ct::CodifferentialDerivative, T>>>
    eval(mesh_type const &m, Field<Expression<ct::CodifferentialDerivative, T> > const &expr, MeshEntityId const &s,
         index_sequence<VOLUME>)
    {
        MeshEntityId D = M::delta_index(M::dual(s));
        return -(get_d(m, std::get<0>(expr.args), s + D) - get_d(m, std::get<0>(expr.args), s - D)) *
               m.inv_dual_volume(s);
    }

    //! *Form<IR> => Form<N-IL>


    template<typename T, size_t I>
    static inline traits::value_type_t<Field<Expression<ct::HodgeStar, T> >>
    eval(mesh_type const &m, Field<Expression<ct::HodgeStar, T>> const &expr, MeshEntityId const &s,
         index_sequence<I>)
    {
        auto const &l = std::get<0>(expr.args);
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
    static inline traits::value_type_t<Field<Expression<ct::P_ExteriorDerivative<I>, T>>>
    eval(mesh_type const
         &m, Field<Expression<ct::P_ExteriorDerivative<I>, T>> const &expr,
         MeshEntityId const &s,
         index_sequence<EDGE>
    )
    {
        return (get_v(m, std::get<0>(expr.args), s + M::DI(I)) -
                get_v(m, std::get<0>(expr.args), s - M::DI(I))
               ) * m.inv_volume(s) * m_p_curl_factor_[(I + 3 - M::sub_index(s)) % 3];
    }


    template<typename T, size_t I>
    static inline traits::value_type_t<Field<Expression<ct::P_CodifferentialDerivative<I>, T>>>
    eval(mesh_type const
         &m, Field<Expression<ct::P_CodifferentialDerivative<I>, T>> const &expr,
         MeshEntityId const &s,
         index_sequence<FACE>
    )
    {

        return (get_v(m, std::get<0>(expr.args), s + M::DI(I)) -
                get_v(m, std::get<0>(expr.args), s - M::DI(I))
               ) * m.inv_dual_volume(s) * m_p_curl_factor_[(I + 3 - M::sub_index(s)) % 3];
    }

////***************************************************************************************************
//
////! map_to
    template<typename T, size_t I>
    inline static T
    mapto(mesh_type const &m, T const &r, MeshEntityId const &s, index_sequence<VERTEX, I>,
          st::is_primary_t<T> *_p = nullptr) { return r; }

    template<typename TF, size_t I>
    inline static traits::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<I, I>,
          std::enable_if_t<!st::is_primary<TF>::value>
          *_p = nullptr) { return eval(m, expr, s); }

    template<typename TV, typename OM, size_t I, int DOF>
    inline static TV
    mapto(mesh_type const &m, Field<TV, OM, index_const<I>, index_const<DOF> > const &f, MeshEntityId const &s,
          index_sequence<I, I>) { return f[s]; };

    template<typename TF>
    static inline traits::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<VERTEX, EDGE>)
    {
        MeshEntityId X = M::delta_index(s);
        return (eval(m, expr, s - X) + eval(m, expr, s + X)) * 0.5;
    }


    template<typename TV, typename ...Others> static inline TV
    mapto(mesh_type const &m, Field<nTuple<TV, 3>, Others...> const &expr, MeshEntityId const &s,
          index_sequence<VERTEX, EDGE>)
    {
        int n = M::sub_index(s);
        MeshEntityId X = M::delta_index(s);
        return (eval(m, expr, s - X)[n] + eval(m, expr, s + X)[n]) * 0.5;
    }


    template<typename TF>
    static inline traits::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<VERTEX, FACE>)
    {

        auto const &l = expr;
        auto X = M::delta_index(M::dual(s));
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);

        return (
                       eval(m, l, (s - Y - Z)) +
                       eval(m, l, (s - Y + Z)) +
                       eval(m, l, (s + Y - Z)) +
                       eval(m, l, (s + Y + Z))
               ) * 0.25;
    }


    template<typename TV, typename ...Others>
    static inline
    TV mapto(mesh_type const &m, Field<nTuple<TV, 3>, Others...> const &expr, MeshEntityId const &s,
             index_sequence<VERTEX, FACE>)
    {

        int n = M::sub_index(s);
        auto const &l = expr;
        auto X = M::delta_index(M::dual(s));
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);

        return (
                       eval(m, l, (s - Y - Z))[n] +
                       eval(m, l, (s - Y + Z))[n] +
                       eval(m, l, (s + Y - Z))[n] +
                       eval(m, l, (s + Y + Z))[n]
               ) * 0.25;
    }

    template<typename TF>
    static inline
    traits::value_type_t<TF>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<VERTEX, VOLUME>)
    {
        auto const &l = expr;

        auto X = M::DI(0, s);
        auto Y = M::DI(1, s);
        auto Z = M::DI(2, s);

        return (eval(m, l, s - X - Y - Z) +
                eval(m, l, s - X - Y + Z) +
                eval(m, l, s - X + Y - Z) +
                eval(m, l, s - X + Y + Z) +
                eval(m, l, s + X - Y - Z) +
                eval(m, l, s + X - Y + Z) +
                eval(m, l, s + X + Y - Z) +
                eval(m, l, s + X + Y + Z)
               ) * 0.125;
    }


    template<typename TF>
    static inline
    nTuple<typename traits::value_type<TF>::type, 3>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<EDGE, VERTEX>)
    {
        typedef nTuple<typename traits::value_type<TF>::type, 3> field_value_type;

        auto const &l = expr;

        MeshEntityId DA = M::_DA;
        MeshEntityId X = M::_DI;
        MeshEntityId Y = M::_DJ;
        MeshEntityId Z = M::_DK;

        return nTuple<typename traits::value_type<TF>::type, 3>
                {
                        static_cast<typename traits::value_type<TF>::type>((eval(m, l, s - X) +
                                                                            eval(m, l, s + X)) * 0.5),
                        static_cast<typename traits::value_type<TF>::type>((eval(m, l, s - Y) +
                                                                            eval(m, l, s + Y)) * 0.5),
                        static_cast<typename traits::value_type<TF>::type>((eval(m, l, s - Z) +
                                                                            eval(m, l, s + Z)) * 0.5)

                };


    }


    template<typename TF>
    static inline
    nTuple<typename traits::value_type<TF>::type, 3>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<FACE, VERTEX>)
    {
        auto const &l = expr;

        MeshEntityId X = M::_DI;
        MeshEntityId Y = M::_DJ;
        MeshEntityId Z = M::_DK;


        return nTuple<typename traits::value_type<TF>::type, 3>
                {
                        static_cast<typename traits::value_type<TF>::type>(eval(m, l, (s - Y - Z)) +
                                                                           eval(m, l, (s - Y + Z)) +
                                                                           eval(m, l, (s + Y - Z)) +
                                                                           eval(m, l, (s + Y + Z)) * 0.25),
                        static_cast<typename traits::value_type<TF>::type>(eval(m, l, (s - Z - X)) +
                                                                           eval(m, l, (s - Z + X)) +
                                                                           eval(m, l, (s + Z - X)) +
                                                                           eval(m, l, (s + Z + X)) * 0.25),
                        static_cast<typename traits::value_type<TF>::type>(eval(m, l, (s - X - Y)) +
                                                                           eval(m, l, (s - X + Y)) +
                                                                           eval(m, l, (s + X - Y)) +
                                                                           eval(m, l, (s + X + Y)) * 0.25)
                };


    }


    template<typename TF>
    static inline
    typename traits::value_type<TF>::type
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<VOLUME, VERTEX>)
    {
        auto const &l = expr;

        auto X = m.DI(0, s);
        auto Y = m.DI(1, s);
        auto Z = m.DI(2, s);

        return (
                       eval(m, l, ((s - X - Y - Z))) +
                       eval(m, l, ((s - X - Y + Z))) +
                       eval(m, l, ((s - X + Y - Z))) +
                       eval(m, l, ((s - X + Y + Z))) +
                       eval(m, l, ((s + X - Y - Z))) +
                       eval(m, l, ((s + X - Y + Z))) +
                       eval(m, l, ((s + X + Y - Z))) +
                       eval(m, l, ((s + X + Y + Z)))
               ) * 0.125;
    }


    template<typename TF>
    static inline
    typename traits::value_type<TF>::type
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<VOLUME, FACE>)
    {
        auto X = M::delta_index(M::dual(s));

        return (eval(m, expr, s - X) + eval(m, expr, s + X)) * 0.5;
    }


    template<typename TF>
    static inline
    typename traits::value_type<TF>::type
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<VOLUME, EDGE>)
    {
        auto const &l = expr;
        auto X = M::delta_index(s);
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);

        return (
                       eval(m, l, s - Y - Z) +
                       eval(m, l, s - Y + Z) +
                       eval(m, l, s + Y - Z) +
                       eval(m, l, s + Y + Z)
               ) * 0.25;
    }


    template<typename TF>
    static inline
    nTuple<typename traits::value_type<TF>::type, 3>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<FACE, VOLUME>)
    {
        auto const &l = expr;

        auto X = m.DI(0, M::dual(s));
        auto Y = m.DI(1, M::dual(s));
        auto Z = m.DI(2, M::dual(s));

        return nTuple<traits::value_type_t<TF>, 3>
                {
                        (eval(m, l, s - X) + eval(m, l, s + X)) * 0.5,
                        (eval(m, l, s - Y) + eval(m, l, s + Y)) * 0.5,
                        (eval(m, l, s - Z) + eval(m, l, s + Z)) * 0.5
                };


    }


    template<typename TF>
    static inline
    nTuple<typename traits::value_type<TF>::type, 3>
    mapto(mesh_type const &m, TF const &expr, MeshEntityId const &s, index_sequence<EDGE, VOLUME>)
    {
        auto const &l = expr;

        auto X = m.DI(0, s);
        auto Y = m.DI(1, s);
        auto Z = m.DI(2, s);

        return nTuple<typename traits::value_type<TF>::type, 3>
                {
                        (eval(m, l, s - Y - Z) + eval(m, l, s - Y + Z) +
                         eval(m, l, s + Y - Z) + eval(m, l, s + Y + Z)) * 0.25,
                        (eval(m, l, s - Z - X) + eval(m, l, s - Z + X) +
                         eval(m, l, s + Z - X) + eval(m, l, s + Z + X)) * 0.25,
                        (eval(m, l, s - X - Y) + eval(m, l, s - X + Y) +
                         eval(m, l, s + X - Y) + eval(m, l, s + X + Y)) * 0.25
                };


    }


    //***************************************************************************************************
    //
    //! Form<IL> ^ Form<IR> => Form<IR+IL>
    template<typename ...T, size_t IL, size_t IR>
    static inline
    traits::value_type_t<Field<Expression<ct::Wedge, T...>>>
    eval(mesh_type const &m, Field<Expression<ct::Wedge, T...>> const &expr, MeshEntityId const &s,
         index_sequence<IL, IR>)
    {
        return m.inner_product(mapto(m, std::get<0>(expr.args), s, index_sequence<IL, IR + IL>()),
                               mapto(m, std::get<1>(expr.args), s, index_sequence<IR, IR + IL>()),
                               s);

    }


    template<typename TL, typename TR>
    static inline
    traits::value_type_t<Field<Expression<ct::Wedge, TL, TR>>>
    eval(mesh_type const &m, Field<Expression<ct::Wedge, TL, TR>> const &expr, MeshEntityId const &s,
         index_sequence<EDGE, EDGE>)
    {
        auto const &l = std::get<0>(expr.args);
        auto const &r = std::get<1>(expr.args);

        auto Y = M::delta_index(M::rotate(M::dual(s)));
        auto Z = M::delta_index(
                M::inverse_rotate(M::dual(s)));

        return ((eval(m, l, s - Y) + eval(m, l, s + Y))
                * (eval(m, l, s - Z) + eval(m, l, s + Z)) * 0.25);
    }

    static MeshEntityId sw(MeshEntityId s, u_int16_t w)
    {
        s.w = w;
        return s;
    }

    template<typename TL, typename TR, size_t I>
    static inline
    traits::value_type_t<Field<Expression<ct::Cross, TL, TR>>>
    eval(mesh_type const &m, Field<Expression<ct::Cross, TL, TR>> const &expr, MeshEntityId const &s,
         index_sequence<I, I>)
    {
        return eval(m, std::get<0>(expr.args), sw(s, (s.w + 1) % 3)) *
               eval(m, std::get<1>(expr.args), sw(s, (s.w + 2) % 3))
               -
               eval(m, std::get<0>(expr.args), sw(s, (s.w + 2) % 3)) *
               eval(m, std::get<1>(expr.args), sw(s, (s.w + 1) % 3));
    }

    template<typename TL, typename TR, size_t I>
    static inline
    traits::value_type_t<Field<Expression<ct::Dot, TL, TR>>>
    eval(mesh_type const &m, Field<Expression<ct::Dot, TL, TR>> const &expr, MeshEntityId const &s,
         index_sequence<I, I>)
    {
        return eval(m, std::get<0>(expr.args), sw(s, 0)) * eval(m, std::get<1>(expr.args), sw(s, 0)) +
               eval(m, std::get<0>(expr.args), sw(s, 1)) * eval(m, std::get<1>(expr.args), sw(s, 1)) +
               eval(m, std::get<0>(expr.args), sw(s, 2)) * eval(m, std::get<1>(expr.args), sw(s, 2));
    }


    template<typename TL, typename TR, size_t I>
    static inline
    traits::value_type_t<Field<Expression<_impl::divides, TL, TR>>>
    eval(mesh_type const &m, Field<Expression<_impl::divides, TL, TR>> const &expr, MeshEntityId const &s,
         index_sequence<I, VERTEX>)
    {
        return eval(m, std::get<0>(expr.args), s) /
               mapto(m, std::get<1>(expr.args), s, index_sequence<VERTEX, I>());
    }

    template<typename TL, typename TR, size_t I>
    static inline traits::value_type_t<Field<Expression<_impl::multiplies, TL, TR>>>
    eval(mesh_type const &m, Field<Expression<_impl::multiplies, TL, TR>> const &expr, MeshEntityId const &s,
         index_sequence<I, VERTEX>)
    {
        return eval(m, std::get<0>(expr.args), s) *
               mapto(m, std::get<1>(expr.args), s, index_sequence<VERTEX, I>());
    }


    template<typename ...T, size_t ...I>
    static inline
    traits::value_type_t<Field<Expression<ct::MapTo, T...> >>
    eval(mesh_type const &m, Field<Expression<ct::MapTo, T...>> const &expr, MeshEntityId const &s,
         index_sequence<I...>)
    {
        return mapto(m, std::get<0>(expr.args), s, index_sequence<I...>());
    };


    ///***************************************************************************************************
    /// @name general_algebra General algebra
    /// @{
    ///***************************************************************************************************


    template<typename T>
    inline static T const &
    eval(mesh_type const &m, T const &v, MeshEntityId const &s, st::is_primary_t<T> *_p = nullptr) { return v; }


    template<typename T>
    inline static traits::primary_type_t<T>
    eval(mesh_type const &m, T const &v, MeshEntityId const &s, st::is_expression_ntuple_t<T> *_p = nullptr)
    {
        traits::primary_type_t<T> res;
        res = v;
        return std::move(res);
    }

    template<typename TOP, typename ... T>
    inline static typename traits::value_type<Field<Expression<TOP, T...> > >::type
    eval(mesh_type const &m, Field<Expression<TOP, T...> > const &expr, MeshEntityId const &s)
    {
        return eval(m, expr, s, traits::iform_list_t<T...>());
    }

    //******************************************************************************************************************
    // for element-wise arithmetic operation
    template<typename TOP, typename ...T, size_t ... I> inline static
    typename traits::value_type<Field<Expression<TOP, T...> > >::type
    _invoke_helper(mesh_type const &m, Field<Expression<TOP, T...>> const &expr, MeshEntityId const &s,
                   index_sequence<I...>)
    {
        return expr.m_op_(mapto(m, std::get<I>(expr.args), s,
                                index_sequence<traits::iform<T>::value,
                                        traits::iform<Field<Expression<TOP, T...> > >::value>())...);
    };


    template<typename TOP, typename ... T, size_t ...I>
    inline static traits::value_type_t<Field<Expression<TOP, T...> >>
    eval(mesh_type const &m, Field<Expression<TOP, T...> > const &expr, MeshEntityId const &s,
         index_sequence<I...>)
    {
        return _invoke_helper(m, expr, s, index_sequence_for<T...>());
    }
    //******************************************************************************************************************


};// struct DiffScheme<TGeo, diff_scheme::tags::finite_volume>

template<typename TM> constexpr Real CalculusPolicy<TM>::m_p_curl_factor_[3];
//template<typename TM> static  Real  FiniteVolume<TM, std::enable_if_t<std::is_base_of<mesh_as::MeshEntityIdCoder, TM>::entity>>::m_p_curl_factor2_[3];
}}}// namespace simpla

#endif /* FDM_H_ */
