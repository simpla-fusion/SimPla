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


template<typename TM, class Enable = void>
struct FiniteVolume {};

namespace ct=calculus::tags;
namespace st=simpla::traits;


/**
 * @ingroup diff_scheme
 *
 * finite volume
 */
template<typename TM>
struct FiniteVolume<TM>
{
    typedef FiniteVolume<TM> this_type;
    typedef TM mesh_type;
    mesh_type const &m;
    typedef mesh::MeshEntityIdCoder M;
    typedef mesh::MeshEntityId MeshEntitId;
public:
    typedef this_type calculus_policy;

    FiniteVolume(TM const &m_) : m(m_) {}

    virtual ~FiniteVolume() {}

    static std::string class_name() { return "FiniteVolume"; }

    virtual std::ostream &print(std::ostream &os, int indent = 1) const
    {
        os << std::setw(indent) << " " << "[FiniteVolume]," << std::endl;
        return os;
    }

    void deploy() {}

public:

//    template<typename ...T>
//    inline constexpr traits::value_type_t<Field<T...> >
//    eval(Field<T...> &expr, MeshEntitId const& s) const { return eval(expr, s); }
//
//    template<typename ...T>
//    inline constexpr traits::value_type_t<Field<T...> >
//    eval(Field<T...> const &expr, MeshEntitId const& s) const { return eval(expr, s); }
//
//    template<typename T>
//    inline constexpr T &eval(T &f, MeshEntitId const& s) const { return f; }
//
//
//    template<typename T>
//    inline constexpr T const &eval(T const &f, MeshEntitId const& s) const { return f; }



//    template<typename TV, typename OM, size_t IFORM> inline constexpr
//    typename traits::value_type<Field<TV, OM, index_const<IFORM>>>::type &
//    eval(Field<TV, OM, index_const<IFORM>> &f, MeshEntitId const& s) const { return f[s]; };

    template<typename TV, typename OM, size_t IFORM>
    inline constexpr TV
    eval(Field <TV, OM, index_const<IFORM>> const &f, MeshEntitId const &s) const { return f[s]; };


private:

    template<typename FExpr>
    inline constexpr traits::value_type_t<FExpr>
    get_v(FExpr const &f, MeshEntitId const s) const
    {
        return eval(f, s) * m.volume(s);
    }

    template<typename FExpr>
    inline constexpr traits::value_type_t<FExpr>
    get_d(FExpr const &f, MeshEntitId const s) const
    {
        return eval(f, s) * m.dual_volume(s);
    }

public:

    //***************************************************************************************************
    // Exterior algebra
    //***************************************************************************************************

    //! grad<0>
    template<typename T>
    inline traits::value_type_t<Field < Expression < ct::ExteriorDerivative, T>>>
    eval(Field<Expression < ct::ExteriorDerivative, T>
    > const &f,
    MeshEntitId const &s,
            index_sequence<mesh::VERTEX>
    ) const
    {
        MeshEntitId D = M::delta_index(s);
        return (get_v(std::get<0>(f.args), s + D) - get_v(std::get<0>(f.args), s - D)) * m.inv_volume(s);
    }


    //! curl<1>
    template<typename T>
    inline traits::value_type_t<Field < Expression < ct::ExteriorDerivative, T>>>
    eval(Field<Expression < ct::ExteriorDerivative, T>
    > const &expr,
    MeshEntitId const &s, index_sequence<EDGE>
    ) const
    {

        MeshEntitId X = M::delta_index(M::dual(s));
        MeshEntitId Y = M::rotate(X);
        MeshEntitId Z = M::inverse_rotate(X);


        return ((get_v(std::get<0>(expr.args), s + Y) - get_v(std::get<0>(expr.args), s - Y))
                - (get_v(std::get<0>(expr.args), s + Z) - get_v(std::get<0>(expr.args), s - Z))
               ) * m.inv_volume(s);


    }


    //! div<2>
    template<typename T>
    constexpr inline traits::value_type_t<Field < Expression < ct::ExteriorDerivative, T>>>
    eval(Field<Expression < ct::ExteriorDerivative, T>
    > const &expr,
    MeshEntitId const &s, index_sequence<FACE>
    ) const
    {
        return (get_v(std::get<0>(expr.args), s + M::_DI) - get_v(std::get<0>(expr.args), s - M::_DI)
                + get_v(std::get<0>(expr.args), s + M::_DJ) - get_v(std::get<0>(expr.args), s - M::_DJ)
                + get_v(std::get<0>(expr.args), s + M::_DK) - get_v(std::get<0>(expr.args), s - M::_DK)
               ) * m.inv_volume(s);

    }


    //! div<1>
    template<typename T>
    constexpr inline traits::value_type_t<Field < Expression < ct::CodifferentialDerivative, T>>>
    eval(Field<Expression < ct::CodifferentialDerivative, T>>
    const &expr,
    MeshEntitId const &s,
            index_sequence<EDGE>
    ) const
    {
        return -(get_d(std::get<0>(expr.args), s + M::_DI) - get_d(std::get<0>(expr.args), s - M::_DI)
                 + get_d(std::get<0>(expr.args), s + M::_DJ) - get_d(std::get<0>(expr.args), s - M::_DJ)
                 + get_d(std::get<0>(expr.args), s + M::_DK) - get_d(std::get<0>(expr.args), s - M::_DK)
        ) * m.inv_dual_volume(s);
    }

    //! curl<2>
    template<typename T> inline traits::value_type_t<Field < Expression < ct::CodifferentialDerivative, T>>>
    eval(Field<Expression < ct::CodifferentialDerivative, T>>
    const &expr,
    MeshEntitId const &s,
            index_sequence<FACE>
    ) const
    {

        MeshEntitId X = M::delta_index(s);
        MeshEntitId Y = M::rotate(X);
        MeshEntitId Z = M::inverse_rotate(X);

        return -((get_d(std::get<0>(expr.args), s + Y) - get_d(std::get<0>(expr.args), s - Y))
                 - (get_d(std::get<0>(expr.args), s + Z) - get_d(std::get<0>(expr.args), s - Z))
        ) * m.inv_dual_volume(s);
    }


    //! grad<3>
    template<typename T>
    inline traits::value_type_t<Field < Expression < ct::CodifferentialDerivative, T>>>
    eval(Field<Expression < ct::CodifferentialDerivative, T>
    > const &expr,
    MeshEntitId const &s,
            index_sequence<VOLUME>
    ) const
    {
        MeshEntitId D = M::delta_index(M::dual(s));
        return -(get_d(std::get<0>(expr.args), s + D) - get_d(std::get<0>(expr.args), s - D)) *
               m.inv_dual_volume(s);
    }

    //! *Form<IR> => Form<N-IL>


    template<typename T, size_t I>
    inline traits::value_type_t<Field < Expression < ct::HodgeStar, T> >>
    eval(Field<Expression < ct::HodgeStar, T>>
    const &expr,
    MeshEntitId const &s, index_sequence<I>
    ) const
    {
        auto const &l = std::get<0>(expr.args);
        int i = M::iform(s);
        MeshEntitId X = (i == VERTEX || i == VOLUME) ? M::_DI : M::delta_index(M::dual(s));
        MeshEntitId Y = M::rotate(X);
        MeshEntitId Z = M::inverse_rotate(X);


        return (
                       get_v(l, ((s - X) - Y) - Z) +
                       get_v(l, ((s - X) - Y) + Z) +
                       get_v(l, ((s - X) + Y) - Z) +
                       get_v(l, ((s - X) + Y) + Z) +
                       get_v(l, ((s + X) - Y) - Z) +
                       get_v(l, ((s + X) - Y) + Z) +
                       get_v(l, ((s + X) + Y) - Z) +
                       get_v(l, ((s + X) + Y) + Z)

               ) * m.inv_dual_volume(s) * 0.125;


    };

    ////***************************************************************************************************
    //! p_curl<1>
    static constexpr Real m_p_curl_factor_[3] = {0, 1, -1};

    template<typename T, size_t I>
    inline traits::value_type_t<Field < Expression < ct::P_ExteriorDerivative<I>, T>>>
    eval(Field<Expression < ct::P_ExteriorDerivative<I>, T>>
    const &expr,
    MeshEntitId const &s,
            index_sequence<EDGE>
    ) const
    {
        return (get_v(std::get<0>(expr.args), s + M::DI(I)) -
                get_v(std::get<0>(expr.args), s - M::DI(I))
               ) * m.inv_volume(s) * m_p_curl_factor_[(I + 3 - M::sub_index(s)) % 3];
    }


    template<typename T, size_t I>
    inline traits::value_type_t<Field < Expression < ct::P_CodifferentialDerivative<I>, T>>>
    eval(Field<Expression < ct::P_CodifferentialDerivative<I>, T>>
    const &expr,
    MeshEntitId const &s,
            index_sequence<FACE>
    ) const
    {

        return (get_v(std::get<0>(expr.args), s + M::DI(I)) -
                get_v(std::get<0>(expr.args), s - M::DI(I))
               ) * m.inv_dual_volume(s) * m_p_curl_factor_[(I + 3 - M::sub_index(s)) % 3];
    }

////***************************************************************************************************
//
////! map_to
    template<typename T, size_t I>
    inline constexpr T
    mapto(T const &r, MeshEntitId const &s, index_sequence <VERTEX, I>,
          st::is_primary_t<T> *_p = nullptr) const { return r; }

    template<typename TF, size_t I>
    inline constexpr traits::value_type_t<TF>
    mapto(TF const &expr, MeshEntitId const &s, index_sequence <I, I>, std::enable_if_t<!st::is_primary<TF>::value>
    *_p = nullptr) const { return eval(expr, s); }

    template<typename TV, typename OM, size_t I>
    inline constexpr TV
    mapto(Field <TV, OM, index_const<I>> const &f, MeshEntitId const &s, index_sequence <I, I>) const { return f[s]; };

    template<typename TF>
    inline traits::value_type_t<TF>
    mapto(TF const &expr, MeshEntitId const &s, index_sequence <VERTEX, EDGE>) const
    {
        MeshEntitId X = M::delta_index(s);
        return (eval(expr, s - X) + eval(expr, s + X)) * 0.5;
    }


    template<typename TV, typename ...Others>
    inline TV
    mapto(Field<nTuple < TV, 3>, Others...

    > const &expr,
    MeshEntitId const &s, index_sequence<VERTEX, EDGE>
    ) const
    {
        int n = M::sub_index(s);
        MeshEntitId X = M::delta_index(s);
        return (eval(expr, s - X)[n] + eval(expr, s + X)[n]) * 0.5;
    }


    template<typename TF>
    inline traits::value_type_t<TF>
    mapto(TF const &expr, MeshEntitId const &s, index_sequence <VERTEX, FACE>) const
    {

        auto const &l = expr;
        auto X = M::delta_index(M::dual(s));
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);

        return (
                       eval(l, (s - Y - Z)) +
                       eval(l, (s - Y + Z)) +
                       eval(l, (s + Y - Z)) +
                       eval(l, (s + Y + Z))
               ) * 0.25;
    }


    template<typename TV, typename ...Others>
    constexpr inline
    TV
    mapto(Field<nTuple < TV, 3>, Others...

    > const &expr,
    MeshEntitId const &s, index_sequence<VERTEX, FACE>
    ) const
    {

        int n = M::sub_index(s);
        auto const &l = expr;
        auto X = M::delta_index(M::dual(s));
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);

        return (
                       eval(l, (s - Y - Z))[n] +
                       eval(l, (s - Y + Z))[n] +
                       eval(l, (s + Y - Z))[n] +
                       eval(l, (s + Y + Z))[n]
               ) * 0.25;
    }

    template<typename TF>
    constexpr inline
    traits::value_type_t<TF>
    mapto(TF const &expr, MeshEntitId const &s, index_sequence <VERTEX, VOLUME>) const
    {
        auto const &l = expr;

        auto X = m.DI(0, s);
        auto Y = m.DI(1, s);
        auto Z = m.DI(2, s);

        return (eval(l, s - X - Y - Z) +
                eval(l, s - X - Y + Z) +
                eval(l, s - X + Y - Z) +
                eval(l, s - X + Y + Z) +
                eval(l, s + X - Y - Z) +
                eval(l, s + X - Y + Z) +
                eval(l, s + X + Y - Z) +
                eval(l, s + X + Y + Z)
               ) * 0.125;
    }


    template<typename TF>
    const inline
    nTuple<typename traits::value_type<TF>::type, 3>
    mapto(TF const &expr, MeshEntitId const &s, index_sequence <EDGE, VERTEX>) const
    {
        typedef nTuple<typename traits::value_type<TF>::type, 3> field_value_type;

        auto const &l = expr;

        MeshEntitId DA = M::_DA;
        MeshEntitId X = M::_DI;
        MeshEntitId Y = M::_DJ;
        MeshEntitId Z = M::_DK;

        return nTuple<typename traits::value_type<TF>::type, 3>
                {
                        static_cast<typename traits::value_type<TF>::type>((eval(l, s - X) +
                                                                            eval(l, s + X)) * 0.5),
                        static_cast<typename traits::value_type<TF>::type>((eval(l, s - Y) +
                                                                            eval(l, s + Y)) * 0.5),
                        static_cast<typename traits::value_type<TF>::type>((eval(l, s - Z) +
                                                                            eval(l, s + Z)) * 0.5)

                };


    }


    template<typename TF>
    constexpr inline
    nTuple<typename traits::value_type<TF>::type, 3>
    mapto(TF const &expr, MeshEntitId const &s, index_sequence <FACE, VERTEX>) const
    {
        auto const &l = expr;

        MeshEntitId X = M::_DI;
        MeshEntitId Y = M::_DJ;
        MeshEntitId Z = M::_DK;


        return nTuple<typename traits::value_type<TF>::type, 3>
                {
                        static_cast<typename traits::value_type<TF>::type>(eval(l, (s - Y - Z)) +
                                                                           eval(l, (s - Y + Z)) +
                                                                           eval(l, (s + Y - Z)) +
                                                                           eval(l, (s + Y + Z)) * 0.25),
                        static_cast<typename traits::value_type<TF>::type>(eval(l, (s - Z - X)) +
                                                                           eval(l, (s - Z + X)) +
                                                                           eval(l, (s + Z - X)) +
                                                                           eval(l, (s + Z + X)) * 0.25),
                        static_cast<typename traits::value_type<TF>::type>(eval(l, (s - X - Y)) +
                                                                           eval(l, (s - X + Y)) +
                                                                           eval(l, (s + X - Y)) +
                                                                           eval(l, (s + X + Y)) * 0.25)
                };


    }


    template<typename TF>
    constexpr inline
    typename traits::value_type<TF>::type
    mapto(TF const &expr, MeshEntitId const &s, index_sequence <VOLUME, VERTEX>) const
    {
        auto const &l = expr;

        auto X = m.DI(0, s);
        auto Y = m.DI(1, s);
        auto Z = m.DI(2, s);

        return (
                       eval(l, ((s - X - Y - Z))) +
                       eval(l, ((s - X - Y + Z))) +
                       eval(l, ((s - X + Y - Z))) +
                       eval(l, ((s - X + Y + Z))) +
                       eval(l, ((s + X - Y - Z))) +
                       eval(l, ((s + X - Y + Z))) +
                       eval(l, ((s + X + Y - Z))) +
                       eval(l, ((s + X + Y + Z)))
               ) * 0.125;
    }


    template<typename TF>
    constexpr inline
    typename traits::value_type<TF>::type
    mapto(TF const &expr, MeshEntitId const &s, index_sequence <VOLUME, FACE>) const
    {
        auto X = M::delta_index(M::dual(s));

        return (eval(expr, s - X) + eval(expr, s + X)) * 0.5;
    }


    template<typename TF>
    constexpr inline
    typename traits::value_type<TF>::type
    mapto(TF const &expr, MeshEntitId const &s, index_sequence <VOLUME, EDGE>) const
    {
        auto const &l = expr;
        auto X = M::delta_index(s);
        auto Y = M::rotate(X);
        auto Z = M::inverse_rotate(X);

        return (
                       eval(l, s - Y - Z) +
                       eval(l, s - Y + Z) +
                       eval(l, s + Y - Z) +
                       eval(l, s + Y + Z)
               ) * 0.25;
    }


    template<typename TF>
    constexpr inline
    nTuple<typename traits::value_type<TF>::type, 3>
    mapto(TF const &expr, MeshEntitId const &s, index_sequence <FACE, VOLUME>) const
    {
        auto const &l = expr;

        auto X = m.DI(0, M::dual(s));
        auto Y = m.DI(1, M::dual(s));
        auto Z = m.DI(2, M::dual(s));

        return nTuple<traits::value_type_t<TF>, 3>
                {
                        (eval(l, s - X) + eval(l, s + X)) * 0.5,
                        (eval(l, s - Y) + eval(l, s + Y)) * 0.5,
                        (eval(l, s - Z) + eval(l, s + Z)) * 0.5
                };


    }


    template<typename TF>
    constexpr inline
    nTuple<typename traits::value_type<TF>::type, 3>
    mapto(TF const &expr, MeshEntitId const &s, index_sequence <EDGE, VOLUME>) const
    {
        auto const &l = expr;

        auto X = m.DI(0, s);
        auto Y = m.DI(1, s);
        auto Z = m.DI(2, s);

        return nTuple<typename traits::value_type<TF>::type, 3>
                {
                        (eval(l, s - Y - Z) + eval(l, s - Y + Z) +
                         eval(l, s + Y - Z) + eval(l, s + Y + Z)) * 0.25,
                        (eval(l, s - Z - X) + eval(l, s - Z + X) +
                         eval(l, s + Z - X) + eval(l, s + Z + X)) * 0.25,
                        (eval(l, s - X - Y) + eval(l, s - X + Y) +
                         eval(l, s + X - Y) + eval(l, s + X + Y)) * 0.25
                };


    }


    //***************************************************************************************************
    //
    //! Form<IL> ^ Form<IR> => Form<IR+IL>
    template<typename ...T, size_t IL, size_t IR>
    constexpr inline
    traits::value_type_t<Field < Expression < ct::Wedge, T...>>>
    eval(Field<Expression < ct::Wedge, T...>>
    const &expr,
    MeshEntitId const &s, index_sequence<IL, IR>
    ) const
    {
        return m.inner_product(mapto(std::get<0>(expr.args), s, index_sequence < IL, IR + IL > ()),
                               mapto(std::get<1>(expr.args), s, index_sequence < IR, IR + IL > ()),
                               s);

    }


    template<typename TL, typename TR>
    constexpr inline
    traits::value_type_t<Field < Expression < ct::Wedge, TL, TR>>>
    eval(Field<Expression < ct::Wedge, TL, TR>>
    const &expr,
    MeshEntitId const &s, index_sequence<EDGE, EDGE>
    ) const
    {
        auto const &l = std::get<0>(expr.args);
        auto const &r = std::get<1>(expr.args);

        auto Y = M::delta_index(M::rotate(M::dual(s)));
        auto Z = M::delta_index(
                M::inverse_rotate(M::dual(s)));

        return ((eval(l, s - Y) + eval(l, s + Y))
                * (eval(l, s - Z) + eval(l, s + Z)) * 0.25);
    }


    template<typename TL, typename TR, size_t I>
    constexpr inline
    traits::value_type_t<Field < Expression < ct::Cross, TL, TR>>>
    eval(Field<Expression < ct::Cross, TL, TR>>
    const &expr,
    MeshEntitId const &s, index_sequence<I, I>
    ) const
    {
        return cross(eval(std::get<0>(expr.args), s), eval(std::get<1>(expr.args), s));
    }

    template<typename TL, typename TR, size_t I>
    constexpr inline
    traits::value_type_t<Field < Expression < ct::Dot, TL, TR>>>
    eval(Field<Expression < ct::Dot, TL, TR>>
    const &expr,
    MeshEntitId const &s, index_sequence<I, I>
    ) const
    {
        return dot(eval(std::get<0>(expr.args), s), eval(std::get<1>(expr.args), s));
    }


    template<typename TL, typename TR, size_t I>
    constexpr inline
    traits::value_type_t<Field < Expression < _impl::divides, TL, TR>>>
    eval(Field<Expression < _impl::divides, TL, TR>>
    const &expr,
    MeshEntitId const &s, index_sequence<I, VERTEX>
    ) const
    {
        return eval(std::get<0>(expr.args), s) / mapto(std::get<1>(expr.args), s, index_sequence < VERTEX, I > ());
    }

    template<typename TL, typename TR, size_t I>
    constexpr inline traits::value_type_t<Field < Expression < _impl::multiplies, TL, TR>>>
    eval(Field<Expression < _impl::multiplies, TL, TR>>
    const &expr,
    MeshEntitId const &s,
            index_sequence<I, VERTEX>
    ) const
    {
        return eval(std::get<0>(expr.args), s) * mapto(std::get<1>(expr.args), s, index_sequence < VERTEX, I > ());
    }


    template<typename ...T, size_t ...I>
    constexpr inline
    traits::value_type_t<Field < Expression < ct::MapTo, T...> >>
    eval(Field<Expression < ct::MapTo, T...>>
    const &expr,
    MeshEntitId const &s, index_sequence<I...>
    ) const
    {
        return mapto(std::get<0>(expr.args), s, index_sequence < I... > ());
    };


    ///***************************************************************************************************
    /// @name general_algebra General algebra
    /// @{
    ///***************************************************************************************************


    template<typename T>
    inline constexpr T const &
    eval(T const &v, MeshEntitId const &s, st::is_primary_t<T> *_p = nullptr) const { return v; }


    template<typename T>
    inline constexpr traits::primary_type_t<T>
    eval(T const &v, MeshEntitId const &s, st::is_expression_ntuple_t<T> *_p = nullptr) const
    {
        traits::primary_type_t<T> res;
        res = v;
        return std::move(res);
    }

    template<typename TOP, typename ... T>
    inline constexpr traits::value_type_t<Field < Expression < TOP, T...> > >
    eval(Field<Expression < TOP, T...>
    > const &expr,
    MeshEntitId const &s
    ) const
    {
        return eval(expr, s, traits::iform_list_t<T...>());
    }

    //******************************************************************************************************************
    // for element-wise arithmetic operation
    template<typename TOP, typename ...T, size_t ... I> inline constexpr
    typename traits::value_type<Field < Expression < TOP, T...> > >

    ::type
    _invoke_helper(Field <Expression<TOP, T...>> const &expr, MeshEntitId const &s, index_sequence<I...>) const
    {
        return expr.m_op_(mapto(std::get<I>(expr.args), s,
                                index_sequence < traits::iform<T>::value,
                                traits::iform<Field < Expression<TOP, T...> > > ::value > ())...);
    };


    template<typename TOP, typename ... T, size_t ...I>
    inline constexpr traits::value_type_t<Field < Expression < TOP, T...> >>
    eval(Field<Expression < TOP, T...>
    > const &expr,
    MeshEntitId const &s, index_sequence<I...>
    ) const
    {
        return _invoke_helper(expr, s, index_sequence_for<T...>());
    }
    //******************************************************************************************************************


};// struct DiffScheme<TGeo, diff_scheme::tags::finite_volume>

template<typename TM> constexpr Real  FiniteVolume<TM>::m_p_curl_factor_[3];
//template<typename TM> constexpr Real  FiniteVolume<TM, std::enable_if_t<std::is_base_of<mesh::MeshEntityIdCoder, TM>::entity>>::m_p_curl_factor2_[3];
}}}// namespace simpla

#endif /* FDM_H_ */
