/**
 * @file linear.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_LINEAR_H
#define SIMPLA_LINEAR_H

#include "../../gtl/nTuple.h"

namespace simpla { namespace manifold { namespace policy
{


#define DECLARE_FUNCTION_PREFIX inline static
#define DECLARE_FUNCTION_SUFFIX /*const*/


/**
 * @ingroup interpolate
 * @brief basic linear interpolate
 */
struct LinearInterpolator
{
private:


    typedef LinearInterpolator this_type;

public:

    typedef LinearInterpolator interpolate_policy;

private:

    template<typename M, typename TD, typename TIDX> DECLARE_FUNCTION_PREFIX auto
    gather_impl_(M const &m, TD const &f, TIDX const &idx) DECLARE_FUNCTION_SUFFIX -> decltype(
    traits::index(f, std::get<0>(idx)) *
    std::get<1>(idx)[0])
    {

        auto X = (M::_DI) << 1;
        auto Y = (M::_DJ) << 1;
        auto Z = (M::_DK) << 1;

        typename M::point_type r = std::get<1>(idx);
        typename M::index_type s = std::get<0>(idx);

        return traits::index(f, ((s + X) + Y) + Z) * (r[0]) * (r[1]) * (r[2]) //
               + traits::index(f, (s + X) + Y) * (r[0]) * (r[1]) * (1.0 - r[2]) //
               + traits::index(f, (s + X) + Z) * (r[0]) * (1.0 - r[1]) * (r[2]) //
               + traits::index(f, (s + X)) * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]) //
               + traits::index(f, (s + Y) + Z) * (1.0 - r[0]) * (r[1]) * (r[2]) //
               + traits::index(f, (s + Y)) * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]) //
               + traits::index(f, s + Z) * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]) //
               + traits::index(f, s) * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
    }

public:

    template<typename M, typename TF, typename TX> DECLARE_FUNCTION_PREFIX auto
    gather(M const &m, TF const &f, TX const &r) DECLARE_FUNCTION_SUFFIX//
    ENABLE_IF_DECL_RET_TYPE((traits::iform<TF>::value
                             == VERTEX), (gather_impl_(m, f, m.coordinates_global_to_local(r, 0))))

    template<typename M, typename TF> DECLARE_FUNCTION_PREFIX auto
    gather(M const &m, TF const &f, typename M::point_type const &r) DECLARE_FUNCTION_SUFFIX
    ENABLE_IF_DECL_RET_TYPE((traits::iform<TF>::value
                             == EDGE),
                            traits::make_nTuple(
                                    gather_impl_(m, f, m.coordinates_global_to_local(r, 1)),
                                    gather_impl_(m, f, m.coordinates_global_to_local(r, 2)),
                                    gather_impl_(m, f, m.coordinates_global_to_local(r, 4))
                            ))

    template<typename M, typename TF> DECLARE_FUNCTION_PREFIX auto
    gather(M const &m, TF const &f, typename M::point_type const &r) DECLARE_FUNCTION_SUFFIX
    ENABLE_IF_DECL_RET_TYPE((traits::iform<TF>::value == FACE),
                            traits::make_nTuple(
                                    gather_impl_(m, f, m.coordinates_global_to_local(r, 6)),
                                    gather_impl_(m, f, m.coordinates_global_to_local(r, 5)),
                                    gather_impl_(m, f, m.coordinates_global_to_local(r, 3))
                            ))

    template<typename M, typename TF> DECLARE_FUNCTION_PREFIX auto
    gather(M const &m, TF const &f, typename M::point_type const &x) DECLARE_FUNCTION_SUFFIX
    ENABLE_IF_DECL_RET_TYPE((traits::iform<TF>::value == VOLUME),
                            gather_impl_(m, f, m.coordinates_global_to_local(x, 7)))

private:
    template<typename M, typename TF, typename IDX, typename TV> DECLARE_FUNCTION_PREFIX void
    scatter_impl_(M const &m, TF &f, IDX const &idx,
                  TV const &v) DECLARE_FUNCTION_SUFFIX
    {

        auto X = (M::_DI) << 1;
        auto Y = (M::_DJ) << 1;
        auto Z = (M::_DK) << 1;

        typename M::point_type r = std::get<1>(idx);
        typename M::index_type s = std::get<0>(idx);

        traits::index(f, ((s + X) + Y) + Z) += v * (r[0]) * (r[1]) * (r[2]);
        traits::index(f, (s + X) + Y) += v * (r[0]) * (r[1]) * (1.0 - r[2]);
        traits::index(f, (s + X) + Z) += v * (r[0]) * (1.0 - r[1]) * (r[2]);
        traits::index(f, s + X) += v * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
        traits::index(f, (s + Y) + Z) += v * (1.0 - r[0]) * (r[1]) * (r[2]);
        traits::index(f, s + Y) += v * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]);
        traits::index(f, s + Z) += v * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]);
        traits::index(f, s) += v * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);

    }


    template<typename M, typename TF, typename TX, typename TV> DECLARE_FUNCTION_PREFIX void
    scatter_(M const &m, std::integral_constant<int, VERTEX>, TF &
    f, TX const &x, TV const &u) DECLARE_FUNCTION_SUFFIX
    {
        scatter_impl_(m, f, m.coordinates_global_to_local(x, 0), u);
    }

    template<typename M, typename TF, typename TX, typename TV> DECLARE_FUNCTION_PREFIX void
    scatter_(M const &m, std::integral_constant<int, EDGE>, TF &
    f, TX const &x, TV const &u) DECLARE_FUNCTION_SUFFIX
    {

        scatter_impl_(m, f, m.coordinates_global_to_local(x, 1), u[0]);
        scatter_impl_(m, f, m.coordinates_global_to_local(x, 2), u[1]);
        scatter_impl_(m, f, m.coordinates_global_to_local(x, 4), u[2]);

    }

    template<typename M, typename TF, typename TX, typename TV> DECLARE_FUNCTION_PREFIX void
    scatter_(M const &m, std::integral_constant<int, FACE>, TF &f,
             TX const &x, TV const &u) DECLARE_FUNCTION_SUFFIX
    {

        scatter_impl_(m, f, m.coordinates_global_to_local(x, 6), u[0]);
        scatter_impl_(m, f, m.coordinates_global_to_local(x, 5), u[1]);
        scatter_impl_(m, f, m.coordinates_global_to_local(x, 3), u[2]);
    }

    template<typename M, typename TF, typename TX, typename TV> DECLARE_FUNCTION_PREFIX void
    scatter_(M const &m, std::integral_constant<int, VOLUME>,
             TF &f, TX const &x, TV const &u) DECLARE_FUNCTION_SUFFIX
    {
        scatter_impl_(m, f, m.coordinates_global_to_local(x, 7), u);
    }

public:
    template<typename M, typename TF, typename ...Args> DECLARE_FUNCTION_PREFIX void
    scatter(M const &m, TF &f, Args &&...args) DECLARE_FUNCTION_SUFFIX
    {
        scatter_(m, traits::iform<TF>(), f, std::forward<Args>(args)...);
    }

private:
    template<typename M, typename TV> DECLARE_FUNCTION_PREFIX TV
    sample_(M const &m, std::integral_constant<int, VERTEX>, typename M::id_type s,
            TV const &v) DECLARE_FUNCTION_SUFFIX { return v; }

    template<typename M, typename TV> DECLARE_FUNCTION_PREFIX TV
    sample_(M const &m, std::integral_constant<int, VOLUME>, typename M::id_type s,
            TV const &v) DECLARE_FUNCTION_SUFFIX { return v; }

    template<typename M, typename TV> DECLARE_FUNCTION_PREFIX TV
    sample_(M const &m, std::integral_constant<int, EDGE>,
            typename M::id_type s, nTuple<TV, 3> const &v) DECLARE_FUNCTION_SUFFIX
    {
        return v[M::sub_index(s)];
    }

    template<typename M, typename TV> DECLARE_FUNCTION_PREFIX TV
    sample_(M const &m, std::integral_constant<int, FACE>,
            typename M::id_type s, nTuple<TV, 3> const &v) DECLARE_FUNCTION_SUFFIX
    {
        return v[M::sub_index(s)];
    }
//
//    template<typename M,int IFORM,  typename TV>
//    DECLARE_FUNCTION_PREFIX TV sample_(M const & m,std::integral_constant<int, IFORM>, typename M::id_type s,
//                                       TV const &v) DECLARE_FUNCTION_SUFFIX { return v; }

public:

//    template<typename M,int IFORM,  typename TV>
//    DECLARE_FUNCTION_PREFIX auto generate(TI const &s, TV const &v) DECLARE_FUNCTION_SUFFIX
//    DECL_RET_TYPE((sample_(M const & m,std::integral_constant<int, IFORM>(), s, v)))


    template<int IFORM, typename M, typename TV>
    DECLARE_FUNCTION_PREFIX auto
    sample(M const &m, typename M::id_type s, TV const &v) DECLARE_FUNCTION_SUFFIX
    DECL_RET_TYPE((sample_(m, std::integral_constant<int, IFORM>(), s, v)))


public:


    LinearInterpolator() { }

    virtual ~LinearInterpolator() { }

    /**
     * A radial basis function (RBF) is a real-valued function whose value depends only
     * on the distance from the origin, so that \f$\phi(\mathbf{x}) = \phi(\|\mathbf{x}\|)\f$;
     * or alternatively on the distance from some other point c, called a center, so that
     * \f$\phi(\mathbf{x}, \mathbf{c}) = \phi(\|\mathbf{x}-\mathbf{c}\|)\f$.
     */
    template<typename M>
    Real RBF(M const &m, typename M::point_type const &x0, typename M::point_type const &x1,
             typename M::vector_type const &a) const
    {
        typename M::vector_type r;
        r = (x1 - x0) / a;
        // @NOTE this is not  an exact  RBF
        return (1.0 - std::abs(r[0])) * (1.0 - std::abs(r[1])) * (1.0 - std::abs(r[2]));
    }

    template<typename M>
    Real RBF(M const &m, typename M::point_type const &x0,
             typename M::point_type const &x1, Real const &a) const
    {

        return (1.0 - m.distance(x1, x0) / a);
    }

};

}}}//namespace simpla
#endif //SIMPLA_LINEAR_H
