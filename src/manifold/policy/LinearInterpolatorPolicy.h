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
using namespace simpla::mesh;


/**
 * @ingroup interpolate
 * @brief basic linear interpolate
 */
template<typename TM>
struct LinearInterpolator
{
public:

    typedef LinearInterpolator<TM> interpolate_policy;

    LinearInterpolator(TM const &m) : m_(m) { }

    virtual ~LinearInterpolator() { }

private:
    typedef TM mesh_type;
    mesh_type const &m_;

    typedef LinearInterpolator this_type;


    template<typename TD, typename TIDX> constexpr inline auto
    gather_impl_(TD const &f, TIDX const &idx) const
    {

        auto X = (mesh_type::_DI) << 1;
        auto Y = (mesh_type::_DJ) << 1;
        auto Z = (mesh_type::_DK) << 1;

        point_type r = std::get<1>(idx);
        index_type s = std::get<0>(idx);

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

    template<typename TF, typename TX> constexpr inline auto
    gather(TF const &f, TX const &r, traits::iform_is_t<TF, VERTEX> *_p = nullptr) const
    {
        return gather_impl_(f, m_.coordinates_global_to_local(r, 0));
    }


    template<typename TF> constexpr inline auto
    gather(TF const &f, point_type const &r, traits::iform_is_t<TF, EDGE> *_p = nullptr) const
    {
        return
                traits::make_nTuple(
                        gather_impl_(f, m_.coordinates_global_to_local(r, 1)),
                        gather_impl_(f, m_.coordinates_global_to_local(r, 2)),
                        gather_impl_(f, m_.coordinates_global_to_local(r, 4))
                );
    }

    template<typename TF> constexpr inline auto
    gather(TF const &f, point_type const &r, traits::iform_is_t<TF, FACE> *_p = nullptr) const
    {
        return
                traits::make_nTuple(
                        gather_impl_(f, m_.coordinates_global_to_local(r, 6)),
                        gather_impl_(f, m_.coordinates_global_to_local(r, 5)),
                        gather_impl_(f, m_.coordinates_global_to_local(r, 3))
                );
    }

    template<typename TF> constexpr inline auto
    gather(TF const &f, point_type const &x, traits::iform_is_t<TF, VOLUME> *_p = nullptr) const
    {
        return gather_impl_(f, m_.coordinates_global_to_local(x, 7));
    }

private:
    template<typename TF, typename IDX, typename TV> inline void
    scatter_impl_(TF &f, IDX const &idx, TV const &v) const
    {

        auto X = (mesh_type::_DI) << 1;
        auto Y = (mesh_type::_DJ) << 1;
        auto Z = (mesh_type::_DK) << 1;

        point_type r = std::get<1>(idx);
        index_type s = std::get<0>(idx);

        traits::index(f, ((s + X) + Y) + Z) += v * (r[0]) * (r[1]) * (r[2]);
        traits::index(f, (s + X) + Y) += v * (r[0]) * (r[1]) * (1.0 - r[2]);
        traits::index(f, (s + X) + Z) += v * (r[0]) * (1.0 - r[1]) * (r[2]);
        traits::index(f, s + X) += v * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
        traits::index(f, (s + Y) + Z) += v * (1.0 - r[0]) * (r[1]) * (r[2]);
        traits::index(f, s + Y) += v * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]);
        traits::index(f, s + Z) += v * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]);
        traits::index(f, s) += v * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);

    }


    template<typename TF, typename TX, typename TV> inline void
    scatter_(index_sequence <VERTEX>, TF &f, TX const &x, TV const &u) const
    {
        scatter_impl_(f, m_.coordinates_global_to_local(x, 0), u);
    }

    template<typename TF, typename TX, typename TV> inline void
    scatter_(index_sequence <EDGE>, TF &f, TX const &x, TV const &u) const
    {

        scatter_impl_(f, m_.coordinates_global_to_local(x, 1), u[0]);
        scatter_impl_(f, m_.coordinates_global_to_local(x, 2), u[1]);
        scatter_impl_(f, m_.coordinates_global_to_local(x, 4), u[2]);

    }

    template<typename TF, typename TX, typename TV> inline void
    scatter_(index_sequence <FACE>, TF &f, TX const &x, TV const &u) const
    {

        scatter_impl_(f, m_.coordinates_global_to_local(x, 6), u[0]);
        scatter_impl_(f, m_.coordinates_global_to_local(x, 5), u[1]);
        scatter_impl_(f, m_.coordinates_global_to_local(x, 3), u[2]);
    }

    template<typename TF, typename TX, typename TV> inline void
    scatter_(index_sequence <VOLUME>, TF &f, TX const &x, TV const &u) const
    {
        scatter_impl_(f, m_.coordinates_global_to_local(x, 7), u);
    }

public:
    template<typename TF, typename ...Args> inline void
    scatter(TF &f, Args &&...args) const
    {
        scatter_(traits::iform<TF>(), f, std::forward<Args>(args)...);
    }

private:
    template<typename TV> constexpr inline TV
    sample_(index_sequence <VERTEX>, id_type s, TV const &v) const { return v; }

    template<typename TV> constexpr inline auto
    sample_(index_sequence <VOLUME>, id_type s, TV const &v) const { return v; }

    template<typename TV> constexpr inline auto
    sample_(index_sequence <EDGE>, id_type s, nTuple<TV, 3> const &v) const
    {
        return v[mesh_type::sub_index(s)];
    }

    template<typename TV> constexpr inline auto
    sample_(index_sequence <FACE>, id_type s, nTuple<TV, 3> const &v) const
    {
        return v[mesh_type::sub_index(s)];
    }
//
//    template<typename M,int IFORM,  typename TV>
//    constexpr inline TV sample_(M const & index_sequence< IFORM>, id_type s,
//                                       TV const &v) const { return v; }

public:

//    template<typename M,int IFORM,  typename TV>
//    constexpr inline auto generate(TI const &s, TV const &v) const
//    DECL_RET_TYPE((sample_(M const & index_sequence< IFORM>(), s, v)))


    template<int IFORM, typename TV> constexpr inline auto
    sample(id_type s, TV const &v) const DECL_RET_TYPE((sample_(index_sequence<IFORM>(), s, v)))


    /**
     * A radial basis function (RBF) is a real-valued function whose value depends only
     * on the distance from the origin, so that \f$\phi(\mathbf{x}) = \phi(\|\mathbf{x}\|)\f$;
     * or alternatively on the distance from some other point c, called a center, so that
     * \f$\phi(\mathbf{x}, \mathbf{c}) = \phi(\|\mathbf{x}-\mathbf{c}\|)\f$.
     */

    Real RBF(point_type const &x0, point_type const &x1, vector_type const &a) const
    {
        vector_type r;
        r = (x1 - x0) / a;
        // @NOTE this is not  an exact  RBF
        return (1.0 - std::abs(r[0])) * (1.0 - std::abs(r[1])) * (1.0 - std::abs(r[2]));
    }


    Real RBF(point_type const &x0, point_type const &x1, Real const &a) const
    {

        return (1.0 - m_.distance(x1, x0) / a);
    }

};

}}}//namespace simpla
#endif //SIMPLA_LINEAR_H
