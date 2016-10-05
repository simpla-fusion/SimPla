/**
 * @file linear.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_LINEAR_H
#define SIMPLA_LINEAR_H

#include "../../toolbox/nTuple.h"
#include "../../mesh/EntityId.h"

namespace simpla { namespace manifold { namespace schemes
{
using namespace simpla::mesh;

template<typename TM, class Enable = void>
struct LinearInterpolator { };


/**
 * @ingroup interpolate
 * @brief basic linear interpolate
 */
template<typename TM>
struct LinearInterpolator<TM, std::enable_if_t<std::is_base_of<mesh::MeshEntityIdCoder, TM>::value>>
{
public:

    typedef LinearInterpolator<TM> interpolate_policy;

    LinearInterpolator(TM const &m_) : m(m_) { }

    virtual ~LinearInterpolator() { }

    static std::string class_name() { return "LinearInterpolator"; }

    void deploy() { }


    virtual std::ostream &print(std::ostream &os, int indent = 1) const
    {
        os << std::setw(indent) << " " << "[LinearInterpolator]," << std::endl;
        return os;
    }

private:
    typedef TM mesh_type;
    typedef LinearInterpolator<mesh_type> this_type;
    typedef mesh::MeshEntityIdCoder M;
    mesh_type const &m;

    template<typename TD, typename TIDX>
    inline auto
    gather_impl_(TD const &f, TIDX const &idx) const -> decltype(
    traits::index(f, std::get<0>(idx)) *
    std::get<1>(idx)[0])
    {

        MeshEntityId X = (M::_DI);
        MeshEntityId Y = (M::_DJ);
        MeshEntityId Z = (M::_DK);

        point_type r = std::get<1>(idx);
        MeshEntityId s = std::get<0>(idx);

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

    template<typename TF> constexpr inline traits::field_value_t<TF>
    gather(TF const &f, point_type const &r, ENABLE_IF((traits::iform<TF>::value == VERTEX))) const
    {
        return gather_impl_(f, m.point_global_to_local(r, 0));
    }

    template<typename TF> constexpr inline traits::field_value_t<TF>
    gather(TF const &f, point_type const &r, ENABLE_IF((traits::iform<TF>::value == EDGE))) const
    {
        return traits::field_value_t<TF>{
                gather_impl_(f, m.point_global_to_local(r, 1)),
                gather_impl_(f, m.point_global_to_local(r, 2)),
                gather_impl_(f, m.point_global_to_local(r, 4))
        };
    }

    template<typename TF> constexpr inline traits::field_value_t<TF>
    gather(TF const &f, point_type const &r, ENABLE_IF((traits::iform<TF>::value == FACE))) const
    {
        return traits::field_value_t<TF>{
                gather_impl_(f, m.point_global_to_local(r, 6)),
                gather_impl_(f, m.point_global_to_local(r, 5)),
                gather_impl_(f, m.point_global_to_local(r, 3))
        };
    }

    template<typename TF> constexpr inline traits::field_value_t<TF>
    gather(TF const &f, point_type const &x, ENABLE_IF((traits::iform<TF>::value == VOLUME))) const
    {
        return gather_impl_(f, m.point_global_to_local(x, 7));
    }


private:
    template<typename TF, typename IDX, typename TV>
    inline void
    scatter_impl_(TF &f, IDX const &idx,
                  TV const &v) const
    {

        MeshEntityId X = (M::_DI);
        MeshEntityId Y = (M::_DJ);
        MeshEntityId Z = (M::_DK);

        point_type r = std::get<1>(idx);
        MeshEntityId s = std::get<0>(idx);

        traits::index(f, ((s + X) + Y) + Z) += v * (r[0]) * (r[1]) * (r[2]);
        traits::index(f, (s + X) + Y) += v * (r[0]) * (r[1]) * (1.0 - r[2]);
        traits::index(f, (s + X) + Z) += v * (r[0]) * (1.0 - r[1]) * (r[2]);
        traits::index(f, s + X) += v * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
        traits::index(f, (s + Y) + Z) += v * (1.0 - r[0]) * (r[1]) * (r[2]);
        traits::index(f, s + Y) += v * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]);
        traits::index(f, s + Z) += v * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]);
        traits::index(f, s) += v * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);

    }


    template<typename TF, typename TX, typename TV>
    inline void
    scatter_(std::integral_constant<int, VERTEX>, TF &
    f, TX const &x, TV const &u) const
    {
        scatter_impl_(f, m.point_global_to_local(x, 0), u);
    }

    template<typename TF, typename TX, typename TV>
    inline void
    scatter_(std::integral_constant<int, EDGE>, TF &
    f, TX const &x, TV const &u) const
    {

        scatter_impl_(f, m.point_global_to_local(x, 1), u[0]);
        scatter_impl_(f, m.point_global_to_local(x, 2), u[1]);
        scatter_impl_(f, m.point_global_to_local(x, 4), u[2]);

    }

    template<typename TF, typename TX, typename TV>
    inline void
    scatter_(std::integral_constant<int, FACE>, TF &f,
             TX const &x, TV const &u) const
    {

        scatter_impl_(f, m.point_global_to_local(x, 6), u[0]);
        scatter_impl_(f, m.point_global_to_local(x, 5), u[1]);
        scatter_impl_(f, m.point_global_to_local(x, 3), u[2]);
    }

    template<typename TF, typename TX, typename TV>
    inline void
    scatter_(std::integral_constant<int, VOLUME>,
             TF &f, TX const &x, TV const &u) const
    {
        scatter_impl_(f, m.point_global_to_local(x, 7), u);
    }

public:
    template<typename TF, typename ...Args>
    inline void
    scatter(TF &f, Args &&...args) const
    {
        scatter_(traits::iform<TF>(), f, std::forward<Args>(args)...);
    }

private:
    template<typename TV>
    inline TV
    sample_(std::integral_constant<int, VERTEX>, MeshEntityId const &s, TV const &v) const { return v; }

    template<typename TV>
    inline TV
    sample_(std::integral_constant<int, VOLUME>, MeshEntityId const &s, TV const &v) const { return v; }

    template<typename TV>
    inline TV
    sample_(std::integral_constant<int, EDGE>, MeshEntityId const &s, nTuple<TV, 3> const &v) const
    {
        return v[M::sub_index(s)];
    }

    template<typename TV>
    inline TV
    sample_(std::integral_constant<int, FACE>, MeshEntityId const &s, nTuple<TV, 3> const &v) const
    {
        return v[M::sub_index(s)];
    }
//
//    template<typename M,int IFORM,  typename TV>
//    inline TV sample_(M const & m,std::integral_constant<int, IFORM>, id_type s,
//                                       TV const &v) const { return v; }

public:

//    template<typename M,int IFORM,  typename TV>
//    inline auto generate(TI const &s, TV const &v) const
//    DECL_RET_TYPE((sample_(M const & m,std::integral_constant<int, IFORM>(), s, v)))


    template<int IFORM, typename TV>
    inline traits::value_type_t<TV>
    sample(MeshEntityId const &s, TV const &v) const
    {
        return sample_(std::integral_constant<int, IFORM>(), s, v);
    }
//    DECL_RET_TYPE((sample_(std::integral_constant<int, IFORM>(), s, v)))


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

        return (1.0 - m.distance(x1, x0) / a);
    }

};

}}}//namespace simpla
#endif //SIMPLA_LINEAR_H
