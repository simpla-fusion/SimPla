/**
 * @file linear.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_LINEAR_H
#define SIMPLA_LINEAR_H

#include <simpla/engine/EntityId.h>
#include <simpla/algebra/ExpressionTemplate.h>
#include <simpla/utilities/macro.h>
#include <simpla/algebra/nTuple.h>
#include <simpla/utilities/type_traits.h>

#include "simpla/algebra/Algebra.h"
#include "Calculus.h"

namespace simpla {

template <typename TM, typename TV, int IFORM, int DOF>
class Field;

namespace manifold {
namespace schemes {
namespace algt = tags;
namespace at = traits;
namespace st = simpla::traits;

/**
 * @ingroup interpolate
 * @brief basic linear interpolate
 */
template <typename TM>
struct InterpolatePolicy {
    typedef TM mesh_type;
    typedef InterpolatePolicy<mesh_type> this_type;
    typedef EntityIdCoder M;

   public:
    InterpolatePolicy() {}

    virtual ~InterpolatePolicy() {}

   private:
    template <typename U, typename M, size_type... I>
    inline U const &eval(Field<U, M, I...> &f, EntityId const &s) {
        return f[s];
    };

    template <typename U, typename M, size_type... I>
    inline U &eval(Field<U, M, I...> const &f, EntityId const &s) {
        return f[s];
    };

    template <typename TD, typename TIDX>
    static inline auto gather_impl_(TD const &f, TIDX const &idx)
        -> decltype(traits::index(f, std::get<0>(idx)) * std::get<1>(idx)[0]) {
        EntityId X = (M::_DI);
        EntityId Y = (M::_DJ);
        EntityId Z = (M::_DK);

        point_type r = std::get<1>(idx);
        EntityId s = std::get<0>(idx);

        return eval(f, ((s + X) + Y) + Z) * (r[0]) * (r[1]) * (r[2]) +    //
               eval(f, (s + X) + Y) * (r[0]) * (r[1]) * (1.0 - r[2]) +    //
               eval(f, (s + X) + Z) * (r[0]) * (1.0 - r[1]) * (r[2]) +    //
               eval(f, (s + X)) * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]) +  //
               eval(f, (s + Y) + Z) * (1.0 - r[0]) * (r[1]) * (r[2]) +    //
               eval(f, (s + Y)) * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]) +  //
               eval(f, s + Z) * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]) +    //
               eval(f, s) * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
    }

   public:
    template <typename TF>
    constexpr static inline at::field_value_t<TF> gather(mesh_type const &m, TF const &f, point_type const &r,
                                                         ENABLE_IF((at::iform<TF>::value == VERTEX))) {
        return gather_impl_(f, m.point_global_to_local(r, 0));
    }

    template <typename TF>
    constexpr static inline at::field_value_t<TF> gather(mesh_type const &m, TF const &f, point_type const &r,
                                                         ENABLE_IF((at::iform<TF>::value == EDGE))) {
        return at::field_value_t<TF>{gather_impl_(f, m.point_global_to_local(r, 1)),
                                     gather_impl_(f, m.point_global_to_local(r, 2)),
                                     gather_impl_(f, m.point_global_to_local(r, 4))};
    }

    template <typename TF>
    constexpr static inline at::field_value_t<TF> gather(mesh_type const &m, TF const &f, point_type const &r,
                                                         ENABLE_IF((at::iform<TF>::value == FACE))) {
        return at::field_value_t<TF>{gather_impl_(f, m.point_global_to_local(r, 6)),
                                     gather_impl_(f, m.point_global_to_local(r, 5)),
                                     gather_impl_(f, m.point_global_to_local(r, 3))};
    }

    template <typename TF>
    constexpr static inline at::field_value_t<TF> gather(mesh_type const &m, TF const &f, point_type const &x,
                                                         ENABLE_IF((at::iform<TF>::value == VOLUME))) {
        return gather_impl_(f, m.point_global_to_local(x, 7));
    }

   private:
    template <typename TF, typename IDX, typename TV>
    static inline void scatter_impl_(TF &f, IDX const &idx, TV const &v) {
        EntityId X = (M::_DI);
        EntityId Y = (M::_DJ);
        EntityId Z = (M::_DK);

        point_type r = std::get<1>(idx);
        EntityId s = std::get<0>(idx);

        eval(f, ((s + X) + Y) + Z) += v * (r[0]) * (r[1]) * (r[2]);
        eval(f, (s + X) + Y) += v * (r[0]) * (r[1]) * (1.0 - r[2]);
        eval(f, (s + X) + Z) += v * (r[0]) * (1.0 - r[1]) * (r[2]);
        eval(f, s + X) += v * (r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
        eval(f, (s + Y) + Z) += v * (1.0 - r[0]) * (r[1]) * (r[2]);
        eval(f, s + Y) += v * (1.0 - r[0]) * (r[1]) * (1.0 - r[2]);
        eval(f, s + Z) += v * (1.0 - r[0]) * (1.0 - r[1]) * (r[2]);
        eval(f, s) += v * (1.0 - r[0]) * (1.0 - r[1]) * (1.0 - r[2]);
    }

    template <typename TF, typename TX, typename TV>
    static inline void scatter_(mesh_type const &m, int_const<VERTEX>, TF &f, TX const &x, TV const &u) {
        scatter_impl_(f, m.point_global_to_local(x, 0), u);
    }

    template <typename TF, typename TX, typename TV>
    static inline void scatter_(mesh_type const &m, int_const<EDGE>, TF &f, TX const &x, TV const &u) {
        scatter_impl_(f, m.point_global_to_local(x, 1), u[0]);
        scatter_impl_(f, m.point_global_to_local(x, 2), u[1]);
        scatter_impl_(f, m.point_global_to_local(x, 4), u[2]);
    }

    template <typename TF, typename TX, typename TV>
    static inline void scatter_(mesh_type const &m, int_const<FACE>, TF &f, TX const &x, TV const &u) {
        scatter_impl_(f, m.point_global_to_local(x, 6), u[0]);
        scatter_impl_(f, m.point_global_to_local(x, 5), u[1]);
        scatter_impl_(f, m.point_global_to_local(x, 3), u[2]);
    }

    template <typename TF, typename TX, typename TV>
    static inline void scatter_(mesh_type const &m, int_const<VOLUME>, TF &f, TX const &x, TV const &u) {
        scatter_impl_(f, m.point_global_to_local(x, 7), u);
    }

   public:
    template <typename TF, typename... Args>
    static inline void scatter(mesh_type const &m, TF &f, Args &&... args) {
        scatter_(m, at::iform<TF>(), f, std::forward<Args>(args)...);
    }

   private:
    template <typename TV>
    static inline TV sample_(mesh_type const &m, int_const<VERTEX>, EntityId const &s, TV const &v) {
        return v;
    }

    template <typename TV, size_type L>
    static inline TV sample_(mesh_type const &m, int_const<VERTEX>, EntityId const &s, nTuple<TV, L> const &v) {
        return v[s.w % L];
    }

    template <typename TV>
    static inline TV sample_(mesh_type const &m, int_const<VOLUME>, EntityId const &s, TV const &v) {
        return v;
    }

    template <typename TV, size_type L>
    static inline TV sample_(mesh_type const &m, int_const<VOLUME>, EntityId const &s, nTuple<TV, L> const &v) {
        return v[s.w % L];
    }

    template <typename TV>
    static inline TV sample_(mesh_type const &m, int_const<EDGE>, EntityId const &s, nTuple<TV, 3> const &v) {
        return v[M::sub_index(s)];
    }

    template <typename TV>
    static inline TV sample_(mesh_type const &m, int_const<FACE>, EntityId const &s, nTuple<TV, 3> const &v) {
        return v[M::sub_index(s)];
    }
    //
    //    template<typename M,int IFORM,  typename TV>
    //    static inline  TV sample_(M const & m,int_const< IFORM>, mesh_id_type s,
    //                                       TV const &v) { return v; }

   public:
    //    template<typename M,int IFORM,  typename TV>
    //    static inline  auto generate(TI const &s, TV const &v)
    //    AUTO_RETURN((sample_(M const & m,int_const< IFORM>(), s, v)))

    template <int IFORM, typename TV>
    static inline at::value_type_t<TV> sample(mesh_type const &m, EntityId const &s, TV const &v) {
        return sample_(m, int_const<IFORM>(), s, v);
    }
    //    AUTO_RETURN((sample_(int_const< IFORM>(), s, v)))

    /**
     * A radial basis function (RBF) is a real-valued function whose value depends only
     * on the distance from the origin, so that \f$\phi(\mathbf{x}) = \phi(\|\mathbf{x}\|)\f$;
     * or alternatively on the distance from some other point c, called a center, so that
     * \f$\phi(\mathbf{x}, \mathbf{c}) = \phi(\|\mathbf{x}-\mathbf{c}\|)\f$.
     */

    Real RBF(mesh_type const &m, point_type const &x0, point_type const &x1, vector_type const &a) {
        vector_type r;
        r = (x1 - x0) / a;
        // @NOTE this is not  an exact  RBF
        return (1.0 - std::abs(r[0])) * (1.0 - std::abs(r[1])) * (1.0 - std::abs(r[2]));
    }

    Real RBF(mesh_type const &m, point_type const &x0, point_type const &x1, Real const &a) {
        return (1.0 - m.distance(x1, x0) / a);
    }

    template <typename V, mesh::MeshEntityType IFORM, size_type DOF, typename U>
    static inline void assign(Field<V, mesh_type, IFORM, DOF> &f, mesh_type const &m, EntityId const &s,
                              nTuple<U, DOF> const &v) {
        for (int i = 0; i < DOF; ++i) { f[M::sw(s, i)] = v[i]; }
    }

    template <typename V, size_type DOF, typename U>
    static inline void assign(Field<V, mesh_type, EDGE, DOF> &f, mesh_type const &m, EntityId const &s,
                              nTuple<U, 3> const &v) {
        for (int i = 0; i < DOF; ++i) { f[M::sw(s, i)] = v[M::sub_index(s)]; }
    }

    template <typename V, size_type DOF, typename U>
    static inline void assign(Field<V, mesh_type, FACE, DOF> &f, mesh_type const &m, EntityId const &s,
                              nTuple<U, 3> const &v) {
        for (int i = 0; i < DOF; ++i) { f[M::sw(s, i)] = v[M::sub_index(s)]; }
    }

    template <typename V, size_type DOF, typename U>
    static inline void assign(Field<V, mesh_type, VOLUME, DOF> &f, mesh_type const &m, EntityId const &s,
                              nTuple<U, DOF> const &v) {
        for (int i = 0; i < DOF; ++i) { f[M::sw(s, i)] = v[i]; }
    }

    template <typename V, mesh::MeshEntityType IFORM, size_type DOF, typename U>
    static inline void assign(Field<V, mesh_type, IFORM, DOF> &f, mesh_type const &m, EntityId const &s, U const &v) {
        for (int i = 0; i < DOF; ++i) { f[M::sw(s, i)] = v; }
    }
};
}
}
}  // namespace simpla
#endif  // SIMPLA_LINEAR_H
