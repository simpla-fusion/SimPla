/**
 * @file DummyMesh.h
 * @author salmon
 * @date 16-5-25 - 上午8:25
 *  */

#ifndef SIMPLA_DUMMYMESH_H
#define SIMPLA_DUMMYMESH_H

#include <memory>

#include "simpla/SIMPLA_config.h"
#include "simpla/algebra/ExpressionTemplate.h"
#include "simpla/utilities/type_traits.h"

#include "simpla/algebra/Algebra.h"
#include "simpla/physics/Field.h"

namespace simpla {

class DummyMesh : public engine::MeshBase {
   public:
    size_type m_dims_[3];
    Real m_lower_[3];
    Real m_upper_[3];

    static constexpr unsigned int ndims = 3;

    typedef DummyMesh mesh_type;

    DummyMesh(size_type const *dims, Real const *lower, Real const *upper)
        : m_dims_{dims[0], dims[1], dims[2]},
          m_lower_{lower[0], lower[1], lower[2]},
          m_upper_{upper[0], upper[1], upper[2]}, MeshBase(<#initializer#>) {}

    ~DummyMesh() = default;

    template <typename TFun>
    void Foreach(TFun const &fun, size_type iform = NODE, size_type dof = 1) const {}

    template <typename V, size_type IFORM, size_type DOF>
    V &getValue(Field<V, mesh_type, IFORM, DOF> &f, EntityId const &s) const {
        return f[s];
    };

    template <typename V, size_type IFORM, size_type DOF>
    V const &getValue(Field<V, mesh_type, IFORM, DOF> const &f, EntityId const &s) const {
        return f[s];
    };

    template <typename T>
    T &getValue(T &v, EntityId const &, ENABLE_IF(traits::is_scalar<T>::value)) const {
        return v;
    };

    //    template<typename T>  auto
    //    GetEntity(T &v, EntityId const *s, ENABLE_IF((st::is_indexable<T, EntityId>::value)))
    //    AUTO_RETURN((GetEntity(v[*s], s + 1)))

    template <typename T>
    auto getValue(T &v, EntityId const &s0,
                  ENABLE_IF((is_indexable<T, EntityId>::value && !traits::is_field<T>::value))) const {
        return ((v[s0]));
    }

    template <typename T>
    auto getValue(T &v, EntityId const &s0,
                  ENABLE_IF((is_callable<T(EntityId)>::value && !traits::is_field<T>::value))) const {
        return ((v(s0)));
    }

    template <typename TOP, typename... Others, size_type... index, typename... Idx>
    auto _invoke_helper(Expression<TOP, Others...> const &expr, int_sequence<index...>, Idx &&... s) const {
       return TOP::eval(getValue(std::get<index>(expr.m_args_), std::forward<Idx>(s)...)...));
    };

    template <typename TV, size_type IFORM, size_type DOF, typename TOP, typename TFun>
    void getValue(Field<TV, mesh_type, IFORM, DOF> &lhs, TOP const &, TFun const &rhs) const {}
};

}  // namespace simpla { namespace get_mesh

#endif  // SIMPLA_DUMMYMESH_H
