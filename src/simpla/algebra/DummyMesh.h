/** 
 * @file DummyMesh.h
 * @author salmon
 * @date 16-5-25 - 上午8:25
 *  */

#ifndef SIMPLA_DUMMYMESH_H
#define SIMPLA_DUMMYMESH_H

#include <memory>

#include <simpla/SIMPLA_config.h>
#include <simpla/mpl/type_traits.h>
#include "Algebra.h"
#include "Expression.h"
#include "Arithmetic.h"
#include "Field.h"

namespace simpla
{

namespace st=simpla::traits;

class DummyMesh
{
    size_type m_dims_[3];
    Real m_lower_[3];
    Real m_upper_[3];

public:
    static constexpr unsigned int ndims = 3;

    typedef DummyMesh mesh_type;

    typedef size_type id_type;


//    template<typename ...Args>
//    DummyMesh(Args &&...args) //mesh::MeshBlock(std::forward<Args>(args)...)
//    {}

    DummyMesh(size_type const *dims, Real const *lower, Real const *upper) :
            m_dims_{dims[0], dims[1], dims[2]},
            m_lower_{lower[0], lower[1], lower[2]},
            m_upper_{upper[0], upper[1], upper[2]}
    {

    }

    ~DummyMesh() {}

    void deploy() {}



//    template<typename TV, mesh::MeshEntityType IFORM> using data_block_type= mesh::DataBlockArray<Real, IFORM>;

//    virtual std::shared_ptr<mesh::MeshBlock> clone() const
//    {
//        return std::dynamic_pointer_cast<mesh::MeshBlock>(std::make_shared<DummyMesh>());
//    };

    template<typename TID>
    size_type hash(TID const &s) const { return s; }

    template<typename TV, size_type IFORM, size_type DOF = 1> using data_block_type=TV;
//    traits::add_extents_t<TV, 1, 1, 1, ((IFORM == VERTEX || IFORM == VOLUME) ? DOF : DOF * 3)>;

    size_type size(size_type IFORM = VERTEX, size_type DOF = 1) const
    {
        return m_dims_[0] * m_dims_[1] * m_dims_[2] * DOF * ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
    }

    template<typename TV, size_type IFORM, size_type DOF>
    bool create_data_block(std::shared_ptr<data_block_type<TV, IFORM, DOF> > *p, void *d = nullptr) const
    {
        if (p == nullptr || (*p) != nullptr) { return false; }
        else
        {
            size_type s = size(IFORM, DOF);
            *p = std::shared_ptr<TV>(new TV[s]);
        }
    };


    template<typename TFun>
    void foreach(TFun const &fun, size_type iform = VERTEX, size_type dof = 1) const
    {
        size_type se = size(iform, dof);

        for (size_type s = 0; s < se; ++s) { fun(s); }
    }


    template<typename V, size_type IFORM, size_type DOF> V &
    access(algebra::declare::Field_<V, mesh_type, IFORM, DOF> &f, id_type const &s) const
    {
        return f.m_data_[f.m_mesh_->hash(s)];
    };

    template<typename V, size_type IFORM, size_type DOF> V const &
    access(algebra::declare::Field_<V, mesh_type, IFORM, DOF> const &f, id_type const &s) const
    {
        return f.m_data_[f.m_mesh_->hash(s)];
    };


    template<typename T> T &
    get_value(T &v, id_type const &,
              ENABLE_IF(algebra::traits::is_scalar<T>::value)) const { return v; };

//    template<typename T>  auto
//    get_value(T &v, id_type const *s, ENABLE_IF((st::is_indexable<T, id_type>::value)))
//    DECL_RET_TYPE((get_value(v[*s], s + 1)))


    template<typename T> auto
    get_value(T &v, id_type const &s0,
              ENABLE_IF((st::is_indexable<T, id_type>::value && !algebra::traits::is_field<T>::value))) const
    DECL_RET_TYPE((v[s0]));

    template<typename T> auto
    get_value(T &v, id_type const &s0,
              ENABLE_IF((st::is_callable<T(id_type)>::value && !algebra::traits::is_field<T>::value))) const
    DECL_RET_TYPE((v(s0)));

    template<typename T> auto
    get_value(T &f, id_type const &s, ENABLE_IF((algebra::traits::is_field<T>::value))) const
    DECL_RET_TYPE((access(f, s)));


    template<typename TOP, typename ...Others, size_type ... index, typename ...Idx> auto
    _invoke_helper(algebra::declare::Expression<TOP, Others...> const &expr, index_sequence<index...>,
                   Idx &&... s) const
    DECL_RET_TYPE((TOP::eval(get_value(std::get<index>(expr.m_args_), std::forward<Idx>(s)...)...)))

    template<typename TOP, typename   ...Others, typename ...Idx> auto
    get_value(algebra::declare::Expression<TOP, Others...> const &expr, Idx &&... s) const
    DECL_RET_TYPE((_invoke_helper(expr, index_sequence_for<Others...>(), std::forward<Idx>(s)...)))

    template<typename TV, size_type IFORM, size_type DOF> void
    apply(algebra::declare::Field_<TV, mesh_type, IFORM, DOF> &lhs, algebra::tags::_clear const &) const
    {
        memset(lhs.m_data_, size(IFORM, DOF) * sizeof(TV), 0);
    }

    template<typename TV, size_type IFORM, size_type DOF, typename TOP, typename TFun> void
    apply(algebra::declare::Field_<TV, mesh_type, IFORM, DOF> &lhs, TOP const &, TFun const &rhs) const
    {
        foreach([&](mesh_type::id_type const &s) { TOP::eval(get_value(lhs, s), get_value(rhs, s)); }, IFORM, DOF);
    }

};


}//namespace simpla { namespace get_mesh

#endif //SIMPLA_DUMMYMESH_H
