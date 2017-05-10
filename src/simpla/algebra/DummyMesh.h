/** 
 * @file DummyMesh.h
 * @author salmon
 * @date 16-5-25 - 上午8:25
 *  */

#ifndef SIMPLA_DUMMYMESH_H
#define SIMPLA_DUMMYMESH_H

#include <memory>

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/type_traits.h>
#include <simpla/utilities/ExpressionTemplate.h>

#include "Algebra.h"
#include "Field.h"

namespace simpla
{

namespace st=simpla::traits;

class DummyMesh
{
public:
    size_type m_dims_[3];
    Real m_lower_[3];
    Real m_upper_[3];


    static constexpr unsigned int ndims = 3;

    typedef DummyMesh mesh_type;

    typedef size_type id_type;


//    template<typename ...Args>
//    DummyMesh(Args &&...args) //mesh::RectMesh(std::forward<Args>(args)...)
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

//    virtual std::shared_ptr<mesh::RectMesh> clone() const
//    {
//        return std::dynamic_pointer_cast<mesh::RectMesh>(std::make_shared<DummyMesh>());
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
    access(algebra::declare::Field_ <V, mesh_type, IFORM, DOF> &f, id_type const &s) const
    {
        return f.m_data_[f.m_mesh_->hash(s)];
    };

    template<typename V, size_type IFORM, size_type DOF> V const &
    access(algebra::declare::Field_ <V, mesh_type, IFORM, DOF> const &f, id_type const &s) const
    {
        return f.m_data_[f.m_mesh_->hash(s)];
    };


    template<typename T> T &
    getValue(T &v, id_type const &,
              ENABLE_IF(algebra::traits::is_scalar<T>::value)) const { return v; };

//    template<typename T>  auto
//    GetValue(T &v, id_type const *s, ENABLE_IF((st::is_indexable<T, id_type>::value)))
//    AUTO_RETURN((GetValue(v[*s], s + 1)))


    template<typename T> auto
    getValue(T &v, id_type const &s0,
              ENABLE_IF((st::is_indexable<T, id_type>::value && !algebra::traits::is_field<T>::value))) const
    AUTO_RETURN((v[s0]));

    template<typename T> auto
    getValue(T &v, id_type const &s0,
              ENABLE_IF((st::is_callable<T(id_type)>::value && !algebra::traits::is_field<T>::value))) const
    AUTO_RETURN((v(s0)));

    template<typename T> auto
    getValue(T &f, id_type const &s, ENABLE_IF((algebra::traits::is_field<T>::value))) const
    AUTO_RETURN((access(f, s)));


    template<typename TOP, typename ...Others, size_type ... index, typename ...Idx> auto
    _invoke_helper(algebra::declare::Expression<TOP, Others...> const &expr, int_sequence<index...>,
                   Idx &&... s) const
    AUTO_RETURN((TOP::eval(getValue(std::get<index>(expr.m_args_), std::forward<Idx>(s)...)...)))

    template<typename TOP, typename   ...Others, typename ...Idx> auto
    getValue(algebra::declare::Expression<TOP, Others...> const &expr, Idx &&... s) const
    AUTO_RETURN((_invoke_helper(expr, int_sequence_for<Others...>(), std::forward<Idx>(s)...)))

    template<typename TV, size_type IFORM, size_type DOF> void
    apply(algebra::declare::Field_ <TV, mesh_type, IFORM, DOF> &lhs, algebra::tags::_clear const &) const
    {
        memset(lhs.m_data_, size(IFORM, DOF) * sizeof(TV), 0);
    }

    template<typename TV, size_type IFORM, size_type DOF, typename TOP, typename TFun> void
    apply(algebra::declare::Field_ <TV, mesh_type, IFORM, DOF> &lhs, TOP const &, TFun const &rhs) const
    {
        foreach([&](mesh_type::id_type const &s) { TOP::eval(getValue(lhs, s), getValue(rhs, s)); }, IFORM, DOF);
    }

};

namespace algebra
{
namespace declare
{
template<typename TV, typename TM, size_type IFORM, size_type DOF> class Field_<TV, TM, IFORM, DOF>;
}

namespace calculus
{
template<typename TV, size_type IFORM, size_type DOF>
struct calculator<declare::Field_<TV, DummyMesh, IFORM, DOF> >
{
    typedef declare::Field_<TV, DummyMesh, IFORM, DOF> self_type;

    typedef declare::Array_<TV,
            traits::rank<DummyMesh>::value +
            ((IFORM == VERTEX || IFORM == VOLUME ? 0 : 1) * DOF > 1 ? 1 : 0)> data_block_type;

    static std::shared_ptr<data_block_type> create_data_block(DummyMesh const *m)
    {
        size_type dims[4] = {m->m_dims_[0], m->m_dims_[1], m->m_dims_[2], 0};
        return std::make_shared<data_block_type>(dims);
    }

    template<typename ...Args> static void
    apply(self_type &, Args &&...args)
    {

    }
};

}//namespace calculus
}//namespace algebra
}//namespace simpla { namespace get_mesh

#endif //SIMPLA_DUMMYMESH_H
