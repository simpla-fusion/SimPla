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
#include <simpla/toolbox/MemoryPool.h>
#include "Algebra.h"
#include "Expression.h"

namespace simpla
{

class DummyMesh
{
    size_type m_dims_[3];
    Real m_lower_[3];
    Real m_upper_[3];

public:
    static constexpr unsigned int ndims = 3;

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

    template<typename TV, size_type IFORM, size_type DOF>
    bool create_data_block(std::shared_ptr<data_block_type<TV, IFORM, DOF> > *p, void *d = nullptr) const
    {
        if (p == nullptr || (*p) != nullptr) { return false; }
        else
        {
            size_type s = m_dims_[0] * m_dims_[1] * m_dims_[2] * DOF *
                          ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
            *p = sp_alloc_array<TV>(s);
        }
    };


    template<typename TFun>
    void foreach(size_type iform, TFun const &fun, size_type dof = 1)
    {
        size_type se = m_dims_[0] * m_dims_[1] * m_dims_[2] * dof *
                       ((iform == VERTEX || iform == VOLUME) ? 1 : 3);

        for (size_type s = 0; s < se; ++s) { fun(s); }
    }
};

namespace algebra { namespace schemes
{
template<typename ...> struct CalculusPolicy;
template<typename ...> struct InterpolatePolicy;

namespace st=simpla::traits;

template<>
struct CalculusPolicy<DummyMesh>
{
    typedef DummyMesh mesh_type;

    template<typename T> static T &
    get_value(T &v) { return v; };

    template<typename T, typename I0> static st::remove_all_extents_t<T, I0> &
    get_value(T &v, I0 const *s, ENABLE_IF((st::is_indexable<T, I0>::value))) { return get_value(v[*s], s + 1); };

    template<typename T, typename I0> static st::remove_all_extents_t<T, I0> &
    get_value(T &v, I0 const *s, ENABLE_IF((!st::is_indexable<T, I0>::value))) { return v; };


    template<typename T, size_type N> static T &
    get_value(declare::nTuple_<T, N> &v, size_type const &s0) { return v[s0 % N]; };

    template<typename T, size_type N> static T const &
    get_value(declare::nTuple_<T, N> const &v, size_type const &s0) { return v[s0 % N]; };

    template<typename V, size_type IFORM, size_type DOF, typename ...Idx> static V const &
    get_value(declare::Field_<V, mesh_type, IFORM, DOF> const &f, Idx &&...idx)
    {
        return f.m_data_[f.m_mesh_->hash(std::forward<Idx>(idx)...)];
    };

    template<typename V, size_type IFORM, size_type DOF, typename ...Idx> static V &
    get_value(declare::Field_<V, mesh_type, IFORM, DOF> &f, Idx &&...idx)
    {
        return f.m_data_[f.m_mesh_->hash(std::forward<Idx>(idx)...)];
    };

private:
    template<typename T, typename ...Args> static T &
    get_value_(std::integral_constant<bool, false> const &, T &v, Args &&...) { return v; }


    template<typename T, typename I0, typename ...Idx> static st::remove_extents_t<T, I0, Idx...> &
    get_value_(std::integral_constant<bool, true> const &, T &v, I0 const &s0, Idx &&...idx)
    {
        return get_value(v[s0], std::forward<Idx>(idx)...);
    };
public:


    template<typename T, typename I0, typename ...Idx> static st::remove_extents_t<T, I0, Idx...> &
    get_value(T &v, I0 const &s0, Idx &&...idx)
    {
        return get_value_(std::integral_constant<bool, st::is_indexable<T, I0>::value>(),
                          v, s0, std::forward<Idx>(idx)...);
    };
public:
    template<typename TOP, typename ...Others, size_type ... index, typename ...Idx> static auto
    _invoke_helper(declare::Expression<TOP, Others...> const &expr, index_sequence<index...>, Idx &&... s)
    DECL_RET_TYPE((TOP::eval(get_value(std::get<index>(expr.m_args_), std::forward<Idx>(s)...)...)))

    template<typename TOP, typename   ...Others, typename ...Idx> static auto
    get_value(declare::Expression<TOP, Others...> const &expr, Idx &&... s)
    DECL_RET_TYPE((_invoke_helper(expr, index_sequence_for<Others...>(), std::forward<Idx>(s)...)))


    template<typename ...Args> static void apply(mesh_type const *, Args &&...args) {}

};

}}

}//namespace simpla { namespace get_mesh

#endif //SIMPLA_DUMMYMESH_H
