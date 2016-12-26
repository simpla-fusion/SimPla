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

namespace simpla
{

class DummyMesh
{

public:
    static constexpr unsigned int ndims = 3;

    template<typename TV, size_type IFORM, size_type DOF = 1> using data_block_type=
    traits::add_extents_t<TV, 1, 1, 1, ((IFORM == VERTEX || IFORM == VOLUME) ? DOF : DOF * 3)>;

    template<typename ...Args>
    DummyMesh(Args &&...args) //mesh::MeshBlock(std::forward<Args>(args)...)
    {}

    ~DummyMesh() {}

//    template<typename TV, mesh::MeshEntityType IFORM> using data_block_type= mesh::DataBlockArray<Real, IFORM>;

//    virtual std::shared_ptr<mesh::MeshBlock> clone() const
//    {
//        return std::dynamic_pointer_cast<mesh::MeshBlock>(std::make_shared<DummyMesh>());
//    };

    template<typename TID>
    size_type hash(TID const &s) const { return 0; }
//    template<typename TV, size_type IFORM>
//    std::shared_ptr<mesh::DataBlock> create_data_block(void *p) const
//    {
//        auto b = outer_index_box();
//
//        index_type lo[4] = {std::get<0>(b)[0], std::get<0>(b)[1], std::get<0>(b)[2], 0};
//        index_type hi[4] = {std::get<1>(b)[0], std::get<1>(b)[1], std::get<0>(b)[2], 3};
//        return std::dynamic_pointer_cast<mesh::DataBlock>(
//                std::make_shared<data_block_type<TV, IFORM>>(
//                        static_cast<TV *>(p),
//                        (IFORM == VERTEX || IFORM == VOLUME) ? 3 : 4,
//                        lo, hi));
//    };


    template<typename ...Args>
    Real eval(Args &&...args) const { return 1.0; };
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

    template<typename T, size_type N> static T &
    get_value(declare::nTuple_ <T, N> &v, size_type const &s0) { return v[s0]; };

    template<typename T, size_type N> static T const &
    get_value(declare::nTuple_ <T, N> const &v, size_type const &s0) { return v[s0]; };
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
