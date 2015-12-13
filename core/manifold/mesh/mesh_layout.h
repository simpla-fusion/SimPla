/**
 * @file mesh_layout.h
 * @author salmon
 * @date 2015-12-09.
 */

#ifndef SIMPLA_MESH_LAYOUT_H
#define SIMPLA_MESH_LAYOUT_H

#include <vector>
#include "mesh_block.h"
#include "../enable_patch_from_this.h"


namespace simpla
{
template<typename ...> class Expression;

template<typename ...> class Field;


}

namespace simpla { namespace mesh
{
template<typename ...> class MeshPatch;

template<typename ...> class MeshLayout;


namespace traits
{

template<typename T> T const &
patch(size_t id, T const &f) { return (f); };

template<typename ...T> Field<T...>
patch(size_t id, Field<T...> &f) { return std::move(f.patch(id)); };

template<typename TOP, typename ...Args, int ...I> Field<Expression<TOP, Args...>>
_invoke_patch_helper(size_t id, Field<Expression<TOP, Args...>> const &expr,
                     index_sequence<I...>)
{
    return Field<Expression<TOP, Args...>>(expr.m_op_, patch(id, std::get<I>(expr.args))...);
};

template<typename TOP, typename ...Args> Field<Expression<TOP, Args...> >
patch(size_t id, Field<Expression<TOP, Args...> > const &expr)
{
    return _invoke_patch_helper(id, expr, make_index_sequence<sizeof...(Args)>());
};
}//namespace traits

template<typename TM>
class MeshLayout<TM> : public std::map<size_t, std::shared_ptr<MeshPatch<TM>>>
{
private:
    typedef MeshPatch<TM> patch_type;
    typedef MeshLayout<TM> layout_type;
public:
};

template<typename TM>
class MeshPatch<TM> : public TM, public EnablePatchFromThis<MeshPatch<TM>>
{

private:
    typedef typedef MeshPatch<TM> this_type;
    typedef MeshPatch<TM> mesh_patch_type;
    typedef MeshLayout<TM> layout_type;
    typedef TM mesh_base;
    typedef EnablePatchFromThis<MeshPatch<TM>> patch_base;
public:
    typedef TM mesh_type;
    typedef TM m;
    typedef typename m::id_type id_type;
    typedef typename m::point_type point_type;
    typedef typename m::box_type box_type;
    typedef typename m::index_tuple index_tuple;

    MeshPatch();

    virtual ~MeshPatch();


    template<typename LHS, typename RHS>
    void time_integral(Real dt, LHS &lhs, const RHS &rhs) const;

    using patch_base::patch;

    virtual std::tuple<size_t, std::shared_ptr<this_type>> new_patch();

    virtual void erase_patch(size_t id);

    virtual void refinement(size_t id);

    virtual void coarsen(size_t id);


private:
    std::list<std::weak_ptr<PatchBase> > m_registered_entity_;


};

template<typename TM>
MeshPatch<TM>::MeshPatch()
{
}

template<typename TM>
MeshPatch<TM>::~MeshPatch()
{
}

template<typename TM>
std::tuple<size_t, std::shared_ptr<this_type>> MeshPatch<TM>::new_patch()
{
    size_t id;
    std::shared_ptr<this_type> p;
    std::tie(id, p) = patch_base::new_patch();

    for (auto &item:m_registered_entity_)
    {
        p->m_registered_entity_.push_back(item.lock()->patch(id));
    }
}

template<typename TM>
void MeshPatch<TM>::erase_patch(size_t id)
{
    size_t count = 0;
    for (auto &item:m_registered_entity_)
    {
        item.lock()->erase_patch(id);
    }
    patch_base::erase_patch(id);
}

template<typename TM>
void MeshPatch<TM>::refinement(size_t id)
{
    patch_base::refinement(id);
    for (auto &item:m_registered_entity_)
    {
        item.second->refinement(id);
    }

}

template<typename TM>
void  MeshPatch<TM>::coarsen(size_t id)
{
    patch_base::coarsen(id);
    for (auto &item:m_registered_entity_)
    {
        item.second->coarsen(id);
    }
}


template<typename TM>
template<typename LHS, typename RHS> void
MeshPatch<TM>::time_integral(Real dt, LHS &lhs, const RHS &rhs) const
{
    lhs += rhs * dt;

    lhs.refinement();

    for (int i = 0; i < m_refinement_ratio_; ++i)
    {
        for (auto const &item:patch_base::m_patches_)
        {
            auto f_patch = lhs.patch(item.first);

            item.second->time_integral(dt / m_refinement_ratio_, f_patch,
                                       traits::patch(item.first, rhs));

        }
    }
    lhs.coarsen();

}
}}//namespace simpla { namespace mesh

#endif //SIMPLA_MESH_LAYOUT_H

