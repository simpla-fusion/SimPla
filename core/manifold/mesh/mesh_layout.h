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
    typedef MeshPatch<TM> this_type;
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

    template<typename ...Args>
    std::tuple<size_t, std::shared_ptr<this_type>>
            insert_patch(Args &&...args);

    virtual void erase_patch(size_t id);


private:

    void enroll(std::shared_ptr<PatchBase> p) { m_registered_entities_.push_back(p); };

    size_t m_count_ = 0;
    size_t m_refinement_ratio_ = 2;
    std::list<std::weak_ptr<PatchBase> > m_registered_entities_;


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
template<typename ...Args>
std::tuple<size_t, std::shared_ptr<MeshPatch<TM>>>
MeshPatch<TM>::insert_patch(Args &&...args)
{
    size_t id = m_count_;
    ++m_count_;
    auto p = std::dynamic_pointer_cast<MeshPatch<TM>>(
            mesh_base::make_patch(m_refinement_ratio_, std::forward<Args>(args)...));

    // TODO check no overlap!

    auto res = patch_base::insert(id, p);
    if (std::get<1>(res))
    {
        for (auto &entity:m_registered_entities_)
        {
            auto ep = entity.lock()->patch(id);

            ep->refinement();

            p->enroll(ep);
        }
    }
    else
    {
        THROW_EXCEPTION_OUT_OF_RANGE("Can not create new patch!");
    }

    return std::make_tuple(id, p);
}

template<typename TM>
void  MeshPatch<TM>::erase_patch(size_t id)
{

    for (auto &entity:m_registered_entities_)
    {
        entity.lock()->patch(id)->coarsen();

        entity.lock()->erase_patch(id);
    }
    patch_base::erase_patch(id);

}


template<typename TM>
template<typename LHS, typename RHS> void
MeshPatch<TM>::time_integral(Real dt, LHS &lhs, const RHS &rhs) const
{
    lhs += rhs * dt;

    if (patch_base::m_patches_.size() > 0)
    {
        for (auto const &item:patch_base::m_patches_)
        {
            lhs.patch(item.first)->refinement();
        }
        for (int i = 0; i < m_refinement_ratio_; ++i)
        {
            for (auto const &item:patch_base::m_patches_)
            {

                item.second->time_integral(
                        dt / m_refinement_ratio_,
                        *lhs.patch(item.first),
                        traits::patch(item.first, rhs));

            }
            //TODO sync patches
        }
        for (auto const &item:patch_base::m_patches_)
        {
            lhs.patch(item.first)->coarsen();
        }
    }
}
}}//namespace simpla { namespace mesh

#endif //SIMPLA_MESH_LAYOUT_H

