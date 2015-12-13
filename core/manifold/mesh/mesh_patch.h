/**
 * @file mesh_patch.h
 * @author salmon
 * @date 2015-11-19.
 */

#ifndef SIMPLA_MESH_PATCH_H
#define SIMPLA_MESH_PATCH_H

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
template<typename ...> class Patch;

template<typename ...> class Layout;


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
class Patch<TM> : public EnablePatchFromThis<TM>
{

private:
    typedef Patch<TM> this_type;
    typedef Patch<TM> mesh_patch_type;
    typedef TM mesh_base;
    typedef EnablePatchFromThis<TM> patch_base;


public:
    Patch();

    virtual ~Patch();

    void deploy() { };

    template<typename LHS>
    void sync_patch(LHS &lhs) const { };

    template<typename LHS, typename RHS>
    void time_integral(Real dt, LHS &lhs, const RHS &rhs) const;

    using patch_base::patch;

    template<typename ...Args>
    std::tuple<size_t, std::shared_ptr<this_type>> insert_patch(Args &&...args);

    virtual void erase_patch(size_t id);

    template<typename TF, typename ...Args>
    TF &make(Args &&...args)
    {
        auto res = std::make_shared<TF>(*this, std::forward<Args>(args)...);
        enroll(std::dynamic_pointer_cast<PatchBase>(res));
        return *res;
    }


    template<typename TF>
    void enroll(std::shared_ptr<TF> p)
    {
        m_registered_entities_.push_back(std::dynamic_pointer_cast<PatchBase>(p));
    };


private:

    void enroll(std::shared_ptr<PatchBase> p)
    {
        m_registered_entities_.push_back(p);
    };

    size_t m_count_ = 0;
    size_t m_refinement_ratio_ = 2;
    std::list<std::weak_ptr<PatchBase> > m_registered_entities_;


};

template<typename TM>
Patch<TM>::Patch()
{
}

template<typename TM>
Patch<TM>::~Patch()
{
}


template<typename TM>
template<typename ...Args>
std::tuple<size_t, std::shared_ptr<Patch<TM>>>
Patch<TM>::insert_patch(Args &&...args)
{
    size_t id = m_count_;
    ++m_count_;
    auto p = std::dynamic_pointer_cast<Patch<TM>>
            (patch_base::shared_from_this()->patch(std::forward<Args>(args)..., m_refinement_ratio_));

// TODO check no overlap!

//    auto res = patch_base::insert(id, p);
//    if (std::get<1>(res))
//    {
//        for (auto &entity:  m_registered_entities_)
//        {
//            auto ep = entity.lock()->patch(id);
//
//            ep->refinement();
//
//            p->enroll(ep);
//        }
//    }
//    else
//    {
//        THROW_EXCEPTION_OUT_OF_RANGE("Can not create new patch!");
//    }

    return std::make_tuple(id, p);
}

template<typename TM>
void  Patch<TM>::erase_patch(size_t id)
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
Patch<TM>::time_integral(Real dt, LHS &lhs, const RHS &rhs) const
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
            lhs.sync_patches();
        }
        for (auto const &item:patch_base::m_patches_)
        {
            lhs.patch(item.first)->coarsen();
        }
    }
}

}} //namespace simpla { namespace mesh

#endif //SIMPLA_MESH_PATCH_H
