/**
 * @file mesh_patch.h
 * @author salmon
 * @date 2015-11-19.
 */

#ifndef SIMPLA_MESH_PATCH_H
#define SIMPLA_MESH_PATCH_H

#include <vector>
#include <memory>
#include <list>
#include "../../gtl/type_traits.h"
#include "../../gtl/primitives.h"
#include "patch.h"
#include "../../data_model/dataset.h"
#include "../../data_model/attribute.h"


namespace simpla
{
template<typename ...> class Expression;

template<typename ...> class Field;
}

namespace simpla { namespace mesh
{

template<typename TM>
class EnablePatchFromThis : public PatchEntity
{

private:
    typedef EnablePatchFromThis<TM> this_type;

public:

    EnablePatchFromThis();

    virtual ~EnablePatchFromThis();

    virtual void deploy() { };

    virtual void refinement() { }

    virtual void coarsen() { }

    virtual void sync_patch() const { /** TODO sync patches */};

    virtual size_t erase_patch(size_t id);


    template<typename ...Args>
    std::tuple<std::shared_ptr<PatchEntity>, bool> new_patch(Args &&...args);


    void refinement_ratio(size_t r) { m_refinement_ratio_ = r; }

    size_t refinement_ratio() const { return m_refinement_ratio_; }





private:


    size_t m_count_ = 0;
    size_t m_refinement_ratio_ = 2;

    std::list<std::weak_ptr<AttributeBase>> m_registered_attribute_;


};

template<typename TM>
EnablePatchFromThis<TM>::EnablePatchFromThis() { }

template<typename TM>
EnablePatchFromThis<TM>::~EnablePatchFromThis() { }

template<typename TM>
template<typename ...Args>
std::tuple<std::shared_ptr<PatchEntity>, bool>
EnablePatchFromThis<TM>::new_patch(Args &&...args)
{
    auto res = self().refinement(m_refinement_ratio_, std::forward<Args>(args)...);
    res->refinement_ratio(m_refinement_ratio_);
    size_t id = m_count_;
    ++m_count_;
    return insert(id, res);
};

template<typename TM>
size_t
EnablePatchFromThis<TM>::erase_patch(size_t id)
{
    for (auto &entity:m_registered_attribute_)
    {
        entity.lock()->patch_entity(id)->coarsen();

        entity.lock()->erase_patch(id);
    }
    return PatchEntity::erase_patch(id);
}

namespace _impl
{
template<typename T> T const &
patch(size_t id, T const &f) { return (f); };

template<typename ...T> Field<T...>
patch(size_t id, Field<T...> &f) { return std::move(*std::dynamic_pointer_cast<Field<T...> >(f.patch(id))); };

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
#define DEFINE_INVOKE_IF_HAS(_NAME_)                                                                     \
HAS_MEMBER_FUNCTION(_NAME_);                                                                             \
template<typename T>                                                                                     \
auto _NAME_(T &f) -> typename std::enable_if<has_member_function_##_NAME_<T>::value, void>::type         \
{    f._NAME_();}                                                                                        \
template<typename T>                                                                                     \
auto _NAME_(T &f) -> typename std::enable_if<!has_member_function_##_NAME_<T>::value, void>::type        \
{}                                                                                                       \
void _NAME_(){}                                                                                          \
                                                                                                         \
template<typename TFirst, typename ...Args>                                                              \
void _NAME_(TFirst &&first, Args &&... args)                                                             \
{                                                                                                        \
    refinement(std::forward<TFirst>(first));                                                             \
    refinement(std::forward<Args>(args)...);                                                             \
};


DEFINE_INVOKE_IF_HAS(refinement);

DEFINE_INVOKE_IF_HAS(sync_patches);

DEFINE_INVOKE_IF_HAS(coarsen);

#undef DEFINE_INVOKE_IF_HAS


}


template<typename TMesh, typename TFun, typename ...Args>
void default_time_integral(EnablePatchFromThis<TM> const &m, TFun const &fun, Real dt, Args &&...args)
{
    fun(dt, std::forward<Args>(args)...);

    if (m.patches().size() > 0)
    {
        _impl::refinement(std::forward<Args>(args)...);

        for (int i = 0; i < m.refinement_ratio(); ++i)
        {
            for (auto const &item:m.patches())
            {
                default_time_integral(
                        *std::dynamic_pointer_cast<PatchEntity>(item.second), fun,
                        dt / m.refinement_ratio(),
                        _impl::patch(item.first, std::forward<Args>(args))...);
            }
            _impl::sync_patches(std::forward<Args>(args)...);
        }
        _impl::coarsen(std::forward<Args>(args)...);
    }
}


}} //namespace simpla { namespace mesh

#endif //SIMPLA_MESH_PATCH_H
