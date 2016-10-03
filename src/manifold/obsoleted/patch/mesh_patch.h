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
#include "../../toolbox/type_traits.h"
#include "../../sp_def.h"
#include "../../data_model/dataset.h"
#include "Attribute.h"
#include "Patch.h"


namespace simpla
{
template<typename ...> class Expression;

template<typename ...> class Field;
}

namespace simpla { namespace mesh
{


template<typename TM, typename TV, int IFORM>
class PatchPolicy : public PatchEntity
{
    typedef TM mesh_type;

    typedef TV value_type;

    static constexpr int iform = IFORM;

    typedef typename mesh_type::template Attribute<value_type, iform> attribute_type;

    virtual attribute_type &self() = 0;

    virtual attribute_type const &self() const = 0;

    virtual mesh_type const &mesh() const = 0;

    std::map<size_t, std::shared_ptr<attribute_type> > m_patches_;


};

template<typename TM>
class MeshPatch
{

private:
    typedef MeshPatch<TM> this_type;
    typedef TM mesh_type;
    typedef typename mesh_type::box_type box_type;
public:

    MeshPatch();

    virtual ~MeshPatch();

    virtual this_type &self() = 0;

    virtual this_type const &self() const = 0;

    virtual void deploy() { };

    virtual std::shared_ptr<mesh_type> create_patch(box_type const &b, int refinement_ratio) const = 0;

    virtual std::tuple<iterator, bool> refinement(box_type const &);

    virtual void coarsen(size_t id);

    virtual size_t erase_patch(size_t id);


    void refinement_ratio(size_t r) { m_refinement_ratio_ = r; }

    size_t refinement_ratio() const { return m_refinement_ratio_; }


private:


    size_t m_count_ = 0;
    size_t m_refinement_ratio_ = 2;

    std::map<size_t, std::shared_ptr<AttributeBase>> m_patches_;
    typedef typename std::map<size_t, std::shared_ptr<AttributeBase>>::iterator iterator;


};

template<typename TM>
MeshPatch<TM>::MeshPatch() { }

template<typename TM>
MeshPatch<TM>::~MeshPatch() { }

template<typename TM>
template<typename ...Args>
std::tuple<iterator, bool>
MeshPatch<TM>::refinement(box_type const &b)
{
    auto m = std::make_shared<TM>(self());
    m->patch(b);
    m->refinement(m_refinement_ratio_);
    size_t id = m_count_;
    ++m_count_;
    return m_patches_.insert(id, m         );
};

template<typename TM>
size_t
MeshPatch<TM>::erase_patch(size_t id)
{
    size_t res = 0;
    if (m_patches_.find(id) != m_patches_.end())
    {
        for (auto &attr:self().attributes())
        {
            auto p = attr.lock();
            if (p != nullptr)
            {
                p->patch_entity(id)->coarsen();
                p->erase_patch(id);
            }
        }
        res = m_patches_.erase(id);
    }

    return res;
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
void default_time_integral(MeshPatch<TM> const &m, TFun const &fun, Real dt, Args &&...args)
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


}} //namespace simpla { namespace get_mesh

#endif //SIMPLA_MESH_PATCH_H
