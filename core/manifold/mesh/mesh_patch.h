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
#include "patch.h"
#include "../../gtl/type_traits.h"
#include "../../gtl/primitives.h"


namespace simpla
{
template<typename ...> class Expression;

template<typename ...> class Field;

template<typename TMesh, template<typename> class ...Policies> class Manifold;
}

namespace simpla { namespace mesh
{
template<typename ...> class Patch;

namespace traits
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
}
//namespace traits
template<typename ...> class MeshPatch;

template<typename TMesh, template<typename> class ...Policies>
class MeshPatch<Manifold<TMesh, Policies...>>
        : public EnablePatchFromThis<Manifold<TMesh, Policies...>>
{

private:
    typedef Manifold<TMesh, Policies...> mesh_type;
    typedef MeshPatch<mesh_type> this_type;

    typedef EnablePatchFromThis<Manifold<TMesh, Policies...>> base_type;
public:

    MeshPatch();

    virtual ~MeshPatch();

    virtual void deploy() { };

    virtual mesh_type &self() = 0;

    virtual mesh_type const &self() const = 0;

    virtual void refinement() { }

    virtual void coarsen() { }

    virtual void sync_patch() const { /** TODO sync patches */};

    virtual size_t erase_patch(size_t id);


    template<typename ...Args>
    std::tuple<std::shared_ptr<mesh_type>, bool> new_patch(Args &&...args);


    template<typename TObj>
    inline std::shared_ptr<TObj> make() const
    {
        auto res = std::make_shared<TObj>(self());
        const_cast<this_type *>(this)->enroll(res);
        return res;
    }


    template<typename TF>
    void enroll(std::shared_ptr<TF> p)
    {
        m_registered_entities_.push_back(std::dynamic_pointer_cast<PatchEntity>(p));
    };

    void refinement_ratio(size_t r) { m_refinement_ratio_ = r; }

    size_t refinement_ratio() const { return m_refinement_ratio_; }

private:

    void enroll(std::shared_ptr<PatchEntity> p) { m_registered_entities_.push_back(p); };

    size_t m_count_ = 0;
    size_t m_refinement_ratio_ = 2;
    std::list<std::weak_ptr<PatchEntity> > m_registered_entities_;


};

template<typename TMesh, template<typename> class ...Policies>
MeshPatch<Manifold<TMesh, Policies...> >::MeshPatch()
{
}

template<typename TMesh, template<typename> class ...Policies>
MeshPatch<Manifold<TMesh, Policies...> >::~MeshPatch()
{
}

template<typename TMesh, template<typename> class ...Policies>
template<typename ...Args>
std::tuple<std::shared_ptr<Manifold<TMesh, Policies...>>, bool>
MeshPatch<Manifold<TMesh, Policies...> >::new_patch(Args &&...args)
{
    auto res = self().refinement(m_refinement_ratio_, std::forward<Args>(args)...);
    res->refinement_ratio(m_refinement_ratio_);
    size_t id = m_count_;
    ++m_count_;
    return base_type::insert(id, res);
};

//
//template<typename TMesh, template<typename> class ...Policies>
//template<typename ...Args>
//std::pair<size_t, std::shared_ptr<Manifold<TMesh, Policies...> > >
//MeshPatch<Manifold<TMesh, Policies...> >::add_patch(Manifold<TMesh, Policies...> const &m)
//{
//    size_t id = m_count_;
//    ++m_count_;
////    auto p = std::dynamic_pointer_cast<MeshPatch<TM>>
////            (
////
////                    patch_base::shared_from_this()->
////
////                            patch(std::forward<Args>(args)
////
////                            ..., m_refinement_ratio_));
//// TODO check no overlap!
////    auto res = patch_base::insert(id, p);
////    if (std::get<1>(res))
////    {
////        for (auto &entity:  m_registered_entities_)
////        {
////            auto ep = entity.lock()->patch(id);
////
////            ep->refinement();
////
////            p->enroll(ep);
////        }
////    }
////    else
////    {
////        THROW_EXCEPTION_OUT_OF_RANGE("Can not create new patch!");
////    }
//
//    return std::make_tuple(id, p);
//}
//
template<typename TMesh, template<typename> class ...Policies> size_t
MeshPatch<Manifold<TMesh, Policies...> >::erase_patch(size_t id)
{
    for (auto &entity:m_registered_entities_)
    {
        entity.lock()->patch_entity(id)->coarsen();

        entity.lock()->erase_patch(id);
    }
    return base_type::erase_patch(id);
}

namespace _impl
{

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
void default_time_integral(MeshPatch<TMesh> const &m, TFun const &fun, Real dt, Args &&...args)
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
                        *item.second, fun,
                        dt / m.refinement_ratio(),
                        traits::patch(item.first, std::forward<Args>(args))...);

            }
            _impl::sync_patches(std::forward<Args>(args)...);
        }
        _impl::coarsen(std::forward<Args>(args)...);
    }
}


}} //namespace simpla { namespace mesh

#endif //SIMPLA_MESH_PATCH_H
