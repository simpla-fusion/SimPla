/**
 * @file amr.h
 * @author salmon
 * @date 2015-11-20.
 */

#ifndef SIMPLA_AMR_H
#define SIMPLA_AMR_H

#include <vector>
#include <memory>
#include <list>
#include "../../gtl/type_traits.h"
#include "../../gtl/primitives.h"
#include "../../data_model/dataset.h"
#include "Attribute.h"
#include "Patch.h"

namespace simpla { template<typename ...> class Field; }

namespace simpla { namespace manifold { namespace policy
{

template<typename TMesh>
class AMR
{
private:
    typedef AMR<TMesh> this_type;

public:
    typedef TMesh mesh_type;
    typedef TMesh entity_type;

    typedef typename mesh_type::point_type point_type;
    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::index_tuple index_tuple;
    typedef typename mesh_type::patch_type patch_type;

    typedef this_type amr_policy;

    AMR();

    virtual ~AMR();

    virtual mesh_type &self() = 0;

    virtual mesh_type const &self() const = 0;


    virtual void sync_patch() const { /** TODO sync patches */};

    virtual std::tuple<std::shared_ptr<entity_type>, bool>
    insert(size_t id, std::shared_ptr<entity_type> p)
    {
        auto res = m_patches_.insert(std::make_pair(id, p));
        return std::make_tuple(res.first->second, res.second);
    };

    virtual size_t coarsen(size_t id);

    template<typename ...Args>
    std::tuple<std::shared_ptr<entity_type>, bool> refinement(Args &&...args);

    void refinement_ratio(size_t r) { m_refinement_ratio_ = r; }

    size_t refinement_ratio() const { return m_refinement_ratio_; }


private:

    size_t m_count_ = 0;

    size_t m_refinement_ratio_ = 2;

    std::map<size_t, std::shared_ptr<entity_type>> m_patches_;


};

template<typename TM>
AMR<TM>::AMR() { }

template<typename TM>
AMR<TM>::~AMR() { }

template<typename TM> template<typename ...Args>
std::tuple<std::shared_ptr<TM>, bool>
AMR<TM>::refinement(Args &&...args)
{
    auto res = self().refinement(m_refinement_ratio_, std::forward<Args>(args)...);
    size_t id = m_count_;
    ++m_count_;
    return insert(id, res);
};

template<typename TM>
size_t
AMR<TM>::coarsen(size_t id)
{
    for (auto &entity:self().attributes())
    {
        entity.lock()->patch_entity(id)->coarsen();

        entity.lock()->erase_patch(id);
    }
    return m_patches_.erase(id);
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

template<typename TOP, typename ...Args> Field<Expression<TOP, Args...>>
patch(size_t id, Field<Expression<TOP, Args...>> const &expr)
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
void default_time_integral(AMR<TMesh> const &m, TFun const &fun, Real dt, Args &&...args)
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


AMR::AMR()
{

}

AMR::~AMR()
{

}
}}}//namespace simpla{namespace Manifold{namespace policy

#endif //SIMPLA_AMR_H
