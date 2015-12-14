/**
 * @file field_dense.h
 *
 *  Created on: @date{ 2015-1-30}
 *      @author: salmon
 */

#ifndef FIELD_DENSE_H_
#define FIELD_DENSE_H_

#include "field_comm.h"
#include "field_traits.h"


#include <algorithm>
#include <cstdbool>
#include <memory>
#include <string>

#include "../gtl/type_traits.h"
#include "../dataset/dataset.h"
#include "../parallel/parallel.h"
#include "../manifold/mesh/patch.h"
#include "../manifold/manifold_traits.h"


namespace simpla
{
template<typename ...> struct Field;

/**
 * @ingroup field
 * @{
 */

/**
 *  Simple Field
 */
template<typename TG, int IFORM, typename TV>
struct Field<TV, TG, std::integral_constant<int, IFORM> >
        : public mesh::EnablePatchFromThis<Field<TV, TG, std::integral_constant<int, IFORM> >>
{
public:

    typedef TV value_type;

    typedef TG mesh_type;
    typedef mesh::EnablePatchFromThis<Field<TV, TG, std::integral_constant<int, IFORM> >>
            base_type;
    static constexpr int iform = IFORM;
private:

    typedef typename mesh_type::id_type id_type;

    typedef typename mesh_type::point_type point_type;

    typedef Field<value_type, mesh_type, std::integral_constant<int, iform> > this_type;

    typedef Field<value_type, mesh_type, std::integral_constant<int, iform> > field_type;

    typedef typename traits::field_value_type<this_type>::type field_value_type;

    mesh_type const &m_mesh_;

    std::shared_ptr<DataSet> m_dataset_;
public:

    //create construct
    Field(mesh_type const &m) : m_mesh_(m), m_dataset_(nullptr) { }

    //copy construct
    Field(this_type const &other) : m_dataset_(other.m_dataset_), m_mesh_(other.m_mesh_) { }

    // move construct
    Field(this_type &&other) : m_dataset_(other.m_dataset_), m_mesh_(other.m_mesh_) { }

    virtual ~Field() { }

    virtual this_type &self() { return *this; }

    virtual this_type const &self() const { return *this; }

    virtual std::shared_ptr<this_type>
    patch(size_t id)
    {
        std::shared_ptr<this_type> res;

        auto it = base_type::patches().find(id);
        if (it != base_type::patches().end())
        {
            res = it->second;
        }
        else
        {
            std::tie(res, std::ignore) = base_type::insert(
                    id, m_mesh_.patch(id)->template make<this_type>());
        }

        return res;
    }

//    virtual void swap(this_type &other)
//    {
//        std::swap(m_mesh_, other.m_mesh_);
//        std::swap(m_dataset_, other.m_dataset_);
//    }

    virtual bool empty() const { return m_dataset_ == nullptr; }

    virtual void deploy()
    {
        if (m_dataset_ == nullptr)
        {
            m_dataset_ = m_mesh_.template dataset<value_type, iform>();
        }
        m_dataset_->deploy();
    }

    virtual void clear()
    {
        deploy();
        m_dataset_->clear();
    }

    virtual mesh_type const &mesh() const { return m_mesh_; }

    virtual DataSet const &dataset() const { return *m_dataset_; }

    virtual DataSet &dataset()
    {
        deploy();
        return *m_dataset_;
    }


    /**
     * @name assignment
     * @{
     */
    inline this_type &operator=(this_type const &other)
    {
        apply(_impl::_assign(), other);
        return *this;
    }

    template<typename Other>
    inline this_type &operator=(Other const &other)
    {
        apply(_impl::_assign(), other);
        return *this;
    }

    template<typename Other>
    inline this_type &operator+=(Other const &other)
    {
        apply(_impl::plus_assign(), other);
        return *this;
    }

    template<typename Other>
    inline this_type &operator-=(Other const &other)
    {
        apply(_impl::minus_assign(), other);

        return *this;
    }

    template<typename Other>
    inline this_type &operator*=(Other const &other)
    {
        apply(_impl::multiplies_assign(), other);
        return *this;
    }

    template<typename Other>
    inline this_type &operator/=(Other const &other)
    {
        apply(_impl::divides_assign(), other);
        return *this;
    }

    template<typename TRange, typename Func>
    void accept(TRange const &r0, Func const &fun)
    {
        m_mesh_.template for_each_value<value_type, iform>(*this, r0, fun);
    };


private:

    template<typename TOP, typename ...Args>
    void apply(TOP const &op, Args &&... args)
    {
        deploy();
        m_mesh_.apply(op, *this, std::forward<Args>(args)...);
    }

public:
    /** @name as_function
     *  @{*/

    template<typename ...Args>
    auto gather(Args &&...args) const
    DECL_RET_TYPE((m_mesh_.gather(*this, std::forward<Args>(args)...)))

    template<typename ...Args>
    auto operator()(Args &&...args) const
    DECL_RET_TYPE((m_mesh_.gather(*this, std::forward<Args>(args)...)))

    auto range() const
    DECL_RET_TYPE(m_mesh_.template range<iform>())
/**@}*/



public:
    /**
     *  @name as container
     *  @{
     */

    void sync() { m_mesh_.sync(*m_dataset_); }


    value_type &operator[](id_type const &s)
    {
        return m_mesh_.template at<value_type>(*m_dataset_, s);
    }

    value_type const &operator[](id_type const &s) const
    {
        return m_mesh_.template at<value_type>(*m_dataset_, s);
    }

    value_type &at(id_type const &s)
    {
        return m_mesh_.template at<value_type>(*m_dataset_, s);
    }

    value_type at(id_type const &s) const
    {
        return m_mesh_.template at<value_type>(*m_dataset_, s);
    }
/**
 * @}
 */


}; // struct Field

namespace traits
{

template<typename TV, typename TM, typename ...Others>
struct type_id<Field<TV, TM, Others...> >
{
    static const std::string name()
    {
        return "Feild<" + type_id<TV>::name() + " , "
               + type_id<TM>::name() + "," + type_id<Others...>::name() + ">";
    }
};


template<typename TV, typename TM, typename ...Others>
struct mesh_type<Field<TV, TM, Others...> >
{
    typedef TM type;
};


template<typename TV, typename ...Policies>
struct value_type<Field<TV, Policies...>>
{
    typedef TV type;
};

template<typename TV, typename TM, typename TFORM, typename ...Others>
struct iform<Field<TV, TM, TFORM, Others...> > : public TFORM
{
};


template<typename TV, int I, typename TM>
Field<TV, TM, std::integral_constant<int, I> >
make_field(TM const &mesh)
{
    return Field<TV, TM, std::integral_constant<int, I>>(mesh);
};


}// namespace traits

}// namespace simpla

#endif /* FIELD_DENSE_H_ */
