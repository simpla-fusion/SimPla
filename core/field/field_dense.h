/**
 * @file field_dense.h
 *
 *  Created on: @date{ 2015-1-30}
 *      @author: salmon
 */

#ifndef FIELD_DENSE_H_
#define FIELD_DENSE_H_

#include "field_comm.h"


#include <algorithm>
#include <cstdbool>
#include <memory>
#include <string>

#include "../gtl/type_traits.h"

#include "../manifold/domain_traits.h"
#include "../manifold/manifold_traits.h"
#include "../dataset/dataset_traits.h"
#include "../parallel/distributed_object.h"

#include "field_traits.h"


namespace simpla
{
template<typename ...>
struct Field;

/**
 * @ingroup field
 * @{
 */

/**
 *  Simple Field
 */
template<typename TG, int IFORM, typename TV>
struct Field<TV, TG, std::integral_constant<int, IFORM> >
        : public TG::template storage_type<TV>
{
public:

    typedef TV value_type;

    typedef TG mesh_type;

    static constexpr int iform = IFORM;

private:

    typedef typename TG::template storage_type<TV> storage_policy;

    typedef typename mesh_type::id_type id_type;

    typedef typename mesh_type::point_type point_type;

    typedef Field<value_type, mesh_type, std::integral_constant<int, IFORM> > this_type;

    typedef typename traits::field_value_type<this_type>::type field_value_type;

    mesh_type const &m_mesh_;

public:


    //create construct
    Field(mesh_type const &m)
            : m_mesh_(m)
    {
    }


    ~Field()
    {
    }

    //copy construct
    Field(this_type const &other)
            : storage_policy(other), m_mesh_(other.m_mesh_)
    {
    }

    // move construct
    Field(this_type &&other)
            : storage_policy(other), m_mesh_(other.m_mesh_)
    {
    }


    void swap(this_type &other)
    {
        std::swap(m_mesh_, other.m_mesh_);
        storage_policy::swap(other);
    }


    /**
     * @name assignment
     * @{
     */
    template<typename Other>
    inline this_type &operator=(Other const &other)
    {
        action(_impl::_assign(), other);
        return *this;
    }

    template<typename Other>
    inline this_type &operator+=(Other const &other)
    {
        action(_impl::plus_assign(), other);
        return *this;
    }

    template<typename Other>
    inline this_type &operator-=(Other const &other)
    {
        action(_impl::minus_assign(), other);

        return *this;
    }

    template<typename Other>
    inline this_type &operator*=(Other const &other)
    {

        action(_impl::multiplies_assign(), other);

        return *this;
    }

    template<typename Other>
    inline this_type &operator/=(Other const &other)
    {
        action(_impl::divides_assign(), other);
        return *this;
    }

private:

    template<typename TOP, typename ...Args>
    void action(TOP const &op, Args &&... args)
    {
        m_mesh_.template action<iform>(op, *this, std::forward<Args>(args)...);
    }

    template<typename TOP, typename ...Args>
    void action(TOP const &op, Args &&... args) const
    {
        m_mesh_.template action<iform>(op, *this, std::forward<Args>(args)...);
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

    void clear()
    {
        m_mesh_.clear(*this);
    }
    void sync()
    {
        m_mesh_.sync(*this);
    }
    value_type &operator[](id_type const &s)
    {
        return m_mesh_.access(*this, s);
    }

    value_type const &operator[](id_type const &s) const
    {
        return m_mesh_.access(*this, s);
    }

    template<typename ...Args>
    value_type &at(Args &&... args)
    {
        return m_mesh_.access(*this, std::forward<Args>(args)...);
    }

    template<typename ...Args>
    value_type const &at(Args &&... args) const
    {
        return m_mesh_.access(*this, std::forward<Args>(args)...);
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
        return "Feild<" + type_id<TV>::name() + " , " + type_id<TM>::name() + "," + type_id<Others...>::name() + ">";
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
