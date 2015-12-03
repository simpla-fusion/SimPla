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
#include "../manifold/manifold_traits.h"
#include "../dataset/dataset.h"
#include "../parallel/parallel.h"
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
    storage_policy m_data_;
public:


    //create construct
    Field(mesh_type const &m) : m_mesh_(m)
    {
    }


    ~Field()
    {
    }

    //copy construct
    Field(this_type const &other)
            : m_data_(other.m_data_), m_mesh_(other.m_mesh_)
    {
    }

    // move construct
    Field(this_type &&other)
            : m_data_(other.m_data_), m_mesh_(other.m_mesh_)
    {
    }

    void swap(this_type &other)
    {
        std::swap(m_mesh_, other.m_mesh_);
        std::swap(m_data_, other.m_data_);
    }

    bool empty() const { return m_data_ == nullptr; }

    void deploy()
    {
        if (m_data_ == nullptr)
            TRY_IT((m_mesh_.template alloc_memory<iform, value_type>(&m_data_)));
    }

    void clear() { m_mesh_.template clear<iform, value_type>(&m_data_); }

    DataSet dataset() const { return std::move(m_mesh_.template dataset<value_type, iform>(m_data_)); }

    DataSet dataset() { return std::move(m_mesh_.template dataset<value_type, iform>(m_data_)); }

    storage_policy &data() { return m_data_; }

    storage_policy const &data() const { return m_data_; }


    /**
     * @name assignment
     * @{
     */
    inline this_type &operator=(this_type const &other)
    {
        action(_impl::_assign(), other);
        return *this;
    }

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

    template<typename TRange, typename Func>
    void accept(TRange const &r0, Func const &fun)
    {
        parallel::parallel_for(r0,
                               [&](TRange const &r)
                               {
                                   for (auto const &item:r)
                                   {
                                       fun(item, (*this)[traits::get<0>(item)]);
                                   }
                               }
        );
    };

    template<typename TRange, typename Func>
    void accept(TRange const &r0, Func const &fun) const
    {
        parallel::parallel_for(r0,
                               [&](TRange const &r)
                               {

                                   for (auto const &item:r)
                                   {
                                       fun(item, (*this)[traits::get<0>(item)]);
                                   }
                               }
        );
    };

private:

    template<typename TOP, typename ...Args>
    void action(TOP const &op, Args &&... args)
    {
        deploy();
        m_mesh_.for_each(op, this, std::forward<Args>(args)...);
    }

    template<typename TOP, typename ...Args>
    void action(TOP const &op, Args &&... args) const
    {

        m_mesh_.for_each(op, *this, std::forward<Args>(args)...);
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

    void sync() { m_mesh_.sync(*this); }

    void declare_as(std::string const &s) { const_cast<mesh_type &>(m_mesh_).enroll(s, this->dataset(), IFORM); }


    value_type &operator[](id_type const &s)
    {
        return m_mesh_.template at<value_type>(m_data_, s);
    }

    value_type const &operator[](id_type const &s) const
    {
        return m_mesh_.template at<value_type>(m_data_, s);
    }

    template<typename ...Args>
    value_type &at(Args &&... args)
    {
        return m_mesh_.template at<value_type>(*this, std::forward<Args>(args)...);
    }

    template<typename ...Args>
    value_type const &at(Args &&... args) const
    {
        return m_mesh_.template at<value_type>(*this, std::forward<Args>(args)...);
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
