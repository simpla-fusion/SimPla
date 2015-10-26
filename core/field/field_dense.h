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
struct Field<Domain<TG, std::integral_constant<int, IFORM> >, TV>
        : public DistributedObject, public DataSet
{
public:

    typedef TV value_type;

    typedef TG mesh_type;

    typedef Domain<TG, std::integral_constant<int, IFORM> > domain_type;

    static constexpr int iform = traits::iform<domain_type>::value;

private:
    typedef DataSet storage_policy;

    typedef typename mesh_type::id_type id_type;

    typedef typename mesh_type::point_type point_type;

    typedef Field<domain_type, value_type> this_type;

    typedef typename traits::field_value_type<this_type>::type field_value_type;


    domain_type m_domain_;

    mesh_type const &m_mesh_;

public:


    //create construct
    Field(mesh_type const &m)
            : DistributedObject(m.comm()), m_domain_(m), m_mesh_(m)
    {
        DataSet::datatype = traits::datatype<value_type>::create();
        DataSet::dataspace = m_mesh_.template dataspace<iform>();
    }


    ~Field()
    {
    }

    //copy construct
    Field(this_type const &other)
            : DistributedObject(other), DataSet(other), m_domain_(other.m_domain_), m_mesh_(other.m_mesh_)
    {
    }

    // move construct
    Field(this_type &&other)
            : DistributedObject(other), DataSet(other), m_domain_(other.m_domain_), m_mesh_(other.m_mesh_)
    {
    }

    // split construct
    template<typename TSplit>
    Field(this_type &f, TSplit const &split)
            : DistributedObject(f.mesh().comm()), m_domain_(f.domain(), split), m_mesh_(f.mesh())
    {
        DataSet::datatype = f.DataSet::datatype;
        DataSet::data = f.DataSet::data;
    }


    void swap(this_type &other)
    {
        std::swap(m_domain_, other.m_domain_);
        std::swap(m_mesh_, other.m_mesh_);
        DistributedObject::swap(other);
        DataSet::swap(other);
    }

    void deploy()
    {
        if (!DataSet::is_valid())
        {
            DataSet::deploy();

            for (auto const &item :  m_mesh_.template connections<iform>())
            {
                DistributedObject::add_link_send(
                        &item.coord_offset[0],
                        m_mesh_.template dataspace<iform>(item.send_range),
                        DataSet::datatype, &(DataSet::data));

                DistributedObject::add_link_recv(
                        &item.coord_offset[0],
                        m_mesh_.template dataspace<iform>(item.recv_range),
                        DataSet::datatype, &(DataSet::data));
            }
        }
    }

    domain_type const &domain() const { return m_domain_; }

    domain_type &domain() { return m_domain_; }


    /**
     * @name assignment
     * @{
     */



    template<typename Other>
    inline this_type &operator=(Other const &other)
    {
        action(_impl::_assign(), *this, other);
        return *this;
    }

    template<typename Other>
    inline this_type &operator+=(Other const &other)
    {


        action(_impl::plus_assign(), *this, other);
        return *this;
    }

    template<typename Other>
    inline this_type &operator-=(Other const &other)
    {
        action(_impl::minus_assign(), *this, other);

        return *this;
    }

    template<typename Other>
    inline this_type &operator*=(Other const &other)
    {

        action(_impl::multiplies_assign(), *this, other);

        return *this;
    }

    template<typename Other>
    inline this_type &operator/=(Other const &other)
    {
        action(_impl::divides_assign(), *this, other);
        return *this;
    }

    template<typename TOP, typename ...Args>
    void action(TOP const &op, Args &&... args)
    {
        deploy();

        DistributedObject::wait();

        m_domain_.action(op, std::forward<Args>(args)...);

        DistributedObject::sync();
    }

    template<typename TOP, typename ...Args>
    void action(TOP const &op, Args &&... args) const
    {
        m_domain_.action(op, std::forward<Args>(args)...);
    }

    /** @name as_function
     *  @{*/

    template<typename ...Args>
    auto gather(Args &&...args) const
    DECL_RET_TYPE((m_mesh_.gather(*this, std::forward<Args>(args)...)))

    template<typename ...Args>
    auto operator()(Args &&...args) const
    DECL_RET_TYPE((m_mesh_.gather(*this, std::forward<Args>(args)...)))


/**@}*/



public:
    /**
     *  @name as container
     *  @{
     */
    value_type &operator[](id_type const &s)
    {
        return DataSet::template get_value<value_type>(m_mesh_.hash(s));
    }

    value_type const &operator[](id_type const &s) const
    {
        return DataSet::template get_value<value_type>(m_mesh_.hash(s));
    }

    template<typename ...Args>
    value_type &at(Args &&... args)
    {
        return DataSet::template get_value<value_type>(m_mesh_.hash(std::forward<Args>(args)...));
    }

    template<typename ...Args>
    value_type const &at(Args &&... args) const
    {
        return DataSet::template get_value<value_type>(m_mesh_.hash(std::forward<Args>(args)...));
    }
/**
 * @}
 */


}; // struct Field

namespace traits
{

template<typename ... TM, typename ...Others>
struct type_id<Field<Domain<TM ...>, Others...>>
{
    static const std::string name()
    {
        return "Feild<" + type_id<Domain<TM...> >::name() + "," + type_id<Others...>::name() + ">";
    }
};


template<typename ... TM, typename TV, typename ...Policies>
struct value_type<Field<Domain<TM ...>, TV, Policies...>>
{
    typedef TV type;
};

template<int I, typename TV, typename TM>
Field<Domain<TM, std::integral_constant<int, I>>, TV>
make_field(TM const &mesh)
{
    return Field<Domain<TM, std::integral_constant<int, I>>, TV>(mesh);
};


}// namespace traits

}// namespace simpla

#endif /* FIELD_DENSE_H_ */
