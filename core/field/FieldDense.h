/**
 * @file field_dense.h
 *
 *  Created on: @date{ 2015-1-30}
 *      @author: salmon
 */

#ifndef FIELD_DENSE_H_
#define FIELD_DENSE_H_

#include "FieldTraits.h"


#include <algorithm>
#include <cstdbool>
#include <memory>
#include <string>

#include "../gtl/type_traits.h"
#include "../parallel/parallel.h"
#include "../manifold/ManifoldTraits.h"


namespace simpla
{
template<typename ...> struct Field;

/**
 * @ingroup field
 * @{
 */

template<typename TV, typename TMesh, int IFORM, typename ...Policies>
class Field<TV, TMesh, std::integral_constant<int, IFORM>, Policies...>
        : public Policies ...
//         , public TMesh::template Attribute<TV, IFORM>
//          ,std::enable_shared_from_this<field<TV, TMesh, std::integral_constant<int, IFORM>, Policies...>>
//         , public mesh::EnablePatchFromThis<field<TV, TMesh, std::integral_constant<int, IFORM> >>
{
private:
    typedef Field<TV, TMesh, std::integral_constant<int, IFORM>, Policies...> this_type;
public:
    typedef TMesh mesh_type;

    typedef TV value_type;

    static constexpr int iform = IFORM;

    typedef typename mesh_type::template Attribute<value_type, iform> attribute_type;

    std::shared_ptr<attribute_type> m_data_;
private:

    typedef typename mesh_type::id_type id_type;

    typedef typename mesh_type::point_type point_type;

    typedef typename traits::field_value_type<this_type>::type field_value_type;

    typedef typename this_type::calculus_policy calculus_policy;

    typedef typename this_type::interpolate_policy interpolate_policy;
public:


    //create construct
    Field(mesh_type &m, std::string const &name = "")
            : m_data_(m.template create_attribute<value_type, iform>(name)) { }

    Field(mesh_type const &m)
            : m_data_(m.template create_attribute<value_type, iform>()) { }

    //copy construct
    Field(this_type const &other) : m_data_(other.m_data_) { }

    // move construct
    Field(this_type &&other) : m_data_(other.m_data_) { }

    virtual ~Field() { }

    virtual std::shared_ptr<attribute_type> attribute() { return m_data_; }

    virtual std::shared_ptr<const attribute_type> attribute() const { return m_data_; }

    virtual void sync() { attribute()->sync(); }

    virtual void clear() { attribute()->clear(); }

    virtual void deploy() { attribute()->deploy(); }

    virtual mesh_type const &mesh() const { return attribute()->mesh(); }

    virtual data_model::DataSet data_set() { return attribute()->data_set(); }

    virtual data_model::DataSet data_set() const { return attribute()->data_set(); }

    template<typename ...Args>
    void accept(Args &&...args) { attribute()->accept(std::forward<Args>(args)...); }

    /**
     * @name assignment
     * @{
     */

    value_type &operator[](id_type const &s) { return m_data_->at(s); }

    value_type const &operator[](id_type const &s) const { return m_data_->at(s); }

    inline this_type &operator=(this_type const &other)
    {
        apply(_impl::_assign(), *this, other);
        return *this;
    }

    template<typename Other>
    inline this_type &operator=(Other const &other)
    {
        apply(_impl::_assign(), *this, other);
        return *this;
    }

    template<typename Other>
    inline this_type &operator+=(Other const &other)
    {
        apply(_impl::plus_assign(), *this, other);
        return *this;
    }

    template<typename Other>
    inline this_type &operator-=(Other const &other)
    {
        apply(_impl::minus_assign(), *this, other);

        return *this;
    }

    template<typename Other>
    inline this_type &operator*=(Other const &other)
    {
        apply(_impl::multiplies_assign(), *this, other);
        return *this;
    }

    template<typename Other>
    inline this_type &operator/=(Other const &other)
    {
        apply(_impl::divides_assign(), *this, other);
        return *this;
    }

private:

    template<typename TOP, typename ...Args>
    void apply(TOP const &op, this_type &f, Args &&... args)
    {
        deploy();

        mesh_type const &m = mesh();

        m.template update<IFORM>(
                [&](typename mesh_type::range_type const &r)
                {
                    for (auto const &s:r)
                    {
                        op(calculus_policy::eval(m, f, s),
                           calculus_policy::eval(m, std::forward<Args>(args), s)...);
                    }
                }, f
        );
    }

public:
    /** @name as_function
     *  @{*/

    template<typename ...Args>
    field_value_type gather(Args &&...args) const
    {
        return interpolate_policy::gather(mesh(), *this, std::forward<Args>(args)...);
    }


    template<typename ...Args>
    field_value_type operator()(Args &&...args) const
    {
        return interpolate_policy::gather(mesh(), *this, std::forward<Args>(args)...);
    }


    template<typename Other>
    void assign(id_type const &s, Other const &other)
    {
        m_data_->at(s) = interpolate_policy::template sample<iform>(mesh(), s, other);
    }

    template<typename Other>
    void add(id_type const &s, Other const &other)
    {
        m_data_->at(s) += interpolate_policy::template sample<iform>(mesh(), s, other);
    }

    /**@}*/
    typename mesh_type::range_type range() const { return mesh().template range<iform>(); }


}; // struct field

namespace traits
{


template<typename TV, typename TM, typename ...Others> struct mesh_type<Field<TV, TM, Others...> >
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


}// namespace traits

}// namespace simpla

#endif /* FIELD_DENSE_H_ */
