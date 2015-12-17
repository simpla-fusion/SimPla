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
#include "../parallel/parallel.h"
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
template<typename TV, typename TMesh, int IFORM, typename ...Policies>
class Field<TV, TMesh, std::integral_constant<int, IFORM>, Policies...>
        : public TMesh::template Attribute<TV, IFORM>,
          public Policies ...
//          ,std::enable_shared_from_this<Field<TV, TMesh, std::integral_constant<int, IFORM>, Policies...>>
//         , public mesh::EnablePatchFromThis<Field<TV, TMesh, std::integral_constant<int, IFORM> >>
{
private:
    typedef Field<TV, TMesh, std::integral_constant<int, IFORM>, Policies...> this_type;
public:
    typedef typename TMesh::template Attribute<TV, IFORM> base_type;

    typedef TMesh mesh_type;
    typedef TV value_type;

    using base_type::mesh;
    using base_type::sync;

    static constexpr int iform = IFORM;
private:
//    typedef typename mesh::template EnablePatchFromThis<this_type> patch_base;

    typedef typename mesh_type::id_type id_type;

    typedef typename mesh_type::point_type point_type;

    typedef typename traits::field_value_type<this_type>::type field_value_type;

    typedef typename this_type::calculus_policy calculus_policy;
    typedef typename this_type::interpolate_policy interpolate_policy;
public:


    //create construct
    Field(mesh_type &m, std::string const &name = "")
            : base_type(*m.template create_attribute<this_type>(name)) { }

    Field(mesh_type const &m) : base_type(m) { }

    //copy construct
    Field(this_type const &other) : base_type(other) { }

    // move construct
    Field(this_type &&other) : base_type(other) { }

    virtual ~Field() { }


    /**
     * @name assignment
     * @{
     */
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
        base_type::deploy();
//        calculus_policy::template apply<IFORM>(mesh(), op, *this, std::forward<Args>(args)...);

        mesh_type const &m = mesh();

        m.template update<IFORM>(
//                m.template range<IFORM>(),
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
        this->at(s) = interpolate_policy::template sample<iform>(mesh(), s, other);
    }

    template<typename Other>
    void add(id_type const &s, Other const &other)
    {
        this->at(s) += interpolate_policy::template sample<iform>(mesh(), s, other);
    }
    /**@}*/






}; // struct Field

namespace traits
{


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


}// namespace traits

}// namespace simpla

#endif /* FIELD_DENSE_H_ */
