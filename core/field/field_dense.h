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
//#include "../manifold/patch/patch.h"


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
template<typename TV, typename TG, int IFORM>
class Field<TV, TG, std::integral_constant<int, IFORM> >
        : public TG::template Attribute<TV, IFORM>
//         , public mesh::EnablePatchFromThis<Field<TV, TG, std::integral_constant<int, IFORM> >>
{
private:
    typedef Field<TV, TG, std::integral_constant<int, IFORM> > this_type;
public:
    typedef typename TG::template Attribute<TV, IFORM> attribute_type;

    typedef TG mesh_type;
    typedef TV value_type;

    using attribute_type::mesh;
    using attribute_type::sync;

    static constexpr int iform = IFORM;
private:
//    typedef typename mesh::template EnablePatchFromThis<this_type> patch_base;

    typedef typename mesh_type::id_type id_type;

    typedef typename mesh_type::point_type point_type;

    typedef typename traits::field_value_type<this_type>::type field_value_type;

public:


    //create construct
    Field(mesh_type &m, std::string const &name = "")
            : attribute_type(m.template create_attribute<this_type>(name)) { }

    Field(mesh_type const &m) : attribute_type(m) { }

    //copy construct
    Field(this_type const &other) : attribute_type(other) { }

    // move construct
    Field(this_type &&other) : attribute_type(other) { }

    virtual ~Field() { }


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

private:

    template<typename TOP, typename ...Args>
    void apply(TOP const &op, Args &&... args)
    {
        attribute_type::deploy();
        mesh().template apply<IFORM>(op, *this, std::forward<Args>(args)...);
    }

public:
    /** @name as_function
     *  @{*/

    template<typename ...Args>
    auto gather(Args &&...args) const
    DECL_RET_TYPE((mesh().gather(*this, std::forward<Args>(args)...)))

    template<typename ...Args>
    auto operator()(Args &&...args) const
    DECL_RET_TYPE((mesh().gather(*this, std::forward<Args>(args)...)))


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


template<typename TV, int I, typename TM>
Field<TV, TM, std::integral_constant<int, I> >
make_field(TM const &mesh)
{
    return Field<TV, TM, std::integral_constant<int, I>>(mesh);
};


}// namespace traits

}// namespace simpla

#endif /* FIELD_DENSE_H_ */
