/**
 * @file field_function.h
 *
 *  Created on: 2015-3-10
 *      Author: salmon
 */

#ifndef COREFieldField_FUNCTION_H_
#define COREFieldField_FUNCTION_H_

#include <stddef.h>
#include <cstdbool>
#include <functional>
#include <mutex>

#include "simpla/sp_def.h"
#include "../modlel/geometry.h"

namespace simpla
{
template<typename ...> class Field;

namespace tags { struct function; }

template<typename TV, typename TMesh, int IFORM, typename TFun, typename TBox, typename ...Policies>
class Field<TV, TMesh, std::integral_constant<int, IFORM>, tags::function, TFun, TBox, Policies...>
        : public Policies ...
{
public:
    typedef TV value_type;

    typedef TMesh manifold_type;

    static constexpr int iform = IFORM;

    static constexpr int ndims = manifold_type::ndims;

private:


    typedef typename manifold_type::box_type spatial_domain_type;
    typedef typename manifold_type::id_type id_type;
    typedef typename manifold_type::point_type point_type;
    typedef Field<TV, manifold_type, std::integral_constant<int, IFORM>, tags::function, TFun, TBox, Policies...> this_type;
    typedef typename this_type::calculus_policy calculus_policy;
    typedef typename this_type::interpolate_policy interpolate_policy;
    typedef typename traits::field_value_type<this_type>::type field_value_type;

    manifold_type const &m_mesh_;

    TFun m_fun_;

    spatial_domain_type m_define_domain_;

    std::mutex m_mutex_;

public:

    template<typename TF>
    Field(manifold_type const &m, TF const &fun, TBox const &box) :
            m_mesh_(m), m_fun_(fun), m_define_domain_(box) { }


    Field(this_type const &other) :
            m_mesh_(other.m_mesh_), m_fun_(other.m_fun_), m_define_domain_(other.m_define_domain_) { }

    ~Field() { }


    template<typename TF>
    static this_type create(manifold_type const &m, TFun const &fun)
    {
        return this_type(m, fun, m.box());
    }

    template<typename TDict>
    static this_type create_from_config(manifold_type const &m, TDict const &dict)
    {

        if (!(dict["Value"])) { THROW_EXCEPTION_RUNTIME_ERROR("illegal configure file!"); }

        typename manifold_type::box_type b;
        if (dict["MeshBlock"])
        {
            b = dict["MeshBlock"].template as<typename manifold_type::box_type>();
        } else
        {
            b = m.box();
        }

        return this_type(m, dict["Value"], b);


    }

    bool is_valid() const { return (!!m_fun_); }

    operator bool() const { return !!m_fun_; }

    value_type at(id_type const &s) const
    {

        field_value_type v = this->operator()(m_mesh_.point(s));

        return interpolate_policy::template sample<iform>(m_mesh_, s, v);
    }


    value_type operator[](id_type const &s) const
    {
        return this->at(s);
    }


    field_value_type operator()(point_type const &x) const
    {
        field_value_type res;

        if (geometry::traits::in_set(x, m_define_domain_))
        {
            const_cast<this_type *>(this)->m_mutex_.lock();

            res = static_cast<field_value_type>(m_fun_(m_mesh_.time(), x));

            const_cast<this_type *>(this)->m_mutex_.unlock();
        }
        else { res = 0; }


        return res;

    }

};

} // namespace simpla

#endif /* COREFieldField_FUNCTION_H_ */
