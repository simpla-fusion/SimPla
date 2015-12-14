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

#include "../gtl/primitives.h"
#include "../geometry/geometry.h"

namespace simpla
{
template<typename ...> class Field;


template<int IFORM, typename TM, typename TV, typename TFun, typename TBox>
class Field<TV, TM, std::integral_constant<int, IFORM>, tags::function, TFun, TBox>
{
public:
    typedef TV value_type;

    static constexpr int iform = IFORM;

    static constexpr int ndims = TM::ndims;

private:

    typedef TM mesh_type;

    typedef typename TM::box_type spatial_domain_type;
    typedef typename TM::id_type id_type;
    typedef typename TM::point_type point_type;
    typedef Field<TV, TM, std::integral_constant<int, IFORM>, tags::function, TFun, TBox> this_type;

    typedef typename traits::field_value_type<this_type>::type field_value_type;

    mesh_type const &m_mesh_;

    TFun m_fun_;

    spatial_domain_type m_define_domain_;

    std::mutex m_mutex_;

public:

    template<typename TF>
    Field(mesh_type const &m, TF const &fun, TBox const &box) :
            m_mesh_(m), m_fun_(fun), m_define_domain_(box) { }


    Field(this_type const &other) :
            m_mesh_(other.m_mesh_), m_fun_(other.m_fun_), m_define_domain_(other.m_define_domain_) { }

    ~Field() { }

    bool is_valid() const { return (!!m_fun_); }

    operator bool() const { return !!m_fun_; }

    value_type at(id_type const &s) const
    {

        field_value_type v = this->operator()(m_mesh_.point(s));

        return m_mesh_.template sample<IFORM>(s, v);
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

template<typename TV, typename TM, int IFORM, typename TFun>
using FieldFunction=Field<TV, TM, std::integral_constant<int, IFORM>,
        tags::function, TFun, typename TM::box_type>;

namespace traits
{

template<typename TV, int IFORM, typename TM, typename TFun>
FieldFunction<TV, TM, IFORM, TFun> //
make_field_function(TM const &m, TFun const &fun)
{
    return FieldFunction<TV, TM, IFORM, TFun>(m, fun, m.box());
}


template<typename TV, int IFORM, typename TM, typename TFun>
FieldFunction<TV, TM, IFORM, TFun> //
make_field_function_from_config(TM const &m, TFun const &dict)
{
    typedef TV value_type;

    typedef FieldFunction<TV, TM, IFORM, TFun> field_type;

    if (!(dict["Value"])) { THROW_EXCEPTION_RUNTIME_ERROR("illegal configure file!"); }

    typename TM::box_type b;
    if (dict["Box"])
    {
        b = dict["Box"].template as<typename TM::box_type>();
    } else
    {
        b = m.box();
    }

    return field_type(m, dict["Value"], b);

}
} // namespace traits
} // namespace simpla

#endif /* COREFieldField_FUNCTION_H_ */
