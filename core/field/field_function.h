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

#include "../gtl/primitives.h"

namespace simpla
{
template<typename ...> class Field;


template<int IFORM, typename TM, typename TV, typename TFun>
class Field<TV, TM, std::integral_constant<int, IFORM>, tags::function, TFun>
{
public:


    typedef TV value_type;

    static constexpr int iform = IFORM;

    static constexpr int ndims = TM::ndims;

private:

    typedef TM mesh_type;

    typedef Field<value_type, mesh_type, std::integral_constant<int, IFORM>, tags::function, TFun> this_type;

    typedef typename traits::field_value_type<this_type>::type field_value_type;

    mesh_type const &m_mesh_;

    TFun m_fun_;


public:

    template<typename TF>
    Field(mesh_type const &m, TF const &fun) :
            m_mesh_(m), m_fun_(fun)
    {
    }


    Field(this_type const &other) :
            m_mesh_(other.m_mesh_), m_fun_(other.m_fun_)
    {
    }

    ~Field()
    {
    }

    bool is_valid() const
    {
        return (!!m_fun_);
    }

    operator bool() const
    {
        return !!m_fun_;
    }


    template<typename TID>
    value_type operator[](TID const &s) const
    {
        return m_mesh_.template sample<iform>(s, this->operator()(m_mesh_.point(s), m_mesh_.time()));
    }

    template<typename ...Args>
    field_value_type operator()(Args &&...args) const
    {
        return static_cast<field_value_type>(m_fun_(std::forward<Args>(args)...));
    }


};
namespace traits
{

template<typename TV, int IFORM, typename TM, typename TFun>
Field<TV, TM, std::integral_constant<int, IFORM>, tags::function, TFun> //
make_field_function(TM const &m, TFun const &fun)
{
    return Field<TV, TM, std::integral_constant<int, IFORM>, tags::function, TFun>(m, fun);
}


template<typename TV, int IFORM, typename TM, typename TDict>
Field<TV, TM, std::integral_constant<int, IFORM>, tags::function, TDict> //
make_function_by_config(TM const &m, TDict const &dict)
{
    typedef TV value_type;

    typedef Field<TV, TM, std::integral_constant<int, IFORM>, tags::function, TDict> field_type;

    if (!dict["Value"]) { THROW_EXCEPTION("illegal configure file!"); }

    field_type res(m, dict["Value"]);

//    if (dict["Domain"]) { res.domain().filter(dict["Domain"]); }

    return std::move(res);
}
}//namespace traits
}
// namespace simpla

#endif /* COREFieldField_FUNCTION_H_ */
