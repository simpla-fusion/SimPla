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

template<typename ...> class Domain;

template<typename ...TDomain, typename TV, typename TFun>
class Field<Domain<TDomain...>, TV, tags::function, TFun>
{
public:

    typedef Domain<TDomain...> domain_type;

    typedef TV value_type;

    static constexpr int iform = domain_type::iform;

    static constexpr int ndims = domain_type::ndims;

private:

    typedef Field<domain_type, value_type, tags::function, TFun> this_type;

    typedef typename domain_type::mesh_type mesh_type;

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
    value_type operator[](TID s) const
    {
        return m_mesh_.template sample<iform>(s, this->operator()(m_mesh_.point(s), m_mesh_.time()));
    }

    template<typename ...Args>
    field_value_type operator()(Args &&...args) const
    {
        return static_cast<field_value_type>(m_fun_(std::forward<Args>(args)...));
    }

//    template<typename ...Others>
//    field_value_type operator()(Others &&... others) const
//    {
//        return static_cast<field_value_type>(m_fun_(std::forward<Others>(others)...));
//    }

//    /**
//     *
//     * @param args
//     * @return (x,t) -> m_fun_(x,t,args(x,t))
//     */
//    template<typename ...Args>
//    Field<domain_type, value_type, tags::function,
//            std::function<field_value_type(point_type const &, Real)>> op_on(
//            Args const &...args) const
//    {
//        typedef std::function<field_value_type(point_type const &, Real)> res_function_type;
//
//        typedef Field<domain_type, value_type, tags::function, res_function_type> res_type;
//
//        res_function_type fun = [&](point_type const &x, Real t)
//        {
//            return static_cast<field_value_type>(m_fun_(x, t, static_cast<field_value_type>((args)(x))...));
//        };
//
//        return res_type(m_domain_, fun);
//
//    }

};
namespace traits
{
//template<typename TV, typename TDomain, typename TFun>
//Field<TDomain, TV, tags::function, TFun> make_field_function(
//        TDomain const &domain, TFun const &fun)
//{
//    return std::move(Field<TDomain, TV, tags::function, TFun>(domain, fun));
//}

template<int IFORM, typename TV, typename TM, typename TFun>
Field<Domain<TM, std::integral_constant<int, IFORM> >, TV, tags::function, TFun> //
make_field_function(TM const &m, TFun const &fun)
{
    return Field<Domain<TM, std::integral_constant<int, IFORM> >, TV, tags::function, TFun>(m, fun);
}


template<int IFORM, typename TV, typename TM, typename TDict>
Field<Domain<TM, std::integral_constant<int, IFORM> >, TV, tags::function, TDict> //
make_function_by_config(TM const &m, TDict const &dict)
{
    typedef TV value_type;

    typedef Field<Domain<TM, std::integral_constant<int, IFORM> >, TV, tags::function, TDict> field_type;

    if (!dict["Value"]) {ERROR("illegal configure file!"); }

    field_type res(m, dict["Value"]);

//    if (dict["Domain"]) { res.domain().filter(dict["Domain"]); }

    return std::move(res);
}
}//namespace traits
}
// namespace simpla

#endif /* COREFieldField_FUNCTION_H_ */
