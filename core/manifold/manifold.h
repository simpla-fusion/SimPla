/**
 * @file manifold.h
 *
 * @date 2015-2-9
 * @author salmon
 */

#ifndef CORE_MANIFOLD_H_
#define CORE_MANIFOLD_H_

#include <iostream>
#include <memory>
#include "../gtl/macro.h"

#include "calculate/calculate.h"
#include "interpolate/interpolate.h"

namespace simpla
{

template<typename ...> class Manifold;

template<typename ...> struct Domain;
template<typename ...> struct Field;

template<typename TBase, typename ...Policies>
class Manifold<TBase, Policies ...>
        : public TBase, public Policies ...
{

    typedef TBase base_manifold_type;

    typedef Manifold<base_manifold_type, Policies ...> this_type;

public:

    Manifold() : Policies(static_cast<base_manifold_type &>(*this))... { }

    virtual ~Manifold() { }

    Manifold(this_type const &other) : base_manifold_type(other), Policies(other)... { }

    this_type &operator=(const this_type &other)
    {
        this_type(other).swap(*this);
        return *this;
    }


private:

    TEMPLATE_DISPATCH_DEFAULT(load)

    TEMPLATE_DISPATCH_DEFAULT(deploy)

    TEMPLATE_DISPATCH(swap, inline,)

    TEMPLATE_DISPATCH(print, inline, const)

public:
    void swap(const this_type &other) { _dispatch_swap<base_manifold_type, Policies...>(other); }

    template<typename TDict>
    void load(TDict const &dict)
    {
        auto d = dict["Manifold"];
        _dispatch_load<base_manifold_type, Policies...>(d);
    }


    void deploy()
    {
        _dispatch_deploy<base_manifold_type, Policies...>();
    }

    template<typename OS>
    OS &print(OS &os) const
    {
        os << "Manifold={" << std::endl;
        _dispatch_print<base_manifold_type, Policies...>(os);
        os << "}, # Manifold " << std::endl;
        return os;
    }

    template<typename T>
    inline constexpr T access(T const &v, id_t s) const { return v; }

    template<typename T, size_t ...N>
    inline constexpr nTuple<T, N...> const &
    access(nTuple<T, N...> const &v, id_t s) const { return v; }


    template<typename ...T>
    inline traits::primary_type_t<nTuple<Expression<T...>>>
    access(nTuple<Expression<T...>> const &v, id_t s) const
    {
        traits::primary_type_t<nTuple<Expression<T...> > > res;
        res = v;
        return std::move(res);
    }

    template<typename ...Others>
    inline typename traits::value_type<Field<Others...>>::type &
    access(Field<Others...> &f, id_type s) const
    {
        return f[s];
    }


    template<typename ...Others>
    inline typename traits::value_type<Field<Others...>>::type
    access(Field<Others...> const &f, id_type s) const
    {
        return f[s];
    }


    template<typename ...TD>
    inline auto access(Field<Expression<TD...> > const &f, id_type s) const
    DECL_RET_TYPE((this->eval(f, s)))


}; //class Manifold


}//namespace simpla

#endif /* CORE_MANIFOLD_H_ */
