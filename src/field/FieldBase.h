/**
 * @file FieldBase.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_FIELD_FIELDBASE_H
#define SIMPLA_FIELD_FIELDBASE_H


#include <algorithm>
#include <cstdbool>
#include <memory>
#include <string>
#include "FieldTraits.h"

#include "../gtl/type_traits.h"
#include "../parallel/Parallel.h"
#include "../mesh/Mesh.h"


namespace simpla { namespace field
{
typedef typename mesh::id_type id_type;

typedef typename mesh::point_type point_type;

template<typename ...> struct FieldConcept;
template<typename TV, typename TManifold, int IFORM> using Field=
FieldConcept<TV, TManifold, std::integral_constant<int, IFORM> >;

/**
 * @ingroup field
 * @{
 */

template<typename TV, typename TManifold, int IFORM, typename ...Policies>
class FieldConcept<TV, TManiflod, std::integral_constant<int, IFORM>, Policies...>
        : public TManifold::Attribute, public Policies ...
{
private:
    typedef Field<TV, TManifold, std::integral_constant<int, IFORM>, TBase, Policolies...> this_type;

    typedef TManifold::Attribute base_type;

public:

    typedef TManiflod manifold_type;

    typedef TV value_type;

    static constexpr int iform = IFORM;

    typedef typename traits::field_value_type<this_type>::type field_value_type;

    typedef typename this_type::calculus_policy calculus_policy;

    typedef typename this_type::interpolate_policy interpolate_policy;

private:
    manifold_type const *m = nullptr;
    value_type *m_data_ = nullptr;

public:

    //create construct
    template<typename ...Args>
    Field(Args
    &&... args) :

    base_type(std::forward<Args>(args)

    ...) { }

    //copy construct
    Field(this_type const
    &other) :
    base_type(other) {}

    // move construct
    Field(this_type
    &&other) :
    base_type(other) {}

    virtual ~Field() { }

    void swap(this_type &other) { base_type::swap(other); }

    inline this_type &operator=(this_type const &other)
    {
        apply(_impl::_assign(), *this, other);
        return *this;
    }

    virtual int entity_type() const { return iform; }

    virtual int ele_size_in_byte() const { return sizeof(value_type); }

    virtual void deploy()
    {
        m = base_type::mesh();

        m_data_ = reinterpret_cast<value_type *>(base_type::data());
    }

    /**
     * @name assignment
     * @{
     */

    value_type &operator[](id_type const &s) { return m_data_[m->hash<IFORM>(s)]; }

    value_type const &operator[](id_type const &s) const { return m_data_[m->hash<IFORM>(s)]; }


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

        auto r = m->range<IFORM>();

        for (auto const &s:r)
        {
            op(calculus_policy::eval(m, f, s),
               calculus_policy::eval(m, std::forward<Args>(args), s)...);
        }

    }

public:

/** @name as_function
 *  @{*/

    template<typename ...Args>
    field_value_type gather(Args &&...args) const
    {
        return interpolate_policy::gather(m, *this, std::forward<Args>(args)...);
    }


    template<typename ...Args>
    field_value_type operator()(Args &&...args) const
    {
        return interpolate_policy::gather(m, *this, std::forward<Args>(args)...);
    }


    template<typename Other>
    void assign(id_type const &s, Other const &other)
    {
        m_data_[m->template hash<IFORM>(s)] = interpolate_policy::template sample<iform>(m, s, other);
    }

    template<typename Other>
    void add(id_type const &s, Other const &other)
    {
        m_data_[m->template hash<IFORM>(s)] += interpolate_policy::template sample<iform>(m, s, other);
    }

/**@}*/

    template<typename ...Args>
    void accept(Args &&...args) { data()->accept(std::forward<Args>(args)...); }


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


}}//namespace simpla { namespace field
#endif //SIMPLA_FIELD_FIELDBASE_H
