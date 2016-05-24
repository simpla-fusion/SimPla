/**
 * @file FieldBase.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_FIELD_FIELDBASE_H
#define SIMPLA_FIELD_FIELDBASE_H
//#include "FieldTraits.h"
//#include "../gtl/type_traits.h"
#include "../mesh/MeshAttribute.h"
#include <type_traits>

namespace simpla
{
using mesh::id_type;

using mesh::point_type;

template<typename ...> class Field;

template<typename TV, typename TManifold, unsigned int IFORM>
using FieldAttr= Field<TV, TManifold, std::integral_constant<unsigned int, IFORM>>;


/**
 * @ingroup field
 * @{
 */


template<typename TV, typename TManifold, unsigned int IFORM>
class Field<TV, TManifold, std::integral_constant<int, IFORM>>
        : public mesh::MeshAttribute<TV, TManifold, std::integral_constant<int, IFORM>>::View
{
private:
//    static_assert(std::is_base_of<mesh::MeshBase, TManifold>::value);


    typedef Field<TV, TManifold, std::integral_constant<int, IFORM>> this_type;

    typedef mesh::MeshAttribute<TV, TManifold, std::integral_constant<int, IFORM>> host_type;

    typedef typename host_type::View base_type;

    std::shared_ptr<host_type> m_host_;

public:

    typedef TManifold manifold_type;

    typedef TV value_type;

    static constexpr unsigned int iform = IFORM;

//    typedef typename traits::field_value_type<this_type>::type field_value_type;
//
//    typedef typename manifold_type::calculus_policy calculus_policy;
//
//    typedef typename manifold_type::interpolate_policy interpolate_policy;


public:

    //create construct
    template<typename ...Args>
    Field(Args &&... args) : m_host_(new host_type(std::forward<Args>(args) ...))
    {
        view();
    }

    //copy construct
    Field(this_type const &other) : base_type(other) { }

    // move construct
    Field(this_type &&other) : base_type(other) { }

    virtual ~Field() { }


    void view(mesh::MeshBlockId m_id = 0) { m_host_->view().swap(*this); }


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

//        for (auto const &s: base_type::range())
//        {
//            op(calculus_policy::eval(base_type::mesh(), f, s),
//               calculus_policy::eval(base_type::mesh(), std::forward<Args>(args), s)...);
//        }

    }

public:

/** @name as_function
 *  @{*/

//    template<typename ...Args>
//    field_value_type gather(Args &&...args) const
//    {
//        return interpolate_policy::gather(base_type::mesh(), *this, std::forward<Args>(args)...);
//    }
//
//
//    template<typename ...Args>
//    field_value_type operator()(Args &&...args) const
//    {
//        return interpolate_policy::gather(base_type::mesh(), *this, std::forward<Args>(args)...);
//    }


//    template<typename Other>
//    void assign(id_type const &s, Other const &other)
//    {
//        base_type::get(s) = interpolate_policy::template sample<iform>(mesh(), s, other);
//    }
//
//    template<typename Other>
//    void add(id_type const &s, Other const &other)
//    {
//        base_type::get(s) += interpolate_policy::template sample<iform>(mesh(), s, other);
//    }

/**@}*/

    template<typename ...Args>
    void accept(Args &&...args) { base_type::accept(std::forward<Args>(args)...); }


}; // struct field
//
//namespace traits
//{
//
//
//template<typename TV, typename TM, typename ...Others> struct mesh_type<Field<TV, TM, Others...> > { typedef TM type; };
//
//template<typename TV, typename ...Policies>
//struct value_type<Field<TV, Policies...>> { typedef TV type; };
//
//template<typename TV, typename TM, typename TFORM, typename ...Others>
//struct iform<Field<TV, TM, TFORM, Others...> > : public TFORM { };
//
//
//}// namespace traits


}//namespace simpla
#endif //SIMPLA_FIELD_FIELDBASE_H
