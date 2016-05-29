/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H


#include "../mesh/MeshAttribute.h"
#include "FieldTraits.h"
#include <type_traits>

namespace simpla
{


template<typename ...> class Field;

template<typename TV, typename TManifold, size_t IFORM>
using field_t= Field<TV, TManifold, index_const<IFORM>>;


/**
 * @ingroup field
 * @{
 */


template<typename TV, typename TManifold, size_t IFORM>
class Field<TV, TManifold, index_const<IFORM>>
        : public mesh::MeshAttribute<TV, TManifold, index_const<IFORM>, mesh::tags::DENSE>::View
{
private:
    static_assert(std::is_base_of<mesh::MeshBase, TManifold>::value, "TManifold is not derived from MeshBase");
    typedef Field<TV, TManifold, index_const<IFORM>> this_type;
public:

    virtual bool is_a(std::type_info const &info) const { return typeid(this_type) == info || base_type::is_a(info); }

    virtual std::string get_class_name() const { return class_name(); }

    static std::string class_name()
    {
        return std::string("Field<") +
               traits::type_id<value_type, mesh_type, index_const<IFORM>>::name() + ">";
    }

    typedef mesh::MeshAttribute<TV, TManifold, index_const<IFORM>, mesh::tags::DENSE> attribute_type;

private:
    typedef typename attribute_type::View base_type;
    std::shared_ptr<attribute_type> m_attr_;
public:

    typedef TManifold mesh_type;

    typedef TV value_type;


    static constexpr mesh::MeshEntityType iform = static_cast<mesh::MeshEntityType>(IFORM);

    typedef typename traits::field_value_type<this_type>::type field_value_type;


public:

    //create construct

    Field(std::shared_ptr<attribute_type> attr, std::string const & = "") : m_attr_(attr)
    {
        view(attr->mesh_atlas().root());
    }

    Field(mesh::MeshAtlas const &m, std::string const & = "") : m_attr_(std::make_shared<attribute_type>(m))
    {
        view(m.root());
    }

    Field(mesh_type const &m, std::string const & = "") : m_attr_()
    {
        view(m.root());
        //FIXME
        WARNING << "This function is not completed!" << std::endl;
    }

    //copy construct
    Field(this_type const &other) : base_type(other), m_attr_(other.m_attr_) { }


    // move construct
    Field(this_type &&other) : base_type(other), m_attr_(other.m_attr_) { }

    virtual ~Field() { }

    std::shared_ptr<attribute_type> attribute() { return m_attr_; }

    std::shared_ptr<attribute_type> const attribute() const { return m_attr_; }

    void view(mesh::MeshBlockId const &id) { m_attr_->view(id).swap(*this); }

    void clear() { }

    void sync() { }

    void deploy() { }

    template<typename TFun> void accept(mesh::MeshEntityRange const &, TFun const) { }

    template<typename TFun> void accept(mesh::MeshEntityRange const &, TFun const) const { }

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

    template<typename TOP, typename Other>
    void apply(TOP const &op, this_type &f, Other const &other)
    {

        //TODO: need parallelism
        assert(!base_type::range().empty());

        for (auto const &s: base_type::range())
        {

            value_type tmp;
//            op(/*f[s]*/tmp, base_type::mesh().eval(other, s));
        }
        UNIMPLEMENTED;
    }

public:

/** @name as_function
 *  @{*/

    template<typename ...Args>
    field_value_type gather(Args &&...args) const
    {
        return mesh_type::interpolate_policy::gather(base_type::mesh(), *this, std::forward<Args>(args)...);
    }


    template<typename ...Args>
    field_value_type operator()(Args &&...args) const { return gather(std::forward<Args>(args)...); }


//    template<typename Other>
//    void assign(id_type const &s, Other const &other)
//    {
//        port_type::get(s) = interpolate_policy::template sample<iform>(mesh(), s, other);
//    }
//
//    template<typename Other>
//    void add(id_type const &s, Other const &other)
//    {
//        port_type::get(s) += interpolate_policy::template sample<iform>(mesh(), s, other);
//    }

/**@}*/

//    template<typename ...Args>
//    void accept(Args &&...args) { port_type::accept(std::forward<Args>(args)...); }


};
//
//namespace traits
//{
//template<typename> struct is_field { static constexpr bool value = false; };
//template<typename ...T> struct is_field<Field<T...>> { static constexpr bool value = true; };
//
//}
}//namespace simpla

#endif //SIMPLA_FIELD_H
