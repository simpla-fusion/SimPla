/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include <simpla/SIMPLA_config.h>
#include "Algebra.h"
#include "nTuple.h"

namespace simpla {
namespace algebra {
namespace declare {
template <typename, typename, size_type...>
struct Field_;
}

namespace traits {

//***********************************************************************************************************************

template <typename>
struct mesh_type {
    typedef void type;
};

template <typename TV, typename TM, size_type... I>
struct mesh_type<declare::Field_<TV, TM, I...>> {
    typedef TM type;
};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct is_field<declare::Field_<TV, TM, IFORM, DOF>> : public std::integral_constant<bool, true> {};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct reference<declare::Field_<TV, TM, IFORM, DOF>> {
    typedef declare::Field_<TV, TM, IFORM, DOF> const& type;
};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct reference<const declare::Field_<TV, TM, IFORM, DOF>> {
    typedef declare::Field_<TV, TM, IFORM, DOF> const& type;
};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct value_type<declare::Field_<TV, TM, IFORM, DOF>> {
    typedef TV type;
};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct rank<declare::Field_<TV, TM, IFORM, DOF>> : public index_const<3> {};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct iform<declare::Field_<TV, TM, IFORM, DOF>> : public index_const<IFORM> {};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct dof<declare::Field_<TV, TM, IFORM, DOF>> : public index_const<DOF> {};

template <typename TV, typename TM, size_type IFORM, size_type DOF>
struct field_value_type<declare::Field_<TV, TM, IFORM, DOF>> {
    typedef std::conditional_t<DOF == 1, TV, declare::nTuple_<TV, DOF>> cell_tuple;
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME), cell_tuple,
                               declare::nTuple_<cell_tuple, 3>>
        type;
};

}  // namespace traits{

namespace declare {
template <typename TV, typename TM, size_type IFORM, size_type DOF>
class Field_<TV, TM, IFORM, DOF> : public TM::attribute<TV, IFORM, DOF> {
   private:
    typedef Field_<TV, TM, IFORM, DOF> this_type;
    typedef TM::attribute<TV, IFORM, DOF> base_type;

   public:
    typedef traits::field_value_t<this_type> field_value;
    typedef TV value_type;
    typedef TM mesh_type;

   private:
    typedef typename mesh_type::id_type mesh_id_type;

    typedef calculus::calculator<this_type> calculus_policy;
    friend calculus_policy;

    typedef typename base_type::data_pointer data_pointer;

    data_pointer m_data_;

    mesh_type const* m_mesh_;

   public:
    Field_() : m_mesh_(nullptr), m_data_(nullptr){};

    template <typename... Args>
    explicit Field_(Args&&... args)
        : base_type(std::forward<Args>(args)...), m_mesh_(nullptr), m_data_(nullptr) {}

    virtual ~Field_() {}

    Field_(this_type const& other) = delete;

    Field_(this_type&& other) = delete;

    inline this_type& operator=(this_type const& rhs) { return assign(rhs); }

    template <typename TR>
    inline this_type& operator=(TR const& rhs) {
        return assign(rhs);
    }

    virtual void deploy() {
        base_type::deploy();
        m_data_ = base_type::template data_as<data_type>();
        m_mesh_ = base_type::template mesh_as<mesh_type>();
    }

    /** @name as_function  @{*/
    template <typename... Args>
    decltype(auto) gather(Args&&... args) const {
        return apply(tags::_gather(), std::forward<Args>(args)...);
    }

    template <typename... Args>
    decltype(auto) scatter(field_value const& v, Args&&... args) {
        return apply(tags::_scatter(), v, std::forward<Args>(args)...);
    }

    /**@}*/

    template <typename... TID>
    decltype(auto) at(TID&&... s) {
        return m_data_[m_mesh_->hash(IFORM, DOF, std::forward<TID>(s)...)];
    }
    template <typename... TID>
    decltype(auto) at(TID&&... s) const {
        return m_data_[m_mesh_->hash(IFORM, DOF, std::forward<TID>(s)...)];
    }
    template <typename TID>
    decltype(auto) operator[](TID const& s) {
        return at(s);
    }

    template <typename TID>
    decltype(auto) operator[](TID const& s) const {
        return at(s);
    }

    template <typename... TID>
    decltype(auto) get(TID&&... s) {
        return calculus_policy::get_value(*m_mesh_, *this, std::forward<TID>(s)...);
    }
    template <typename... TID>
    decltype(auto) get(TID&&... s) const {
        return calculus_policy::get_value(*m_mesh_, *this, std::forward<TID>(s)...);
    }
    template <typename... Args>
    decltype(auto) operator()(Args&&... args) {
        return get(std::forward<Args>(args)...);
    }

    template <typename... Args>
    decltype(auto) operator()(Args&&... args) const {
        return get(std::forward<Args>(args)...);
    }

    template <typename... Args>
    this_type& assign(Args&&... args) {
        return apply(tags::_assign(), std::forward<Args>(args)...);
    }

    template <typename... Args>
    this_type& apply(Args&&... args) {
        deploy();
        calculus_policy::apply(*m_mesh_, *this, std::forward<Args>(args)...);
        return *this;
    }

};  // class Field_
}  // namespace declare
}  // namespace algebra
}  // namespace simpla

namespace simpla {
template <typename TV, typename TM, size_type IFORM = VERTEX, size_type DOF = 1>
using Field = algebra::declare::Field_<TV, TM, IFORM, DOF>;
}

#endif  // SIMPLA_FIELD_H
