/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include "simpla/SIMPLA_config.h"

#include "simpla/data/Data.h"
#include "simpla/engine/Attribute.h"
#include "simpla/utilities/type_traits.h"

#include "simpla/algebra/ExpressionTemplate.h"

namespace simpla {

template <typename TM, typename TV, int...>
class Field;
template <typename TM, typename TV, int IFORM, int... DOF>
class Field<TM, TV, IFORM, DOF...> : public engine::AttributeT<TV, IFORM, DOF...> {
   private:
    typedef Field<TM, TV, IFORM, DOF...> this_type;
    typedef engine::AttributeT<TV, IFORM, DOF...> base_type;

   public:
    typedef TV value_type;
    typedef TM mesh_type;
    using typename base_type::data_type;
    typedef engine::AttributeT<TV, IFORM, DOF...> attribute_type;
    mesh_type* m_mesh_;

   public:
    template <typename... Args>
    explicit Field(mesh_type* host, Args&&... args) : base_type(host, std::forward<Args>(args)...), m_mesh_(host) {}
    ~Field() override = default;

    Field(this_type const& other) = delete;  // : base_type(dynamic_cast<base_type const&>(other)){};
    Field(this_type&& other) = delete;       // : base_type(dynamic_cast<base_type&&>(other)){};

    //    Field(this_type const& other, IdxShift const& s) : this_type(other){};

    //    this_type GetSelection(Range<EntityId> const& r) const { return this_type(*this); }

    template <typename... Args>
    static std::shared_ptr<this_type> New(Args&&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    }
    using base_type::Update;
    using base_type::SetUp;
    using base_type::TearDown;

    void DoSetUp() override;
    void DoUpdate() override;
    void DoTearDown() override;
    template <typename Other>
    void Assign(Other&& v) {
        Update();
        m_mesh_->Fill(*this, std::forward<Other>(v));
    }

    this_type& operator=(this_type const& other) {
        Assign(other);
        return *this;
    }
    template <typename TR>
    this_type& operator=(TR&& rhs) {
        Assign(std::forward<TR>(rhs));
        return *this;
    };

    //    auto& operator[](EntityId s) {
    //        return traits::invoke(
    //            traits::index(*dynamic_cast<data_type*>(this), EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]), s.x,
    //            s.y,
    //            s.z);
    //    }
    //    auto const& operator[](EntityId s) const {
    //        return traits::invoke(
    //            traits::index(*dynamic_cast<data_type*>(this), EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]), s.x,
    //            s.y,
    //            s.z);
    //    }
    //    auto operator()(IdxShift S) const { return this_type(*this, S); }
    //*****************************************************************************************************************

    template <typename... Args>
    decltype(auto) gather(Args&&... args) const {
        return m_mesh_->gather(*this, std::forward<Args>(args)...);
    }

    template <typename... Args>
    decltype(auto) scatter(Args&&... args) {
        return m_mesh_->scatter(*this, std::forward<Args>(args)...);
    }

};  // class Field

namespace traits {

template <typename TM, typename TV, int... I>
struct reference<Field<TM, TV, I...>> {
    typedef const engine::AttributeT<TV, I...>& type;
};

template <typename TM, typename TV, int... I>
struct reference<const Field<TM, TV, I...>> {
    typedef const engine::AttributeT<TV, I...>& type;
};

template <typename TM, typename TV, int IFORM, int... DOF>
struct iform<Field<TM, TV, IFORM, DOF...>> : public std::integral_constant<int, IFORM> {};

template <typename TM, typename TV, int IFORM, int... DOF>
struct dof<Field<TM, TV, IFORM, DOF...>>
    : public std::integral_constant<int, reduction_v(tags::multiplication(), 1, DOF...)> {};
}

template <typename TM, typename TV, int IFORM, int... DOF>
void Field<TM, TV, IFORM, DOF...>::DoSetUp() {
    base_type::DoSetUp();
}
template <typename TM, typename TV, int IFORM, int... DOF>
void Field<TM, TV, IFORM, DOF...>::DoUpdate() {
    m_mesh_->InitializeAttribute(this);
    base_type::DoUpdate();
}
template <typename TM, typename TV, int IFORM, int... DOF>
void Field<TM, TV, IFORM, DOF...>::DoTearDown() {
    base_type::DoTearDown();
}
}  // namespace simpla

#endif  // SIMPLA_FIELD_H
