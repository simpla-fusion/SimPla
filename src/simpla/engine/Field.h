/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/Array.h>
#include <simpla/utilities/EntityId.h>
#include <simpla/utilities/ExpressionTemplate.h>
#include <simpla/utilities/FancyStream.h>
#include <simpla/utilities/Range.h>
#include <simpla/utilities/nTuple.h>
#include <simpla/utilities/type_traits.h>
#include "Attribute.h"
#include "Domain.h"
#include "MeshBlock.h"

// namespace std {
//    template <typename TM, typename TV, int IFORM, int... DOF>
//    struct rank<simpla::Field<TM, TV, IFORM, DOF...>> : public std::integral_constant<int, sizeof...(DOF)> {};
//}  // namespace std{

namespace simpla {
template <typename TM, typename TV, int...>
class Field;
}  // namespace simpla

namespace simpla {

namespace traits {
template <typename TM, typename TV, int... I>
struct reference<Field<TM, TV, I...>> {
    typedef const Field<TM, TV, I...>& type;
};

template <typename TM, typename TV, int... I>
struct reference<const Field<TM, TV, I...>> {
    typedef const Field<TM, TV, I...>& type;
};

// template <typename T>
// struct iform : public std::integral_constant<int, VERTEX> {};
//
// template <typename T>
// struct dof : public std::integral_constant<int, 1> {};
//
// template <typename TM, typename TV, int IFORM, int DOF>
// struct iform<Field<TM, TV, IFORM, DOF>> : public std::integral_constant<int, IFORM> {};
//
// template <typename TM, typename TV, int IFORM, int DOF>
// struct dof<Field<TM, TV, IFORM, DOF>> : public std::integral_constant<int, DOF> {};
//
// template <typename TM, typename TV, int IFORM, int DOF>
// struct value_type<Field<TM, TV, IFORM, DOF>> {
//    typedef TV type;
//};

}  // namespace traits {
template <typename TM, typename TV, int IFORM, int... DOF>
class Field<TM, TV, IFORM, DOF...> : public engine::Attribute {
   private:
    typedef Field<TM, TV, IFORM, DOF...> field_type;
    SP_OBJECT_HEAD(field_type, engine::Attribute);

   public:
    typedef TV value_type;
    typedef TM mesh_type;
    static constexpr int iform = IFORM;
    static constexpr int NUM_OF_SUB = (IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3;
    static constexpr int NDIMS = mesh_type::NDIMS;

    typedef std::conditional_t<sizeof...(DOF) == 0, value_type, nTuple<value_type, DOF...>> ele_type;
    typedef std::conditional_t<NUM_OF_SUB == 1, ele_type, nTuple<ele_type, NUM_OF_SUB>> field_value_type;

   private:
    typedef typename mesh_type::template data_type<value_type, IFORM, DOF...> data_type;
    data_type m_data_;
    mesh_type const* m_mesh_ = nullptr;

   public:
    template <typename TGrp, typename... Args>
    explicit Field(TGrp* grp, Args&&... args)
        : engine::Attribute(IFORM, std::integer_sequence<int, DOF...>(), typeid(value_type),
                            dynamic_cast<engine::AttributeGroup*>(grp), std::forward<Args>(args)...) {}

    ~Field() override = default;

    Field(this_type const& other) : engine::Attribute(other), m_data_(other.m_data_), m_mesh_(other.m_mesh_) {}
    Field(this_type&& other)
        : engine::Attribute(std::forward<engine::Attribute>(other)), m_data_(other.m_data_), m_mesh_(other.m_mesh_) {}

    template <typename OtherMesh, typename... Args>
    Field(OtherMesh* other_mesh, Field<OtherMesh, TV, IFORM, DOF...> const& other, Args&&... args)
        : engine::Attribute(IFORM, std::integer_sequence<int, DOF...>(), typeid(value_type),
                            dynamic_cast<engine::AttributeGroup*>(other_mesh), std::forward<Args>(args)...),
          m_data_(other.m_data_),
          m_mesh_(other_mesh) {}

    size_type size() const override {
        return static_cast<size_type>(m_mesh_ == nullptr ? 0 : (m_mesh_->GetNumberOfEntity(IFORM) *
                                                                reduction_v(tags::multiplication(), 1, DOF...)));
    }

    bool empty() const override { return size() == 0; }

    void swap(this_type& other) {
        engine::Attribute::swap(other);
        m_data_.swap(other.m_data_);
        std::swap(m_mesh_, other.m_mesh_);
    }

    auto& Get() { return m_data_; }
    auto const& Get() const { return m_data_; }

    template <typename Other>
    void Set(Other&& v) {
        Update();
        m_mesh_->template Fill<IFORM>(m_data_, std::forward<Other>(v));
    }

    template <typename... Args>
    decltype(auto) Get(Args&&... args) {
        m_mesh_->template GetEntity<IFORM>(m_data_, std::forward<Args>(args)...);
    }
    template <typename... Args>
    decltype(auto) Get(Args&&... args) const {
        m_mesh_->template GetEntity<IFORM>(m_data_, std::forward<Args>(args)...);
    }
    template <typename U, typename... Args>
    void Set(U&& v, Args&&... args) {
        m_mesh_->template SetEntity<IFORM>(m_data_, std::forward<U>(v), std::forward<Args>(args)...);
    }

    template <typename MR, typename UR, int... NR>
    void DeepCopy(Field<MR, UR, NR...> const& other) {
        Update();
        m_data_ = other.Get();
    }
    void Clear() { Set(0); }
    void SetUndefined() { Set(std::numeric_limits<value_type>::signaling_NaN()); }

    this_type& operator=(this_type const& other) {
        m_data_ = other.Get();
        return *this;
    }
    template <typename TR>
    this_type& operator=(TR&& rhs) {
        Set(std::forward<TR>(rhs));
        return *this;
    };

    template <typename... Args>
    decltype(auto) at(Args&&... args) {
        return Get(*this, std::forward<Args>(args)...);
    }
    template <typename... Args>
    decltype(auto) at(Args&&... args) const {
        return Get(std::forward<Args>(args)...);
    }

    template <typename... Args>
    decltype(auto) operator()(Args&&... args) {
        return Get(*this, std::forward<Args>(args)...);
    }
    template <typename... Args>
    decltype(auto) operator()(Args&&... args) const {
        return Get(*this, std::forward<Args>(args)...);
    }

    decltype(auto) operator[](int n) { return m_data_[n]; }
    decltype(auto) operator[](int n) const { return m_data_[n]; }

    decltype(auto) operator[](EntityId s) { return Get(s); }
    decltype(auto) operator[](EntityId s) const { return Get(s); }

    //*****************************************************************************************************************

    template <typename... Args>
    decltype(auto) gather(Args&&... args) const {
        return m_mesh_->gather(*this, std::forward<Args>(args)...);
    }

    template <typename... Args>
    decltype(auto) scatter(Args&&... args) {
        return m_mesh_->scatter(*this, std::forward<Args>(args)...);
    }

    void DoUpdate() override {
        engine::Attribute::DoUpdate();
        if (m_mesh_ == nullptr) { m_mesh_ = dynamic_cast<mesh_type const*>(engine::Attribute::GetMesh()); }
        ASSERT(m_mesh_ != nullptr);

        traits::foreach (m_data_, [&](auto& a, auto i0, auto&&... idx) {
            a.SetSpaceFillingCurve(m_mesh_->GetSpaceFillingCurve(EntityIdCoder::m_sub_index_to_id_[IFORM][i0]));
            a.Update();
        });
    }

    void DoTearDown() override {
        m_mesh_ = nullptr;
        traits::foreach (m_data_, [&](auto& a, auto&&... idx) { a.reset(); });
    }
    void Push(std::shared_ptr<data::DataBlock> p) override {
        Update();
        auto d = std::dynamic_pointer_cast<data::DataMultiArray<value_type, NDIMS>>(p);
        int count = 0;
        traits::foreach (m_data_, [&](auto& a, auto&&... idx) {
            a.swap(d->GetArray(count));
            ++count;
        });
    };
    std::shared_ptr<data::DataBlock> Pop() override {
        auto res = std::make_shared<data::DataMultiArray<value_type, NDIMS>>(NUM_OF_SUB * GetDOF());
        int count = 0;
        traits::foreach (m_data_, [&](auto& a, auto&&... idx) {
            res->GetArray(count).swap(a);
            ++count;
        });
        TearDown();
        return res;
    };

};  // class Field

template <typename TM, typename TL, int... NL>
auto operator<<(Field<TM, TL, NL...> const& lhs, int n) {
    return Expression<tags::bitwise_left_shift, Field<TM, TL, NL...>, int>(lhs, n);
};

template <typename TM, typename TL, int... NL>
auto operator>>(Field<TM, TL, NL...> const& lhs, int n) {
    return Expression<tags::bitwise_right_shifit, Field<TM, TL, NL...>, int>(lhs, n);
};

#define _SP_DEFINE_FIELD_BINARY_FUNCTION(_TAG_, _FUN_)                                        \
    template <typename TM, typename TL, int... NL, typename TR>                               \
    auto _FUN_(Field<TM, TL, NL...> const& lhs, TR const& rhs) {                              \
        return Expression<tags::_TAG_, Field<TM, TL, NL...>, TR>(lhs, rhs);                   \
    };                                                                                        \
    template <typename TL, typename TM, typename TR, int... NR>                               \
    auto _FUN_(TL const& lhs, Field<TM, TR, NR...> const& rhs) {                              \
        return Expression<tags::_TAG_, TL, Field<TM, TR, NR...>>(lhs, rhs);                   \
    };                                                                                        \
    template <typename TM, typename TL, int... NL, typename... TR>                            \
    auto _FUN_(Field<TM, TL, NL...> const& lhs, Expression<TR...> const& rhs) {               \
        return Expression<tags::_TAG_, Field<TM, TL, NL...>, Expression<TR...>>(lhs, rhs);    \
    };                                                                                        \
    template <typename... TL, typename TM, typename TR, int... NR>                            \
    auto _FUN_(Expression<TL...> const& lhs, Field<TM, TR, NR...> const& rhs) {               \
        return Expression<tags::_TAG_, Expression<TL...>, Field<TM, TR, NR...>>(lhs, rhs);    \
    };                                                                                        \
    template <typename ML, typename TL, int... NL, typename MR, typename TR, int... NR>       \
    auto _FUN_(Field<ML, TL, NL...> const& lhs, Field<MR, TR, NR...> const& rhs) {            \
        return Expression<tags::_TAG_, Field<ML, TL, NL...>, Field<MR, TR, NR...>>(lhs, rhs); \
    };

#define _SP_DEFINE_FIELD_UNARY_FUNCTION(_TAG_, _FUN_)              \
    template <typename TM, typename TL, int... NL>                 \
    auto _FUN_(Field<TM, TL, NL...> const& lhs) {                  \
        return Expression<tags::_TAG_, Field<TM, TL, NL...>>(lhs); \
    }

/**
* @defgroup  vector_algebra   Linear algebra on vector fields
* @{
*   Pseudo-Signature  			| Semantics
*  -------------------------------|--------------
*  \f$\Omega^n\f$ =\f$\Omega^n\f$  	            | negate operation
*  \f$\Omega^n\f$ =\f$\Omega^n\f$  	            | positive operation
*  \f$\Omega^n\f$ =\f$\Omega^n\f$ +\f$\Omega^n\f$ 	| add
*  \f$\Omega^n\f$ =\f$\Omega^n\f$ -\f$\Omega^n\f$ 	| subtract
*  \f$\Omega^n\f$ =\f$\Omega^n\f$ *Scalar  	    | multiply
*  \f$\Omega^n\f$ = Scalar * \f$\Omega^n\f$  	    | multiply
*  \f$\Omega^n\f$ = \f$\Omega^n\f$ / Scalar  	    | divide
*
*/

_SP_DEFINE_FIELD_UNARY_FUNCTION(cos, cos)
_SP_DEFINE_FIELD_UNARY_FUNCTION(acos, acos)
_SP_DEFINE_FIELD_UNARY_FUNCTION(cosh, cosh)
_SP_DEFINE_FIELD_UNARY_FUNCTION(sin, sin)
_SP_DEFINE_FIELD_UNARY_FUNCTION(asin, asin)
_SP_DEFINE_FIELD_UNARY_FUNCTION(sinh, sinh)
_SP_DEFINE_FIELD_UNARY_FUNCTION(tan, tan)
_SP_DEFINE_FIELD_UNARY_FUNCTION(tanh, tanh)
_SP_DEFINE_FIELD_UNARY_FUNCTION(atan, atan)
_SP_DEFINE_FIELD_UNARY_FUNCTION(exp, exp)
_SP_DEFINE_FIELD_UNARY_FUNCTION(log, log)
_SP_DEFINE_FIELD_UNARY_FUNCTION(log10, log10)
_SP_DEFINE_FIELD_UNARY_FUNCTION(sqrt, sqrt)
_SP_DEFINE_FIELD_BINARY_FUNCTION(atan2, atan2)
_SP_DEFINE_FIELD_BINARY_FUNCTION(pow, pow)

_SP_DEFINE_FIELD_BINARY_FUNCTION(addition, operator+)
_SP_DEFINE_FIELD_BINARY_FUNCTION(subtraction, operator-)
_SP_DEFINE_FIELD_BINARY_FUNCTION(multiplication, operator*)
_SP_DEFINE_FIELD_BINARY_FUNCTION(division, operator/)
_SP_DEFINE_FIELD_BINARY_FUNCTION(modulo, operator%)
_SP_DEFINE_FIELD_BINARY_FUNCTION(bitwise_xor, operator^)
_SP_DEFINE_FIELD_BINARY_FUNCTION(bitwise_and, operator&)
_SP_DEFINE_FIELD_BINARY_FUNCTION(bitwise_or, operator|)
_SP_DEFINE_FIELD_BINARY_FUNCTION(logical_and, operator&&)
_SP_DEFINE_FIELD_BINARY_FUNCTION(logical_or, operator||)

_SP_DEFINE_FIELD_UNARY_FUNCTION(bitwise_not, operator~)
_SP_DEFINE_FIELD_UNARY_FUNCTION(unary_plus, operator+)
_SP_DEFINE_FIELD_UNARY_FUNCTION(unary_minus, operator-)
_SP_DEFINE_FIELD_UNARY_FUNCTION(logical_not, operator!)

#undef _SP_DEFINE_FIELD_BINARY_FUNCTION
#undef _SP_DEFINE_FIELD_UNARY_FUNCTION
/** @} */

#define _SP_DEFINE_FIELD_COMPOUND_OP(_OP_)                                                            \
    template <typename TM, typename TL, int... NL, typename TR>                                       \
    Field<TM, TL, NL...>& operator _OP_##=(Field<TM, TL, NL...>& lhs, TR const& rhs) {                \
        lhs = lhs _OP_ rhs;                                                                           \
        return lhs;                                                                                   \
    }                                                                                                 \
    template <typename TM, typename TL, int... NL, typename... TR>                                    \
    Field<TM, TL, NL...>& operator _OP_##=(Field<TM, TL, NL...>& lhs, Expression<TR...> const& rhs) { \
        lhs = lhs _OP_ rhs;                                                                           \
        return lhs;                                                                                   \
    }

_SP_DEFINE_FIELD_COMPOUND_OP(+)
_SP_DEFINE_FIELD_COMPOUND_OP(-)
_SP_DEFINE_FIELD_COMPOUND_OP(*)
_SP_DEFINE_FIELD_COMPOUND_OP(/)
_SP_DEFINE_FIELD_COMPOUND_OP(%)
_SP_DEFINE_FIELD_COMPOUND_OP(&)
_SP_DEFINE_FIELD_COMPOUND_OP(|)
_SP_DEFINE_FIELD_COMPOUND_OP (^)
_SP_DEFINE_FIELD_COMPOUND_OP(<<)
_SP_DEFINE_FIELD_COMPOUND_OP(>>)
#undef _SP_DEFINE_FIELD_COMPOUND_OP

#define _SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(_TAG_, _REDUCTION_, _OP_)                                    \
    template <typename TM, typename TL, int... NL, typename TR>                                               \
    bool operator _OP_(Field<TM, TL, NL...> const& lhs, TR const& rhs) {                                      \
        return calculus::reduction<_REDUCTION_>(Expression<tags::_TAG_, Field<TM, TL, NL...>, TR>(lhs, rhs)); \
    };                                                                                                        \
    template <typename TL, typename TM, typename TR, int... NR>                                               \
    bool operator _OP_(TL const& lhs, Field<TM, TR, NR...> const& rhs) {                                      \
        return calculus::reduction<_REDUCTION_>(Expression<tags::_TAG_, TL, Field<TM, TR, NR...>>(lhs, rhs)); \
    };                                                                                                        \
    template <typename TM, typename TL, int... NL, typename... TR>                                            \
    bool operator _OP_(Field<TM, TL, NL...> const& lhs, Expression<TR...> const& rhs) {                       \
        return calculus::reduction<_REDUCTION_>(                                                              \
            Expression<tags::_TAG_, Field<TM, TL, NL...>, Expression<TR...>>(lhs, rhs));                      \
    };                                                                                                        \
    template <typename... TL, typename TM, typename TR, int... NR>                                            \
    bool operator _OP_(Expression<TL...> const& lhs, Field<TM, TR, NR...> const& rhs) {                       \
        return calculus::reduction<_REDUCTION_>(                                                              \
            Expression<tags::_TAG_, Expression<TL...>, Field<TM, TR, NR...>>(lhs, rhs));                      \
    };                                                                                                        \
    template <typename TM, typename TL, int... NL, typename TR, int... NR>                                    \
    bool operator _OP_(Field<TM, TL, NL...> const& lhs, Field<TM, TR, NR...> const& rhs) {                    \
        return calculus::reduction<_REDUCTION_>(                                                              \
            Expression<tags::_TAG_, Field<TM, TL, NL...>, Field<TM, TR, NR...>>(lhs, rhs));                   \
    };

_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(not_equal_to, tags::logical_or, !=)
_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(equal_to, tags::logical_and, ==)
_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(less, tags::logical_and, <)
_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(greater, tags::logical_and, >)
_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(less_equal, tags::logical_and, <=)
_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(greater_equal, tags::logical_and, >=)

#undef _SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR
}  // namespace simpla//namespace algebra
#endif  // SIMPLA_FIELD_H
