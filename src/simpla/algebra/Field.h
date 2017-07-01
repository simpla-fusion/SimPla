/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include <simpla/SIMPLA_config.h>
#include <simpla/engine/Attribute.h>
#include <simpla/engine/Domain.h>
#include <simpla/engine/MeshBlock.h>
#include <simpla/utilities/Array.h>
#include <simpla/utilities/ExpressionTemplate.h>
#include <simpla/utilities/FancyStream.h>
#include <simpla/utilities/Range.h>
#include <simpla/utilities/nTuple.h>
#include "Algebra.h"
#include "CalculusPolicy.h"
namespace simpla {

template <typename>
class calculator;
template <typename TM, typename TV, int...>
class Field;
template <typename TM, typename TV, int IFORM, int... DOF>
class Field<TM, TV, IFORM, DOF...> : public engine::Attribute {  //
   private:
    typedef Field<TM, TV, IFORM, DOF...> field_type;
    SP_OBJECT_HEAD(field_type, engine::Attribute);

   public:
    typedef TV value_type;
    typedef TM mesh_type;
    static constexpr int iform = IFORM;
    static constexpr int NUM_OF_SUB = (IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3;
    static constexpr int NDIMS = mesh_type::NDIMS;

    typedef std::conditional_t<sizeof...(DOF) == 0, value_type, nTuple<value_type, DOF...>> cell_tuple;
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME), cell_tuple, nTuple<cell_tuple, 3>>
        field_value_type;

   private:
    typedef typename mesh_type::template array_type<value_type> array_type;
    typedef typename mesh_type::entity_id_type entity_id_type;
    nTuple<array_type, NUM_OF_SUB, DOF...> m_data_;
    Range<entity_id_type> m_range_;
    mesh_type const* m_mesh_ = nullptr;

   public:
    template <typename... Args>
    explicit Field(Args&&... args)
        : base_type(IFORM, std::integer_sequence<int, DOF...>(), typeid(value_type), std::forward<Args>(args)...) {}

    Field(this_type const& other)
        : base_type(other), m_mesh_(other.m_mesh_), m_range_(other.m_range_), m_data_(other.m_data_) {}

    Field(this_type&& other)
        : base_type(other), m_mesh_(other.m_mesh_), m_range_(other.m_range_), m_data_(other.m_data_) {}

    Field(this_type const& other, EntityRange r)
        : base_type(other), m_data_(other.m_data_), m_mesh_(other.m_mesh_), m_range_(r) {}

    ~Field() override = default;

    size_type size() const override { return m_range_.size(); }

    bool empty() const override { return m_range_.empty(); }

    void Clear() {
        DoUpdate();
        m_data_ = 0;
    }
    void Fill(value_type v) {
        DoUpdate();
        m_data_ = v;
    }
    void SetUndefined() {
        DoUpdate();
        m_data_ = std::numeric_limits<value_type>::signaling_NaN();
    }

    auto& data() { return m_data_; }
    auto const& data() const { return m_data_; }

    this_type& operator=(this_type const& other) {
        Assign(other);
        return *this;
    }

    void swap(this_type& other) {
        engine::Attribute::swap(other);
        std::swap(m_mesh_, other.m_mesh_);
        m_data_.swap(other.m_data_);
        m_range_.swap(other.m_range_);
    }

    template <typename TR>
    this_type& operator=(TR const& rhs) {
        Assign(rhs);
        return *this;
    };
    auto& operator[](int n) { return m_data_[n]; }
    auto const& operator[](int n) const { return m_data_[n]; }

    template <typename... Args>
    auto const& at(Args&&... args) const {
        return m_data_(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto& at(Args&&... args) {
        return m_data_(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto const& operator()(Args&&... args) const {
        return at(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto& operator()(Args&&... args) {
        return at(std::forward<Args>(args)...);
    }
    auto const& at(entity_id_type s) const { return at(s.w, s.x, s.y, s.z); }
    auto& at(entity_id_type s) { return at(s.w, s.x, s.y, s.z); }

    auto& operator[](entity_id_type s) { return at(s); }
    auto const& operator[](entity_id_type s) const { return at(s); }

    this_type operator[](Range<entity_id_type> const& d) const { return this_type(*this, d); }
    this_type operator()(Range<entity_id_type> const& d) const { return this_type(*this, d); }

    //*****************************************************************************************************************
    typedef calculator<mesh_type> calculus_policy;

    template <typename... Args>
    auto gather(Args&&... args) const {
        return calculus_policy::gather(*m_mesh_, *this, std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto scatter(Args&&... args) {
        return calculus_policy::scatter(*m_mesh_, *this, std::forward<Args>(args)...);
    }

    void Update() override {
        engine::Attribute::Update();
        if (m_mesh_ == nullptr) { m_mesh_ = dynamic_cast<mesh_type const*>(engine::Attribute::GetMesh()); }
        ASSERT(m_mesh_ != nullptr);
        m_data_.foreach ([&](int const* idx, array_type& a) {
            if (a.empty()) { a.SetSpaceFillingCurve(m_mesh_->GetSpaceFillingCurve(IFORM, idx[0])); }
        });
    }

    void TearDown() override {
        m_range_.reset();
        m_data_.foreach ([&](array_type& a) { a.reset(); });
        m_mesh_ = nullptr;
    }

    void Unpack(const std::shared_ptr<data::DataBlock>& d, const EntityRange& r) override {
        if (d != nullptr) {
            m_range_ = r;

            auto multi_array = std::dynamic_pointer_cast<data::DataMultiArray<value_type, NDIMS>>(d);
            int count = 0;
            m_data_.foreach ([&](array_type& a) { (*multi_array)[count].swap(a); });
            Click();
        }
        DoUpdate();
    }

    std::shared_ptr<data::DataBlock> Pack() override {
        std::shared_ptr<data::DataBlock> res;
        m_data_.foreach ([&](int const* idx, array_type& a) {});
        DoTearDown();
        return res;
    }
    template <typename TOther>
    void DeepCopy(TOther const& other) {
        DoUpdate();
        m_data_ = other.data();
    }

    template <typename Other>
    void Assign(Other const& other) {
        DoUpdate();
        ASSERT(m_mesh_ != nullptr);
        m_mesh_->Assign(*this, m_range_, other);
        m_mesh_->Assign(*this, m_mesh_->GetRange(std::string(EntityIFORMName[IFORM]) + "_PATCH_BOUNDARY"), 0);
    }

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

#define _SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(_TAG_, _REDUCTION_, _OP_)                                  \
    template <typename TM, typename TL, int... NL, typename TR>                                             \
    bool operator _OP_(Field<TM, TL, NL...> const& lhs, TR const& rhs) {                                    \
        return traits::reduction<_REDUCTION_>(Expression<tags::_TAG_, Field<TM, TL, NL...>, TR>(lhs, rhs)); \
    };                                                                                                      \
    template <typename TL, typename TM, typename TR, int... NR>                                             \
    bool operator _OP_(TL const& lhs, Field<TM, TR, NR...> const& rhs) {                                    \
        return traits::reduction<_REDUCTION_>(Expression<tags::_TAG_, TL, Field<TM, TR, NR...>>(lhs, rhs)); \
    };                                                                                                      \
    template <typename TM, typename TL, int... NL, typename... TR>                                          \
    bool operator _OP_(Field<TM, TL, NL...> const& lhs, Expression<TR...> const& rhs) {                     \
        return traits::reduction<_REDUCTION_>(                                                              \
            Expression<tags::_TAG_, Field<TM, TL, NL...>, Expression<TR...>>(lhs, rhs));                    \
    };                                                                                                      \
    template <typename... TL, typename TM, typename TR, int... NR>                                          \
    bool operator _OP_(Expression<TL...> const& lhs, Field<TM, TR, NR...> const& rhs) {                     \
        return traits::reduction<_REDUCTION_>(                                                              \
            Expression<tags::_TAG_, Expression<TL...>, Field<TM, TR, NR...>>(lhs, rhs));                    \
    };                                                                                                      \
    template <typename TM, typename TL, int... NL, typename TR, int... NR>                                  \
    bool operator _OP_(Field<TM, TL, NL...> const& lhs, Field<TM, TR, NR...> const& rhs) {                  \
        return traits::reduction<_REDUCTION_>(                                                              \
            Expression<tags::_TAG_, Field<TM, TL, NL...>, Field<TM, TR, NR...>>(lhs, rhs));                 \
    };

_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(not_equal_to, tags::logical_or, !=)
_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(equal_to, tags::logical_and, ==)
_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(less, tags::logical_and, <)
_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(greater, tags::logical_and, >)
_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(less_equal, tags::logical_and, <=)
_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(greater_equal, tags::logical_and, >=)
#undef _SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR

}  // namespace simpla

#endif  // SIMPLA_FIELD_H
