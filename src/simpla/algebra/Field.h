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
#include <simpla/utilities/EntityId.h>
#include <simpla/utilities/ExpressionTemplate.h>
#include <simpla/utilities/FancyStream.h>
#include <simpla/utilities/Range.h>
#include <simpla/utilities/nTuple.h>
#include <simpla/utilities/sp_def.h>
#include "Algebra.h"
#include "CalculusPolicy.h"
namespace simpla {

template <typename>
class calculator;

template <typename TM, typename TV, int IFORM = VERTEX, int DOF = 1>
class Field : public engine::Attribute {
   private:
    typedef Field<TM, TV, IFORM, DOF> field_type;
    SP_OBJECT_HEAD(field_type, engine::Attribute);

   public:
    typedef TV value_type;
    typedef TM mesh_type;
    static constexpr int iform = IFORM;
    static constexpr int dof = DOF;
    static constexpr int NDIMS = mesh_type::NDIMS;
    static constexpr int NUMBER_OF_SUB = ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);

    typedef std::true_type prefer_pass_by_reference;
    typedef std::false_type is_expression;
    typedef std::true_type is_field;
    typedef std::conditional_t<DOF == 1, value_type, nTuple<value_type, DOF>> cell_tuple;
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME), cell_tuple, nTuple<cell_tuple, 3>>
        field_value_type;

   private:
    typedef Array<value_type, NDIMS> array_type;
    array_type m_data_[NUMBER_OF_SUB][DOF];

    mesh_type* m_mesh_ = nullptr;
    std::shared_ptr<EntityRange> m_range_ = nullptr;

   public:
    template <typename... Args>
    explicit Field(engine::Domain* d, Args&&... args)
        : engine::Attribute(IFORM, DOF, typeid(value_type), d,
                            std::make_shared<data::DataTable>(std::forward<Args>(args)...)),
          m_mesh_(dynamic_cast<mesh_type*>(engine::Attribute::GetDomain()->GetMesh())){};

    template <typename... Args>
    explicit Field(engine::MeshBase* d, Args&&... args)
        : engine::Attribute(IFORM, DOF, typeid(value_type), d,
                            std::make_shared<data::DataTable>(std::forward<Args>(args)...)),
          m_mesh_(dynamic_cast<mesh_type*>(engine::Attribute::GetDomain()->GetMesh())){};

    Field(this_type const& other) : engine::Attribute(other), m_mesh_(other.m_mesh_), m_range_(other.m_range_) {
        for (int i = 0; i < NUMBER_OF_SUB; ++i)
            for (int j = 0; j < DOF; ++j) { array_type(other.m_data_[i][j]).swap(m_data_[i][j]); }
    }

    Field(this_type&& other) : engine::Attribute(other), m_mesh_(other.m_mesh_), m_range_(other.m_range_) {
        for (int i = 0; i < NUMBER_OF_SUB; ++i)
            for (int j = 0; j < DOF; ++j) { array_type(std::move(other.m_data_[i][j])).swap(m_data_[i][j]); }
    }

    Field(this_type const& other, EntityRange const& r)
        : engine::Attribute(other), m_mesh_(other.m_mesh_), m_range_(std::make_shared<EntityRange>(r)) {
        for (int i = 0; i < NUMBER_OF_SUB; ++i)
            for (int j = 0; j < DOF; ++j) { array_type(other.m_data_[i][j]).swap(m_data_[i][j]); }
    }

    ~Field() override = default;

    size_type size() const override {
        size_type s = 0;
        if (m_range_ == nullptr) {
            for (int i = 0; i < NUMBER_OF_SUB; ++i)
                for (int j = 0; j < DOF; ++j) { s += m_data_[i][j].size(); }
        } else {
            s = m_range_->size() * DOF;
        }
        return s;
    }

    void Clear() {
        SetUp();
        for (int i = 0; i < NUMBER_OF_SUB; ++i)
            for (int j = 0; j < DOF; ++j) { m_data_[i][j].Clear(); }
    }

    bool empty() const override { return m_data_[0][0].empty(); }

    this_type& operator=(this_type const& other) {
        Assign(other);
        return *this;
    }

    template <typename TR>
    this_type& operator=(TR const& rhs) {
        Assign(rhs);
        return *this;
    };

    void Push(const std::shared_ptr<data::DataBlock>& d, const EntityRange* r) override {
        Click();
        m_range_ = r == nullptr ? nullptr : std::make_shared<EntityRange>(*r);

        if (d != nullptr) {
            auto& t = d->cast_as<data::DataMultiArray<value_type, NDIMS>>();
            for (int i = 0; i < NUMBER_OF_SUB; ++i)
                for (int j = 0; j < DOF; ++j) { array_type(t.GetArray(i * DOF + j)).swap(m_data_[i][j]); }
            Tag();
        }
    }

    std::shared_ptr<data::DataBlock> Pop() override {
        auto res = std::make_shared<data::DataMultiArray<value_type, NDIMS>>(NUMBER_OF_SUB);
        for (int i = 0; i < NUMBER_OF_SUB; ++i)
            for (int j = 0; j < DOF; ++j) { array_type(m_data_[i][j]).swap(res->GetArray(i * DOF + j)); }
        return res;
    }

    array_type const* operator[](int n) const {
        return (IFORM == VERTEX || IFORM == VOLUME) ? (&m_data_[0][n % DOF]) : m_data_[n % 3];
    }

    array_type* operator[](int n) {
        return (IFORM == VERTEX || IFORM == VOLUME) ? (&m_data_[0][n % DOF]) : m_data_[n % 3];
    }

    //    value_type& operator()(index_type i, index_type j, index_type k, index_type w = 0) {
    //        return m_data_[w & 0b111][w >> 3](i, j, k);
    //    }
    //
    //    value_type const& operator()(index_type i, index_type j, index_type k, index_type w = 0) const {
    //        return m_data_[w & 0b111][w >> 3](i, j, k);
    //    }

    //*****************************************************************************************************************
    this_type operator[](EntityRange const& d) const { return this_type(*this, d); }

    typedef calculator<mesh_type> calculus_policy;

    value_type const& at(EntityId s) const { return calculus_policy::getValue(*m_mesh_, *this, s); }

    value_type& at(EntityId s) { return calculus_policy::getValue(*m_mesh_, *this, s); }

    value_type const& operator[](EntityId s) const { return at(s); }

    value_type& operator[](EntityId s) { return at(s); }

    template <typename... Args>
    auto gather(Args&&... args) const {
        return calculus_policy::gather(*m_mesh_, *this, std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto scatter(Args&&... args) {
        return calculus_policy::scatter(*m_mesh_, *this, std::forward<Args>(args)...);
    }

    void SetUp() override {
        engine::Attribute::SetUp();

        for (int i = 0; i < NUMBER_OF_SUB; ++i) {
            int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][i];
            auto gw = m_mesh_->GetGhostWidth(tag);
            auto ib_box = m_mesh_->GetIndexBox(tag);
            std::get<0>(ib_box) -= gw;
            std::get<1>(ib_box) += gw;

            for (int j = 0; j < DOF; ++j) {
                if (!m_data_[i][j].empty()) { continue; }
                array_type(ib_box).swap(m_data_[i][j]);
            }
        }
        Tag();
    }

    template <typename TOther>
    void DeepCopy(TOther const& other) {
        for (int i = 0; i < NUMBER_OF_SUB; ++i)
            for (int j = 0; j < DOF; ++j) { m_data_[i][j].DeepCopy(other[i][j]); }
    }

    template <typename Other>
    void Assign(Other const& other, ENABLE_IF(!(concept::is_callable<Other(EntityId)>::value ||
                                                concept::is_callable<Other(point_type const&)>::value))) {
        DoSetUp();
        if (m_range_ == nullptr) {
            for (int i = 0; i < NUMBER_OF_SUB; ++i)
                for (int j = 0; j < DOF; ++j) {
                    m_data_[i][j] = calculus_policy::getValue(*m_mesh_, other, i * DOF + j);
                }
        } else {
            index_tuple ib, ie;
            std::tie(ib, ie) = m_mesh_->GetIndexBox();
            for (int j = 0; j < DOF; ++j) {
                m_range_->foreach ([&](EntityId s) {
                    index_tuple idx = {s.x, s.y, s.z};
                    if (ib[0] <= idx[0] && idx[0] < ie[0] &&  //
                        ib[1] <= idx[1] && idx[1] < ie[1] &&  //
                        ib[2] <= idx[2] && idx[2] < ie[2])    //
                    {
                        m_data_[EntityIdCoder::m_id_to_sub_index_[s.w & 0b111]][j].at(idx) = array_type::getValue(
                            calculus_policy::getValue(*m_mesh_, other, s.w | (j << 3), IdxShift{0, 0, 0}), idx);
                    }
                });
            }
        }
    }

    template <typename TFun>
    void Assign(TFun const& fun, ENABLE_IF((std::is_same<std::result_of_t<TFun(EntityId)>, value_type>::value))) {
        DoSetUp();

        if (m_range_ == nullptr) {
            for (int i = 0; i < NUMBER_OF_SUB; ++i)
                for (int j = 0; j < DOF; ++j) {
                    int w = EntityIdCoder::m_sub_index_to_id_[IFORM][i] | (j << 3);
                    m_data_[i][j] = [&](index_tuple const& idx) {
                        EntityId s;
                        s.w = static_cast<int16_t>(w);
                        s.x = static_cast<int16_t>(idx[0]);
                        s.y = static_cast<int16_t>(idx[1]);
                        s.z = static_cast<int16_t>(idx[2]);
                        return fun(s);
                    };
                }
        } else {
            index_tuple ib, ie;
            std::tie(ib, ie) = m_mesh_->GetIndexBox();

            for (int j = 0; j < DOF; ++j) {
                m_range_->foreach ([&](EntityId s) {
                    if (ib[0] <= s.x && s.x < ie[0] &&  //
                        ib[1] <= s.y && s.y < ie[1] &&  //
                        ib[2] <= s.z && s.z < ie[2])    //
                    {
                        s.w |= j << 3;
                        at(s) = fun(s);
                    }
                });
            }
        }
    }
    template <typename TFun>
    void Assign(TFun const& fun,
                ENABLE_IF((std::is_same<std::result_of_t<TFun(point_type const&)>, value_type>::value))) {
        Assign([&](EntityId s) { return fun(m_mesh_->point(s)); });
    }
    template <typename TFun>
    void Assign(TFun const& fun,
                ENABLE_IF(((!std::is_same<field_value_type, value_type>::value) &&
                           std::is_same<std::result_of_t<TFun(point_type const&)>, field_value_type>::value))) {
        Assign([&](EntityId s) { return calculus_policy::sample(*m_mesh_, s, fun(m_mesh_->point(s))); });
    }

};  // class Field

template <typename TM, typename TV, int IFORM, int DOF>
constexpr int Field<TM, TV, IFORM, DOF>::NUMBER_OF_SUB;  //= ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3) * DOF;

template <typename TM, typename TL, int NL, int DL>
auto operator<<(Field<TM, TL, NL, DL> const& lhs, int n) {
    return Expression<tags::bitwise_left_shift, const Field<TM, TL, NL, DL>, int>(lhs, n);
};

template <typename TM, typename TL, int NL, int DL>
auto operator>>(Field<TM, TL, NL, DL> const& lhs, int n) {
    return Expression<tags::bitwise_right_shifit, const Field<TM, TL, NL, DL>, int>(lhs, n);
};

#define _SP_DEFINE_FIELD_BINARY_FUNCTION(_TAG_, _FUN_)                                                      \
    template <typename TM, typename TL, int NL, int DL, typename TR>                                        \
    auto _FUN_(Field<TM, TL, NL, DL> const& lhs, TR const& rhs) {                                           \
        return Expression<tags::_TAG_, const Field<TM, TL, NL, DL>, const TR>(lhs, rhs);                    \
    };                                                                                                      \
    template <typename TL, typename TM, typename TR, int NR, int DR>                                        \
    auto _FUN_(TL const& lhs, Field<TM, TR, NR, DR> const& rhs) {                                           \
        return Expression<tags::_TAG_, const TL, const Field<TM, TR, NR, DR>>(lhs, rhs);                    \
    };                                                                                                      \
    template <typename TM, typename TL, int NL, int DL, typename... TR>                                     \
    auto _FUN_(Field<TM, TL, NL, DL> const& lhs, Expression<TR...> const& rhs) {                            \
        return Expression<tags::_TAG_, const Field<TM, TL, NL, DL>, const Expression<TR...>>(lhs, rhs);     \
    };                                                                                                      \
    template <typename... TL, typename TM, typename TR, int NR, int DR>                                     \
    auto _FUN_(Expression<TL...> const& lhs, Field<TM, TR, NR, DR> const& rhs) {                            \
        return Expression<tags::_TAG_, const Expression<TL...>, const Field<TM, TR, NR, DR>>(lhs, rhs);     \
    };                                                                                                      \
    template <typename TM, typename TL, int NL, int DL, typename TR, int NR, int DR>                        \
    auto _FUN_(Field<TM, TL, NL, DL> const& lhs, Field<TM, TR, NR, DR> const& rhs) {                        \
        return Expression<tags::_TAG_, const Field<TM, TL, NL, DL>, const Field<TM, TR, NR, DR>>(lhs, rhs); \
    };

#define _SP_DEFINE_FIELD_UNARY_FUNCTION(_TAG_, _FUN_)                     \
    template <typename TM, typename TL, int NL, int DL>                   \
    auto _FUN_(Field<TM, TL, NL, DL> const& lhs) {                        \
        return Expression<tags::_TAG_, const Field<TM, TL, NL, DL>>(lhs); \
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

#define _SP_DEFINE_FIELD_COMPOUND_OP(_OP_)                                                              \
    template <typename TM, typename TL, int NL, int DL, typename TR>                                    \
    Field<TM, TL, NL, DL>& operator _OP_##=(Field<TM, TL, NL, DL>& lhs, TR const& rhs) {                \
        lhs = lhs _OP_ rhs;                                                                             \
        return lhs;                                                                                     \
    }                                                                                                   \
    template <typename TM, typename TL, int NL, int DL, typename... TR>                                 \
    Field<TM, TL, NL, DL>& operator _OP_##=(Field<TM, TL, NL, DL>& lhs, Expression<TR...> const& rhs) { \
        lhs = lhs _OP_ rhs;                                                                             \
        return lhs;                                                                                     \
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

#define _SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(_TAG_, _REDUCTION_, _OP_)                                       \
    template <typename TM, typename TL, int NL, int DL, typename TR>                                             \
    auto operator _OP_(Field<TM, TL, NL, DL> const& lhs, TR const& rhs) {                                        \
        return reduction<_REDUCTION_>(Expression<tags::_TAG_, const Array<TL, NL>, const TR>(lhs, rhs));         \
    };                                                                                                           \
    template <typename TL, typename TM, typename TR, int NR, int DR>                                             \
    auto operator _OP_(TL const& lhs, Array<TR, NR> const& rhs) {                                                \
        return reduction<_REDUCTION_>(Expression<tags::_TAG_, const TL, const Field<TM, TR, NR, DR>>(lhs, rhs)); \
    };                                                                                                           \
    template <typename TM, typename TL, int NL, int DL, typename... TR>                                          \
    auto operator _OP_(Field<TM, TL, NL, DL> const& lhs, Expression<TR...> const& rhs) {                         \
        return reduction<_REDUCTION_>(                                                                           \
            Expression<tags::_TAG_, const Field<TM, TL, NL, DL>, const Expression<TR...>>(lhs, rhs));            \
    };                                                                                                           \
    template <typename... TL, typename TM, typename TR, int NR, int DR>                                          \
    auto operator _OP_(Expression<TL...> const& lhs, Field<TM, TR, NR, DR> const& rhs) {                         \
        return reduction<_REDUCTION_>(                                                                           \
            Expression<tags::_TAG_, const Expression<TL...>, const Field<TM, TR, NR, DR>>(lhs, rhs));            \
    };                                                                                                           \
    template <typename TM, typename TL, int NL, int DL, typename TR, int NR, int DR>                             \
    auto operator _OP_(Field<TM, TL, NL, DL> const& lhs, Field<TM, TR, NR, DR> const& rhs) {                     \
        return reduction<_REDUCTION_>(                                                                           \
            Expression<tags::_TAG_, const Field<TM, TL, NL, DL>, const Field<TM, TR, NR, DR>>(lhs, rhs));        \
    };

_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(not_equal_to, tags::logical_or, !=)
_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(equal_to, tags::logical_and, ==)
_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(less, tags::logical_and, <)
_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(greater, tags::logical_and, >)
_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(less_equal, tags::logical_and, <=)
_SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(greater_equal, tags::logical_and, >=)
#undef _SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR

}  // namespace simpla
//        static int tag[4][3] = {{0, 0, 0}, {1, 2, 4}, {6, 5, 3}, {7, 7, 7}};
//        for (int j = 0; j < NUMBER_OF_SUB; ++j) {
//            VERBOSE << m_data_[j].GetIndexBox() << "~" << m_mesh_->GetIndexBox(tag[IFORM][(j / DOF) % 3]) <<
//            std::endl;
//        }
//        VERBOSE << s.x << "," << s.y << "," << s.z << "   " << std::boolalpha
//                << m_data_[EntityIdCoder::SubIndex<IFORM, DOF>(s)].empty() << std::endl;
//        static constexpr int id_2_sub_edge[3] = {1, 2, 4};
//        static constexpr int id_2_sub_face[3] = {6, 5, 3};
//        if (m_range_.empty()) {
//            for (int i = 0; i < NUMBER_OF_SUB; ++i) {
//                int16_t w = 0;
//                switch (IFORM) {
//                    case VERTEX:
//                        w = static_cast<int16_t>(i << 3);
//                        break;
//                    case EDGE:
//                        w = static_cast<int16_t>(((i % DOF) << 3) | id_2_sub_edge[(i / DOF) % 3]);
//                        break;
//                    case FACE:
//                        w = static_cast<int16_t>(((i % DOF) << 3) | id_2_sub_face[(i / DOF) % 3]);
//                        break;
//                    case VOLUME:
//                        w = static_cast<int16_t>((i << 3) | 0b111);
//                        break;
//                    default:
//                        break;
//                }
//                m_data_[i].Foreach([&](index_tuple const& idx, value_type& v) {
//                    EntityId s;
//                    s.w = w;
//                    s.x = static_cast<int16_t>(idx[0]);
//                    s.y = static_cast<int16_t>(idx[1]);
//                    s.z = static_cast<int16_t>(idx[2]);
//                    v = calculus_policy::getValue(*m_mesh_, other, s);
//                });
//            }
//        } else {
//        }
// namespace declare {
//
// template <typename TM, typename TV, int IFORM, int DOF>
// class Field_ : public Field<TM, TV, IFORM, DOF> {
//    typedef Field_<TM, TV, IFORM, DOF> this_type;
//    typedef Field<TM, TV, IFORM, DOF> base_type;
//
//   public:
//    template <typename... Args>
//    explicit Field_(Args&&... args) : base_type(std::forward<Args>(args)...) {}
//
//    Field_(this_type const& other) : base_type(other){};
//    //    Field_(this_type&& other) = delete;
//    ~Field_() {}
//
//    using base_type::operator[];
//    using base_type::operator=;
//    using base_type::operator();
//
//    this_type operator[](EntityRange const& d) const { return this_type(*this, d); }
//};
//}  // namespace declare
//}  // namespace algebra
//
// template <typename TM, typename TV, int IFORM = VERTEX, int DOF = 1>
// using Field = algebra::declare::Field_<TM, TV, IFORM, DOF>;

#endif  // SIMPLA_FIELD_H
