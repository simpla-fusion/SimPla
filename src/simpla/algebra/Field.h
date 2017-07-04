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
#include <simpla/utilities/type_traits.h>
#include "Algebra.h"
#include "CalculusPolicy.h"
namespace simpla {
namespace _detail {
template <typename T, T... M>
struct nProduct;
template <typename T, T N, T... M>
struct nProduct<T, N, M...> {
    static constexpr T value = N * nProduct<T, M...>::value;
};
template <typename T>
struct nProduct<T> {
    static constexpr T value = 1;
};
}

template <typename>
class calculator;

template <typename TM, typename TV, int...>
class Field;

// namespace traits {
// template <typename TM, typename TV, int... I>
// struct reference<Field<TM, TV, I...>> {
//    typedef const Field<TM, TV, I...>& type;
//};
//
// template <typename TM, typename TV, int... I>
// struct reference<const Field<TM, TV, I...>> {
//    typedef const Field<TM, TV, I...>& type;
//};
//}

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
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME), ele_type, nTuple<ele_type, 3>> field_value_type;

   private:
    typedef typename mesh_type::template array_type<value_type> array_type;
    typedef nTuple<array_type, NUM_OF_SUB, DOF...> data_type;
    //    typedef Array<value_type> array_type;
    nTuple<array_type, NUM_OF_SUB, DOF...> m_data_;
    EntityRange m_range_;
    mesh_type const* m_mesh_ = nullptr;

   public:
    //    template <typename... Args>
    //    explicit Field(Args&&... args) : engine::Attribute(IFORM, DOF, typeid(value_type),
    //    std::forward<Args>(args)...){};
    template <typename TGrp, typename... Args>
    explicit Field(TGrp* grp, Args&&... args)
        : engine::Attribute(IFORM, _detail::nProduct<int, DOF...>::value, typeid(value_type),
                            dynamic_cast<engine::AttributeGroup*>(grp),
                            std::make_shared<data::DataTable>(std::forward<Args>(args)...)) {}

    ~Field() override = default;

    Field(this_type const& other)
        : engine::Attribute(other), m_data_(other.m_data_), m_mesh_(other.m_mesh_), m_range_(other.m_range_) {}
    Field(this_type&& other)
        : engine::Attribute(std::forward<engine::Attribute>(other)),
          m_data_(other.m_data_),
          m_mesh_(other.m_mesh_),
          m_range_(other.m_range_) {}
    Field(this_type const& other, EntityRange r)
        : engine::Attribute(other), m_data_(other.m_data_), m_mesh_(other.m_mesh_), m_range_(r) {}

    size_type size() const override { return m_range_.size(); }

    auto& data() { return m_data_; }
    auto const& data() const { return m_data_; }

    void Clear() {
        DoUpdate();
        *this = 0;
    }
    void SetUndefined() {
        DoUpdate();
        *this = std::numeric_limits<value_type>::signaling_NaN();
    }

    bool empty() const override { return m_range_.empty(); }

    void swap(this_type& other) {
        engine::Attribute::swap(other);
        m_data_.swap(other.m_data_);
        m_range_.swap(other.m_range_);
        std::swap(m_mesh_, other.m_mesh_);
    }

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

    auto const& at(EntityId s) const { return data_type::operator()(s.w, s.x, s.y, s.z); }
    auto& at(EntityId s) { return data_type::operator()(s.w, s.x, s.y, s.z); }

    auto& operator[](EntityId s) { return at(s); }
    auto const& operator[](EntityId s) const { return at(s); }

    this_type operator[](EntityRange const& d) const { return this_type(*this, d); }
    this_type operator()(EntityRange const& d) const { return this_type(*this, d); }

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
            a.SetSpaceFillingCurve(m_mesh_->GetSpaceFillingCurve(IFORM, idx[0]));
        });
    }

    void TearDown() override {
        m_range_.reset();
        m_mesh_ = nullptr;
        m_data_.foreach ([&](array_type& a) { a.reset(); });
    }
    void Push(std::shared_ptr<data::DataBlock> p) override {
        auto d = std::dynamic_pointer_cast<data::DataMultiArray<value_type, NDIMS>>(p);
        int count = 0;
        m_data_.foreach ([&](array_type& a) {
            a.swap(d->GetArray(count));
            ++count;
        });
    };
    std::shared_ptr<data::DataBlock> Pop() override {
        auto res = std::make_shared<data::DataMultiArray<value_type, NDIMS>>(NUM_OF_SUB * GetDOF());
        int count = 0;
        m_data_.foreach ([&](array_type& a) {
            res->GetArray(count).swap(a);
            ++count;
        });
        return res;
    };
    //
    //    void Deserialize(const std::shared_ptr<data::DataBlock>& d) override {
    //        if (d != nullptr) {
    //            m_range_ = r;
    //            //            m_data_ = std::dynamic_pointer_cast<data_type>(d);
    //            Click();
    //        }
    //        DoUpdate();
    //    }
    //
    //    std::shared_ptr<data::DataBlock> Serialize() override {
    //        std::shared_ptr<data::DataBlock> res;
    //        //        auto res = std::dynamic_pointer_cast<data::DataBlock>(m_data_);
    //        DoTearDown();
    //        return res;
    //    }
    template <typename TOther>
    void DeepCopy(TOther const& other) {
        DoUpdate();
        m_data_ = (other);
    }

    this_type& operator=(this_type const& other) {
        m_data_ = (other);
        return *this;
    }
    template <typename TR>
    this_type& operator=(TR const& rhs) {
        DoUpdate();
        ASSERT(m_mesh_ != nullptr);
        Assign(m_range_, rhs);
        //        Assign(m_mesh_->GetRange(std::string(EntityIFORMName[IFORM]) + "_PATCH_BOUNDARY"), 0);
        return *this;
    };

    //    template <typename Others>
    //    void Assign(EntityRange const& r, Others const& expr) {}

    template <typename RHS>
    void Assign(EntityRange const& r, RHS const& rhs, ENABLE_IF((std::is_arithmetic<RHS>::value))) {
        m_data_ = (rhs);
        //        m_data_ = rhs;
        //        if (r.isNull()) {
        //        } else {
        //            m_data_.foreach ([&](int const* sub, auto& lhs) {
        //                int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][sub[0]];
        //                r.foreach ([&] __host__ __device__(auto s) {
        //                    if (tag == (s.w & 0b111)) { lhs.Assign(rhs, s.x, s.y, s.z); }
        // });
        // });
        //        }
    }

    template <typename... Others>
    void Assign(EntityRange const& r, Expression<Others...> const& rhs) {
        m_data_ = (rhs);

        //        m_data_ = expr;
        //        m_data_.foreach ([&](int const* sub, auto& lhs) {
        //            int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][sub[0]];
        //            auto rhs = calculator<mesh_type>::getValue(expr, *m_mesh_, tag | (sub[1] << 3));
        //            if (r.isNull()) {
        //                lhs = rhs;
        //            } else {
        //                r.foreach ([&] __host__ __device__(auto s) {
        //                    if (tag == (s.w & 0b111)) { lhs.Assign(rhs, s.x, s.y, s.z); }
        //                });
        //            }
        //        });
    }
    template <typename TFun>
    void Assign(EntityRange const& r, TFun const& fun,
                ENABLE_IF((std::is_same<std::result_of_t<TFun(EntityId)>, value_type>::value))) {


        //        m_data_.foreach ([&](int const* sub, auto& lhs) {
        //            int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][sub[0]];
        //            int16_t w = static_cast<int16_t>(tag | (sub[1] << 3));
        //            if (r.isNull()) {
        //                lhs = [&] __host__ __device__(auto idx) {
        //                    EntityId s;
        //                    s.w = static_cast<int16_t>(w);
        //                    s.x = static_cast<int16_t>(idx[0]);
        //                    s.y = static_cast<int16_t>(idx[1]);
        //                    s.z = static_cast<int16_t>(idx[2]);
        //                    return fun(s);
        //                };
        //            } else {
        //                r.foreach ([&] __host__ __device__(auto s) {
        //                    if (tag == s.w) {
        //                        s.w = w;
        //                        lhs.Assign(fun(s), s.x, s.y, s.z);
        //                    }
        //                });
        //            }
        //        });
    }

    template <typename TFun>
    void Assign(EntityRange const& r, TFun const& fun,
                ENABLE_IF((std::is_same<std::result_of_t<TFun(point_type)>, value_type>::value))) {
        //        m_data_.foreach ([&](int const* sub, auto& lhs) {
        //            int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][sub[0]];
        //            int16_t w = static_cast<int16_t>(tag | (sub[1] << 3));
        //            if (r.isNull()) {
        //                lhs = [&] __host__ __device__(auto idx) {
        //                    EntityId s;
        //                    s.w = static_cast<int16_t>(w);
        //                    s.x = static_cast<int16_t>(idx[0]);
        //                    s.y = static_cast<int16_t>(idx[1]);
        //                    s.z = static_cast<int16_t>(idx[2]);
        //                    return fun(m_mesh_->point(s));
        //                };
        //            } else {
        //                r.foreach ([&] __host__ __device__(auto s) {
        //                    if (tag == s.w) {
        //                        s.w = w;
        //                        lhs.Assign(fun(m_mesh_->point(s)), s.x, s.y, s.z);
        //                    }
        //                });
        //            }
        //        });
    }

    template <typename TFun>
    void Assign(EntityRange const& r, TFun const& fun,
                ENABLE_IF((std::is_same<std::result_of_t<TFun(EntityId)>, field_value_type>::value &&
                           !std::is_same<field_value_type, value_type>::value))) {
        //        m_data_.foreach ([&](int const* sub, auto& lhs) {
        //            int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][sub[0]];
        //            int16_t w = static_cast<int16_t>(tag | (sub[1] << 3));
        //
        //            int n = sub[(IFORM == VERTEX || IFORM == VOLUME) ? 0 : 1];
        //            if (r.isNull()) {
        //                lhs = [&] __host__ __device__(auto idx) {
        //                    EntityId s;
        //                    s.w = static_cast<int16_t>(w);
        //                    s.x = static_cast<int16_t>(idx[0]);
        //                    s.y = static_cast<int16_t>(idx[1]);
        //                    s.z = static_cast<int16_t>(idx[2]);
        //                    return fun(s)[n];
        //                };
        //            } else {
        //                r.foreach ([&] __host__ __device__(auto s) {
        //                    if (tag == s.w) {
        //                        s.w = w;
        //                        lhs.Assign(fun(s)[n], s.x, s.y, s.z);
        //                    }
        //                });
        //            }
        //        });
    }
    template <typename TFun>
    void Assign(EntityRange const& r, TFun const& fun,
                ENABLE_IF((std::is_same<std::result_of_t<TFun(point_type)>, field_value_type>::value &&
                           !std::is_same<field_value_type, value_type>::value))) {
        //        int16_t w = EntityIdCoder::m_sub_index_to_id_[IFORM][sub[0]];
        //
        //        if (r.isNull()) {
        //            auto sfc = m_mesh_->GetSpaceFillingCurve(IFORM);
        //            int n = (IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3;
        //            sfc.foreach ([&](auto& idx) {
        //                EntityId s;
        //                s.w = static_cast<int16_t>(w);
        //                s.x = static_cast<int16_t>(idx[0]);
        //                s.y = static_cast<int16_t>(idx[1]);
        //                s.z = static_cast<int16_t>(idx[2]);
        //                auto v = fun(m_mesh_->point(s))[n];
        //                m_data_[0](idx) =
        //            });
        //        } else {
        //            r.foreach ([&] __host__ __device__(auto s) {
        //                if (tag == s.w) {
        //                    s.w = w;
        //                    lhs.Assign(fun(m_mesh_->point(s))[n], s.x, s.y, s.z);
        //                }
        //            });
        //        }
        //    });
    }
};  // class Field

// template <typename TM, typename TL, int... NL>
// auto operator<<(Field<TM, TL, NL...> const& lhs, int n) {
//    return Expression<tags::bitwise_left_shift, Field<TM, TL, NL...>, int>(lhs, n);
//};
//
// template <typename TM, typename TL, int... NL>
// auto operator>>(Field<TM, TL, NL...> const& lhs, int n) {
//    return Expression<tags::bitwise_right_shifit, Field<TM, TL, NL...>, int>(lhs, n);
//};

#define _SP_DEFINE_FIELD_BINARY_FUNCTION(_TAG_, _FUN_)                                  \
    template <typename TM, typename TL, int... NL, typename TR>                         \
    auto _FUN_(Field<TM, TL, NL...> const& lhs, TR const& rhs) {                        \
        return _FUN_(lhs.data(), rhs);                                                  \
    };                                                                                  \
    template <typename TL, typename TM, typename TR, int... NR>                         \
    auto _FUN_(TL const& lhs, Field<TM, TR, NR...> const& rhs) {                        \
        return _FUN_(lhs, rhs.data());                                                  \
    };                                                                                  \
    template <typename TM, typename TL, int... NL, typename... TR>                      \
    auto _FUN_(Field<TM, TL, NL...> const& lhs, Expression<TR...> const& rhs) {         \
        return _FUN_(lhs.data(), rhs);                                                  \
    };                                                                                  \
    template <typename... TL, typename TM, typename TR, int... NR>                      \
    auto _FUN_(Expression<TL...> const& lhs, Field<TM, TR, NR...> const& rhs) {         \
        return _FUN_(lhs, rhs.data());                                                  \
    };                                                                                  \
    template <typename ML, typename TL, int... NL, typename MR, typename TR, int... NR> \
    auto _FUN_(Field<ML, TL, NL...> const& lhs, Field<MR, TR, NR...> const& rhs) {      \
        return _FUN_(lhs.data(), rhs.data());                                           \
    };

#define _SP_DEFINE_FIELD_UNARY_FUNCTION(_TAG_, _FUN_) \
    template <typename TM, typename TL, int... NL>    \
    auto _FUN_(Field<TM, TL, NL...> const& lhs) {     \
        return _FUN_(lhs.data());                     \
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

#define _SP_DEFINE_FIELD_BINARY_BOOLEAN_OPERATOR(_TAG_, _REDUCTION_, _OP_)                 \
    template <typename TM, typename TL, int... NL, typename TR>                            \
    bool operator _OP_(Field<TM, TL, NL...> const& lhs, TR const& rhs) {                   \
        return lhs.data() _OP_ rhs;                                                        \
    };                                                                                     \
    template <typename TL, typename TM, typename TR, int... NR>                            \
    bool operator _OP_(TL const& lhs, Field<TM, TR, NR...> const& rhs) {                   \
        return lhs _OP_ rhs.data();                                                        \
    };                                                                                     \
    template <typename TM, typename TL, int... NL, typename... TR>                         \
    bool operator _OP_(Field<TM, TL, NL...> const& lhs, Expression<TR...> const& rhs) {    \
        return lhs.data() _OP_ rhs;                                                        \
    };                                                                                     \
    template <typename... TL, typename TM, typename TR, int... NR>                         \
    bool operator _OP_(Expression<TL...> const& lhs, Field<TM, TR, NR...> const& rhs) {    \
        return lhs _OP_ rhs.data();                                                        \
    };                                                                                     \
    template <typename TM, typename TL, int... NL, typename TR, int... NR>                 \
    bool operator _OP_(Field<TM, TL, NL...> const& lhs, Field<TM, TR, NR...> const& rhs) { \
        return lhs.data() _OP_ rhs.dat();                                                  \
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
