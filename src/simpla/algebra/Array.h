//
// Created by salmon on 16-12-28.
//

#ifndef SIMPLA_ARRAY_H
#define SIMPLA_ARRAY_H
#include "simpla/SIMPLA_config.h"

#include <simpla/utilities/FancyStream.h>
#include <initializer_list>
#include <limits>
#include <memory>
#include <tuple>

#include "simpla/utilities/Log.h"
#include "simpla/utilities/type_traits.h"

#include "ExpressionTemplate.h"
#include "SFC.h"
#include "nTuple.h"

namespace simpla {

template <typename V, typename SFC>
class Array;
typedef nTuple<index_type, 3> IdxShift;

struct ArrayBase {
    virtual std::type_info const& value_type_info() const = 0;
    virtual size_type size() const = 0;
    virtual void* pointer() = 0;
    virtual void const* pointer() const = 0;
    virtual int GetNDIMS() const = 0;
    virtual bool isSlowFirst() const = 0;
    virtual int GetIndexBox(index_type* lo, index_type* hi) const = 0;
    virtual int GetShape(index_type* lo, index_type* hi) const = 0;
    virtual bool empty() const = 0;
    virtual bool isNull() const = 0;
    virtual size_type CopyIn(ArrayBase const& other) = 0;
    virtual size_type CopyOut(ArrayBase& other) const { return other.CopyIn(*this); };
    virtual void Clear() = 0;
    virtual void reset(void*, index_type const* lo, index_type const* hi) = 0;
    virtual void reset(index_box_type const& b) = 0;
    virtual std::shared_ptr<ArrayBase> DuplicateArray() const = 0;
    virtual void Shift(index_type const*) = 0;
    virtual void Select(index_type const*, index_type const*) = 0;
    virtual std::ostream& Print(std::ostream& os, int indent) const = 0;
};

template <typename V, typename SFC = ZSFC<3>>
class Array : public ArrayBase {
   public:
    typedef V value_type;
    static constexpr value_type s_nan = std::numeric_limits<value_type>::signaling_NaN();
    static value_type m_null_;

   private:
    typedef Array<value_type, SFC> this_type;
    SFC m_sfc_;
    std::shared_ptr<value_type> m_holder_ = nullptr;
    value_type* m_data_ = nullptr;

   public:
    Array() = default;
    virtual ~Array(){};

    Array(SFC const& sfc) : m_sfc_(sfc), m_data_(nullptr), m_holder_(nullptr) {}

    Array(this_type const& other) : m_sfc_(other.m_sfc_), m_data_(other.m_data_), m_holder_(other.m_holder_) {}
    Array(this_type& other) : m_sfc_(other.m_sfc_), m_data_(other.m_data_), m_holder_(other.m_holder_) {}

    Array(this_type&& other) noexcept
        : m_sfc_(other.m_sfc_), m_data_(other.m_data_), m_holder_(std::shared_ptr<value_type>(other.m_holder_)) {}

    Array(this_type const& other, IdxShift s) : Array(other) { m_sfc_.Shift(s); }
    template <typename... Args>
    explicit Array(Args&&... args) : m_sfc_(std::forward<Args>(args)...) {}

    template <typename... Args>
    explicit Array(value_type* d, Args&&... args)
        : m_holder_(nullptr), m_data_(d), m_sfc_(std::forward<Args>(args)...) {}

    template <typename... Args>
    explicit Array(std::shared_ptr<value_type> const& d, Args&&... args)
        : m_holder_(d), m_data_(m_holder_.get()), m_sfc_(std::forward<Args>(args)...) {}

    void swap(this_type& other) {
        std::swap(m_holder_, other.m_holder_);
        std::swap(m_data_, other.m_data_);
        m_sfc_.swap(other.m_sfc_);
    }
    std::shared_ptr<ArrayBase> DuplicateArray() const override {
        return std::shared_ptr<ArrayBase>(new this_type(*this));
    };
    void alloc();
    void free();
    std::ostream& Print(std::ostream& os, int indent) const override;

    bool isSlowFirst() const override { return m_sfc_.isSlowFirst(); };
    std::type_info const& value_type_info() const override { return typeid(value_type); };
    size_type size() const override { return m_sfc_.size(); }
    void* pointer() override { return m_data_; }
    void const* pointer() const override { return m_data_; }
    int GetNDIMS() const override { return m_sfc_.GetNDIMS(); }
    int GetIndexBox(index_type* lo, index_type* hi) const override { return m_sfc_.GetIndexBox(lo, hi); }
    int GetShape(index_type* lo, index_type* hi) const override { return m_sfc_.GetShape(lo, hi); }

    auto GetIndexBox() const { return m_sfc_.GetIndexBox(); }
    auto GetShape() const { return m_sfc_.GetShape(); }

    bool empty() const override { return m_data_ == nullptr || size() == 0; }
    bool isNull() const override { return m_data_ == nullptr || size() == 0; }

    size_type CopyIn(ArrayBase const& other) override {
        size_type count = 0;
        if (auto* p = dynamic_cast<this_type const*>(&other)) { count = CopyIn(*p); }
        return count;
    }
    size_type CopyOut(ArrayBase& other) const override {
        size_type count = 0;
        if (auto* p = dynamic_cast<this_type*>(&other)) { count = CopyOut(*p); }
        return count;
    };
    void reset(index_box_type const& b) override {
        m_data_ = nullptr;
        m_holder_.reset();
        m_sfc_.reset(b);
    }

    void reset(void* d, index_type const* lo, index_type const* hi) override {
        m_data_ = reinterpret_cast<value_type*>(d);
        m_holder_.reset();
        m_sfc_.reset(lo, hi);
    };

    template <typename... Args>
    void reset(value_type* d, Args&&... args) {
        m_data_ = d;
        m_holder_.reset();
        m_sfc_.reset(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void reset(std::shared_ptr<value_type> const& d, Args&&... args) {
        m_holder_ = d;
        m_data_ = m_holder_.get();
        m_sfc_.reset(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void reset(Args&&... args) {
        m_sfc_.reset(std::forward<Args>(args)...);
        if (m_sfc_.empty()) {
            m_data_ = nullptr;
            m_holder_.reset();
        }
    }
    template <typename... Args>
    bool in_box(Args&&... args) const {
        return m_sfc_.in_box(std::forward<Args>(args)...);
    }

    SFC const& GetSpaceFillingCurve() const { return m_sfc_; }

    template <typename TFun>
    void Foreach(TFun const& fun) {
        alloc();
        m_sfc_.Foreach([&](auto&&... s) { fun(at(std::forward<decltype(s)>(s)...), std::forward<decltype(s)>(s)...); });
    }
    template <typename TFun>
    void Foreach(TFun const& fun) const {
        if (!isNull()) { return; }
        m_sfc_.Foreach([&](auto&&... s) { fun(at(std::forward<decltype(s)>(s)...), std::forward<decltype(s)>(s)...); });
    }

    std::shared_ptr<value_type>& GetData() { return m_holder_; }
    std::shared_ptr<value_type> const& GetData() const { return m_holder_; }

    value_type* get() { return m_data_; }
    value_type const* get() const { return m_data_; }

    size_type CopyIn(this_type const& other) {
        alloc();
        return m_sfc_.Overlap(other.m_sfc_).Foreach([&] __host__ __device__(auto&&... s) {
            this->Set(other.Get(std::forward<decltype(s)>(s)...), std::forward<decltype(s)>(s)...);
        });
    };
    size_type CopyOut(this_type& other) const { return other.CopyIn(*this); };

    void DeepCopy(value_type const* other) {
        alloc();
        m_sfc_.Copy(m_data_, other);
    }
    void Fill(value_type v) {
        alloc();
        m_sfc_.Foreach([&] __host__ __device__(auto&&... s) { this->Set(v, std::forward<decltype(s)>(s)...); });
    }
    void Clear() override {
        alloc();
        memset(m_data_, 0, m_sfc_.shape_size() * sizeof(value_type));
    }

    this_type& operator=(this_type const& rhs) {
        Assign(rhs);
        return (*this);
    }

    template <typename TR>
    this_type& operator=(TR const& rhs) {
        Assign(rhs);
        return (*this);
    }

    void Shift(index_type const* idx) override { m_sfc_.Shift(idx); }
    void Select(index_type const* lo, index_type const* hi) override { m_sfc_.Select(lo, hi); }

    template <typename... Args>
    void Shift(Args&&... args) {
        m_sfc_.Shift(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void Select(Args&&... args) {
        m_sfc_.Select(std::forward<Args>(args)...);
    }

    template <typename... Args>
    this_type GetShift(Args&&... args) const {
        this_type res(*this);
        res.Shift(std::forward<Args>(args)...);
        return res;
    }
    template <typename... Args>
    this_type GetSelection(Args&&... args) const {
        this_type res(*this);
        res.Select(std::forward<Args>(args)...);
        return res;
    }

    __host__ __device__ value_type& operator[](std::initializer_list<index_type> const& s) {
        return m_data_[m_sfc_.hash(s)];
    }
    __host__ __device__ value_type const& operator[](std::initializer_list<index_type> const& s) const {
        return m_data_[m_sfc_.hash(s)];
    }

    template <typename... Args>
    __host__ __device__ value_type& at(Args&&... args) {
        ASSERT(m_sfc_.in_box(std::forward<Args>(args)...));
        return m_data_[m_sfc_.hash(std::forward<Args>(args)...)];
    }
    template <typename... Args>
    __host__ __device__ value_type const& at(Args&&... args) const {
        ASSERT(m_sfc_.in_box(std::forward<Args>(args)...));
        return m_data_[m_sfc_.hash(std::forward<Args>(args)...)];
    }
    template <typename... Args>
    __host__ __device__ value_type& operator()(index_type s0, Args&&... args) {
        return at(s0, std::forward<Args>(args)...);
    }
    template <typename... Args>
    __host__ __device__ value_type const& operator()(index_type s0, Args&&... args) const {
        return at(s0, std::forward<Args>(args)...);
    }
    template <typename... Args>
    void Set(value_type const& v, Args&&... args) {
        if (m_sfc_.in_box(std::forward<Args>(args)...)) { m_data_[m_sfc_.hash(std::forward<Args>(args)...)] = v; }
    }

    template <typename... Args>
    value_type const& Get(Args&&... args) const {
        return (!m_sfc_.in_box(std::forward<Args>(args)...)) ? s_nan
                                                             : m_data_[m_sfc_.hash(std::forward<Args>(args)...)];
    }

    template <typename RHS>
    void Assign(RHS const& rhs);
};
template <typename V, typename SFC>
constexpr typename Array<V, SFC>::value_type Array<V, SFC>::s_nan;
//
template <typename V, typename SFC>
typename Array<V, SFC>::value_type Array<V, SFC>::m_null_;

template <typename V, typename SFC>
void Array<V, SFC>::alloc() {
    if (m_data_ == nullptr && m_sfc_.shape_size() > 0) {
        if (m_holder_ == nullptr) { m_holder_ = spMakeShared<value_type>(m_data_, m_sfc_.shape_size()); }
        m_data_ = m_holder_.get();

#ifndef SP_ARRAY_INITIALIZE_VALUE
#elif SP_ARRAY_INITIALIZE_VALUE == SNAN
        Fill(std::numeric_limits<V>::signaling_NaN());
#elif SP_ARRAY_INITIALIZE_VALUE == QNAN
        Fill(std::numeric_limits<V>::quiet_NaN());
#elif SP_ARRAY_INITIALIZE_VALUE == DENORM_MIN
        Fill(std::numeric_limits<V>::denorm_min());
#else
        Fill(0);
#endif
    }
}

template <typename V, typename SFC>
void Array<V, SFC>::free() {
    m_holder_.reset();
    m_data_ = nullptr;
}
template <typename V, typename SFC>
std::ostream& Array<V, SFC>::Print(std::ostream& os, int indent) const {
    os << "Array<" << simpla::traits::type_name<value_type>::value() << ">" << GetIndexBox() << std::endl;
    int ndims = GetNDIMS();
    index_type lo[ndims], hi[ndims];
    GetIndexBox(lo, hi);
    FancyPrintNd<3>(os, *this, lo, hi, false, indent);
    return os;
}
namespace traits {
template <typename... T>
struct reference<Array<T...>> {
    typedef Array<T...> type;
};
template <typename T, typename... Others>
struct value_type<Array<T, Others...>> {
    typedef T type;
};
}  //    namespace traits {

namespace detail {
template <typename V, typename... Args>
decltype(auto) array_parser_function(std::false_type, V const& expr, Args&&... args) {
    return expr;
}
template <typename V, typename... Args>
decltype(auto) array_parser_function(std::true_type, V const& expr, Args&&... args) {
    return expr(std::forward<Args>(args)...);
}
template <typename V, typename... Args>
decltype(auto) array_parser(V const& expr, Args&&... args) {
    return array_parser_function(std::integral_constant<bool, traits::is_invocable<V, Args...>::value>(), expr,
                                 std::forward<Args>(args)...);
}
template <typename... V, typename... Args>
decltype(auto) array_parser(Array<V...> const& expr, Args&&... args) {
    return expr.Get(std::forward<Args>(args)...);
}
template <typename... V, typename... Args>
decltype(auto) array_parser(Array<V...>& expr, Args&&... args) {
    return expr.Get(std::forward<Args>(args)...);
}
template <typename TOP, typename... V, typename... Args>
decltype(auto) array_parser(Expression<TOP, V...> const& expr, Args&&... args);

template <size_type... I, typename TExpr, typename... Args>
decltype(auto) eval_helper_(std::index_sequence<I...>, TExpr const& expr, Args&&... args) {
    return expr.m_op_(array_parser(std::get<I>(expr.m_args_), std::forward<Args>(args)...)...);
}
template <typename TOP, typename... V, typename... Args>
decltype(auto) array_parser(Expression<TOP, V...> const& expr, Args&&... args) {
    return eval_helper_(std::index_sequence_for<V...>(), expr, std::forward<Args>(args)...);
}
}  // namespace detail {

template <typename V, typename SFC>
template <typename RHS>
void Array<V, SFC>::Assign(RHS const& rhs) {
    alloc();
    GetSpaceFillingCurve().Overlap(rhs).Foreach([&](auto&&... idx) {
        this->Set(detail::array_parser(rhs, std::forward<decltype(idx)>(idx)...), std::forward<decltype(idx)>(idx)...);
    });
};

template <typename... TL>
std::ostream& operator<<(std::ostream& os, Array<TL...> const& lhs) {
    return lhs.Print(os, 0);
};

template <typename... TL>
std::istream& operator>>(std::istream& is, Array<TL...>& lhs) {
    UNIMPLEMENTED;
    return is;
};

#define _SP_DEFINE_ARRAY_BINARY_OPERATOR(_OP_, _NAME_)                                      \
    template <typename... TL, typename TR>                                                  \
    auto operator _OP_(Array<TL...> const& lhs, TR const& rhs) {                            \
        return Expression<simpla::tags::_NAME_, Array<TL...>, TR>(lhs, rhs);                \
    };                                                                                      \
    template <typename TL, typename... TR>                                                  \
    auto operator _OP_(TL const& lhs, Array<TR...> const& rhs) {                            \
        return Expression<simpla::tags::_NAME_, TL, Array<TR...>>(lhs, rhs);                \
    };                                                                                      \
    template <typename... TL, typename... TR>                                               \
    auto operator _OP_(Array<TL...> const& lhs, Expression<TR...> const& rhs) {             \
        return Expression<simpla::tags::_NAME_, Array<TL...>, Expression<TR...>>(lhs, rhs); \
    };                                                                                      \
    template <typename... TL, typename... TR>                                               \
    auto operator _OP_(Expression<TL...> const& lhs, Array<TR...> const& rhs) {             \
        return Expression<simpla::tags::_NAME_, Expression<TL...>, Array<TR...>>(lhs, rhs); \
    };                                                                                      \
    template <typename... TL, typename... TR>                                               \
    auto operator _OP_(Array<TL...> const& lhs, Array<TR...> const& rhs) {                  \
        return Expression<simpla::tags::_NAME_, Array<TL...>, Array<TR...>>(lhs, rhs);      \
    };

#define _SP_DEFINE_ARRAY_UNARY_OPERATOR(_OP_, _NAME_)               \
    template <typename... TL>                                       \
    auto operator _OP_(Array<TL...> const& lhs) {                   \
        return Expression<simpla::tags::_NAME_, Array<TL...>>(lhs); \
    };

_SP_DEFINE_ARRAY_BINARY_OPERATOR(+, addition)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(-, subtraction)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(*, multiplication)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(/, division)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(%, modulo)

_SP_DEFINE_ARRAY_UNARY_OPERATOR(~, bitwise_not)
_SP_DEFINE_ARRAY_BINARY_OPERATOR (^, bitwise_xor)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(&, bitwise_and)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(|, bitwise_or)

template <typename... TL>
auto operator<<(Array<TL...> const& lhs, unsigned int n) {
    return Expression<simpla::tags::bitwise_left_shift, Array<TL...>, int>(lhs, n);
};

template <typename... TL>
auto operator>>(Array<TL...> const& lhs, unsigned int n) {
    return Expression<simpla::tags::bitwise_right_shifit, Array<TL...>, int>(lhs, n);
};
//_SP_DEFINE_ARRAY_BINARY_OPERATOR(bitwise_left_shift, <<)
//_SP_DEFINE_ARRAY_BINARY_OPERATOR(bitwise_right_shift, >>)

_SP_DEFINE_ARRAY_UNARY_OPERATOR(+, unary_plus)
_SP_DEFINE_ARRAY_UNARY_OPERATOR(-, unary_minus)
_SP_DEFINE_ARRAY_UNARY_OPERATOR(!, logical_not)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(&&, logical_and)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(||, logical_or)

#undef _SP_DEFINE_ARRAY_BINARY_OPERATOR
#undef _SP_DEFINE_ARRAY_UNARY_OPERATOR

#define _SP_DEFINE_ARRAY_BINARY_FUNCTION(_NAME_)                                            \
    template <typename... TL, typename TR>                                                  \
    auto _NAME_(Array<TL...> const& lhs, TR const& rhs) {                                   \
        return Expression<simpla::tags::_NAME_, Array<TL...>, TR>(lhs, rhs);                \
    };                                                                                      \
    template <typename TL, typename... TR>                                                  \
    auto _NAME_(TL const& lhs, Array<TR...> const& rhs) {                                   \
        return Expression<simpla::tags::_NAME_, TL, Array<TR...>>(lhs, rhs);                \
    };                                                                                      \
    template <typename... TL, typename... TR>                                               \
    auto _NAME_(Array<TL...> const& lhs, Expression<TR...> const& rhs) {                    \
        return Expression<simpla::tags::_NAME_, Array<TL...>, Expression<TR...>>(lhs, rhs); \
    };                                                                                      \
    template <typename... TL, typename... TR>                                               \
    auto _NAME_(Expression<TL...> const& lhs, Array<TR...> const& rhs) {                    \
        return Expression<simpla::tags::_NAME_, Expression<TL...>, Array<TR...>>(lhs, rhs); \
    };                                                                                      \
    template <typename... TL, typename... TR>                                               \
    auto _NAME_(Array<TL...> const& lhs, Array<TR...> const& rhs) {                         \
        return Expression<simpla::tags::_NAME_, Array<TL...>, Array<TR...>>(lhs, rhs);      \
    };

#define _SP_DEFINE_ARRAY_UNARY_FUNCTION(_NAME_)                    \
    template <typename... T>                                       \
    auto _NAME_(Array<T...> const& lhs) {                          \
        return Expression<simpla::tags::_NAME_, Array<T...>>(lhs); \
    }

_SP_DEFINE_ARRAY_UNARY_FUNCTION(cos)
_SP_DEFINE_ARRAY_UNARY_FUNCTION(acos)
_SP_DEFINE_ARRAY_UNARY_FUNCTION(cosh)
_SP_DEFINE_ARRAY_UNARY_FUNCTION(sin)
_SP_DEFINE_ARRAY_UNARY_FUNCTION(asin)
_SP_DEFINE_ARRAY_UNARY_FUNCTION(sinh)
_SP_DEFINE_ARRAY_UNARY_FUNCTION(tan)
_SP_DEFINE_ARRAY_UNARY_FUNCTION(tanh)
_SP_DEFINE_ARRAY_UNARY_FUNCTION(atan)
_SP_DEFINE_ARRAY_UNARY_FUNCTION(exp)
_SP_DEFINE_ARRAY_UNARY_FUNCTION(log)
_SP_DEFINE_ARRAY_UNARY_FUNCTION(log10)
_SP_DEFINE_ARRAY_UNARY_FUNCTION(sqrt)
_SP_DEFINE_ARRAY_BINARY_FUNCTION(atan2)
_SP_DEFINE_ARRAY_BINARY_FUNCTION(pow)

#undef _SP_DEFINE_ARRAY_BINARY_FUNCTION
#undef _SP_DEFINE_ARRAY_UNARY_FUNCTION

#define _SP_DEFINE_ARRAY_COMPOUND_OP(_OP_)                                            \
    template <typename... TL, typename TR>                                            \
    Array<TL...>& operator _OP_##=(Array<TL...>& lhs, TR const& rhs) {                \
        lhs = lhs _OP_ rhs;                                                           \
        return lhs;                                                                   \
    }                                                                                 \
    template <typename... TL, typename... TR>                                         \
    Array<TL...>& operator _OP_##=(Array<TL...>& lhs, Expression<TR...> const& rhs) { \
        lhs = lhs _OP_ rhs;                                                           \
        return lhs;                                                                   \
    }

_SP_DEFINE_ARRAY_COMPOUND_OP(+)
_SP_DEFINE_ARRAY_COMPOUND_OP(-)
_SP_DEFINE_ARRAY_COMPOUND_OP(*)
_SP_DEFINE_ARRAY_COMPOUND_OP(/)
_SP_DEFINE_ARRAY_COMPOUND_OP(%)
_SP_DEFINE_ARRAY_COMPOUND_OP(&)
_SP_DEFINE_ARRAY_COMPOUND_OP(|)
_SP_DEFINE_ARRAY_COMPOUND_OP (^)
_SP_DEFINE_ARRAY_COMPOUND_OP(<<)
_SP_DEFINE_ARRAY_COMPOUND_OP(>>)

#undef _SP_DEFINE_ARRAY_COMPOUND_OP

#define _SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR(_OP_, _NAME_, _REDUCTION_)                                    \
    template <typename... TL, typename TR>                                                                     \
    bool operator _OP_(Array<TL...> const& lhs, TR const& rhs) {                                               \
        return calculus::reduction<_REDUCTION_>(Expression<simpla::tags::_NAME_, Array<TL...>, TR>(lhs, rhs)); \
    };                                                                                                         \
    template <typename TL, typename... TR>                                                                     \
    bool operator _OP_(TL const& lhs, Array<TR...> const& rhs) {                                               \
        return calculus::reduction<_REDUCTION_>(Expression<simpla::tags::_NAME_, TL, Array<TR...>>(lhs, rhs)); \
    };                                                                                                         \
    template <typename... TL, typename... TR>                                                                  \
    bool operator _OP_(Array<TL...> const& lhs, Expression<TR...> const& rhs) {                                \
        return calculus::reduction<_REDUCTION_>(                                                               \
            Expression<simpla::tags::_NAME_, Array<TL...>, Expression<TR...>>(lhs, rhs));                      \
    };                                                                                                         \
    template <typename... TL, typename... TR>                                                                  \
    bool operator _OP_(Expression<TL...> const& lhs, Array<TR...> const& rhs) {                                \
        return calculus::reduction<_REDUCTION_>(                                                               \
            Expression<simpla::tags::_NAME_, Expression<TL...>, Array<TR...>>(lhs, rhs));                      \
    };                                                                                                         \
    template <typename... TL, typename... TR>                                                                  \
    bool operator _OP_(Array<TL...> const& lhs, Array<TR...> const& rhs) {                                     \
        return calculus::reduction<_REDUCTION_>(                                                               \
            Expression<simpla::tags::_NAME_, Array<TL...>, Array<TR...>>(lhs, rhs));                           \
    };

_SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR(!=, not_equal_to, simpla::tags::logical_or)
_SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR(==, equal_to, simpla::tags::logical_and)
_SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR(<=, less_equal, simpla::tags::logical_and)
_SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR(>=, greater_equal, simpla::tags::logical_and)
_SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR(<, less, simpla::tags::logical_and)
_SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR(>, greater, simpla::tags::logical_and)
#undef _SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR

}  // namespace simpla{
#endif  // SIMPLA_ARRAY_H
