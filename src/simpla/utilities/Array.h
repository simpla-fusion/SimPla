//
// Created by salmon on 16-12-28.
//

#ifndef SIMPLA_ARRAY_H
#define SIMPLA_ARRAY_H

#include <simpla/SIMPLA_config.h>
#include <cfenv>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include "ExpressionTemplate.h"
#include "FancyStream.h"
#include "Log.h"
#include "Range.h"
#include "SFC.h"
#include "memory.h"
#include "nTuple.h"
namespace simpla {
typedef nTuple<index_type, 3> IdxShift;
template <typename V, int NDIMS, typename SFC = ZSFC<NDIMS>>
struct Array {
   private:
    typedef Array<V, NDIMS, SFC> this_type;

   public:
    typedef V value_type;
    typedef typename SFC::array_index_type array_index_type;

    static const int ndims = NDIMS;

    typedef std::tuple<array_index_type, array_index_type> array_index_box_type;

   private:
    SFC m_sfc_;
    std::shared_ptr<value_type> m_holder_ = nullptr;
    value_type* m_data_ = nullptr;
    static value_type m_snan_;
    static value_type m_null_;

   public:
    Array() = default;

    Array(this_type const& other) : m_sfc_(other.m_sfc_), m_data_(other.m_data_), m_holder_(other.m_holder_) {}

    Array(this_type&& other) noexcept : m_sfc_(other.m_sfc_), m_data_(other.m_data_), m_holder_(other.m_holder_) {}

    Array& operator=(this_type&& other) = delete;

    Array(std::initializer_list<index_type> const& l) : m_sfc_(l), m_holder_(nullptr), m_data_(nullptr) { DoSetUp(); }

    template <typename... Args>
    explicit Array(Args&&... args) : m_sfc_(std::forward<Args>(args)...), m_holder_(nullptr), m_data_(nullptr) {
        DoSetUp();
    }

    template <typename... Args>
    explicit Array(std::shared_ptr<value_type> const& d, Args&&... args)
        : m_sfc_(std::forward<Args>(args)...), m_holder_(d), m_data_(d.get()) {}

    virtual ~Array() = default;

    void swap(this_type& other) {
        std::swap(m_holder_, other.m_holder_);
        std::swap(m_data_, other.m_data_);
        m_sfc_.swap(other.m_sfc_);
    }
    void DoSetUp() {
        if (m_holder_ == nullptr && m_sfc_.size() > 0) {
            m_holder_ = spMakeSharedArray<value_type>(m_sfc_.size());
            m_data_ = m_holder_.get();
            //#ifdef SIMPLA_INITIALIZE_ARRAY_TO_SIGNALING_NAN
            //            spMemoryFill(m_data_, m_snan_, size());
            //#endif
        }
        m_data_ = m_holder_.get();
    }

    void Clear() { Fill(0); }
    void Fill(value_type v) {
        DoSetUp();
        spMemoryFill(m_holder_.get(), v, size());
    }
    template <typename TOther>
    void DeepCopy(TOther const& other) {
        m_sfc_.Foreach([&](array_index_type const& idx) { at(idx) = getValue(other, idx); });
    }

    std::ostream& Print(std::ostream& os, int indent = 0) const { return m_sfc_.Print(os, m_data_, indent); }

    this_type operator()(array_index_type const& IX) const {
        this_type res(*this);
        res.Shift(IX);
        return res;
    }

    void Shift(array_index_type const& offset) { m_sfc_.Shfit(offset); }

    SFC const& GetSpaceFillingCurve() const { return m_sfc_; }

    int GetNDIMS() const { return NDIMS; }
    bool empty() const { return m_data_ == nullptr; }
    std::type_info const& value_type_info() const { return typeid(value_type); }
    size_type size() const { return m_sfc_.size(); }

    array_index_box_type const& GetIndexBox() const { return m_sfc_.GetIndexBox(); }

    void SetData(std::shared_ptr<value_type> const& d) const { m_data_ = d; }
    std::shared_ptr<value_type>& GetData() { return m_holder_; }
    std::shared_ptr<value_type> const& GetData() const { return m_holder_; }
    void* GetRawPointer() { return m_holder_.get(); }
    void* GetRawPointer() const { return m_holder_.get(); }

    template <typename... Args>
    __host__ __device__ value_type& operator()(Args&&... args) {
        return at(std::forward<Args>(args)...);
    }
    template <typename... Args>
    __host__ __device__ value_type const& operator()(Args&&... args) const {
        return at(std::forward<Args>(args)...);
    }
    template <typename... Args>
    __host__ __device__ value_type& at(Args&&... args) {
#ifdef ENABLE_BOUND_CHECK
        auto s = m_sfc_.hash(std::forward<Args>(args)...);
        return (s < m_size_) ? m_data_[s] : m_null_;
#else
        return m_data_[m_sfc_.hash(std::forward<Args>(args)...)];
#endif
    }
    template <typename... Args>
    __host__ __device__ value_type const& at(Args&&... args) const {
#ifdef ENABLE_BOUND_CHECK
        auto s = m_sfc_.hash(std::forward<Args>(args)...);
        return (s < m_size_) ? m_data_[s] : m_snan_;
#else
        return m_data_[m_sfc_.hash(std::forward<Args>(args)...)];
#endif
    }
    template <typename TIdx>
    __host__ __device__ value_type& operator[](TIdx const& idx) {
        return m_data_[m_sfc_.hash(idx)];
    }
    template <typename TIdx>
    __host__ __device__ value_type const& operator[](TIdx const& idx) const {
        return m_data_[m_sfc_.hash(idx)];
    }

    this_type& operator=(this_type const& rhs) {
        DoSetUp();
        m_sfc_.Foreach([&](array_index_type const& idx) { at(idx) = getValue(rhs, idx); });
        show_fe_exceptions();
        return (*this);
    }

    template <typename TR>
    this_type& operator=(TR const& rhs) {
        DoSetUp();
        m_sfc_.Foreach([&](array_index_type const& idx) { at(idx) = getValue(rhs, idx); });
        show_fe_exceptions();
        return (*this);
    }

    template <typename TExpr>
    void Assign(array_index_type const& idx, TExpr const& expr) {
        if (m_sfc_.in_box(idx)) { at(idx) = getValue(expr, idx); }
    }

   public:
    template <typename TOP, typename... Others>
    void Foreach(TOP const& op, Others&&... others) {
        DoSetUp();
        m_sfc_.Foreach(
            [&](array_index_type const& idx) { op(at(idx), getValue(std::forward<Others>(others), idx)...); });
    };

    template <typename TFun>
    void Foreach(TFun const& op,
                 ENABLE_IF(simpla::concept::is_callable<TFun(array_index_type const&, value_type&)>::value)) {
        DoSetUp();
        m_sfc_.Foreach([&](array_index_type const& idx) { op(idx, at(idx)); });
    };
    template <typename TFun>
    void Foreach(
        TFun const& op,
        ENABLE_IF(simpla::concept::is_callable<TFun(array_index_type const&, value_type const&)>::value)) const {
        m_sfc_.Foreach([&](array_index_type const& idx) { op(idx, at(idx)); });
    };

   public:
    template <typename TFun>
    __host__ __device__ constexpr value_type getValue(
        TFun const& op, array_index_type const& s,
        ENABLE_IF(simpla::concept::is_callable<TFun(array_index_type const&)>::value)) {
        return op(s);
    };
    __host__ __device__ constexpr value_type const& getValue(value_type const& v, array_index_type const& s) {
        return v;
    };

    template <typename U, typename RSFC>
    __host__ __device__ constexpr auto getValue(Array<U, NDIMS, RSFC> const& v, array_index_type const& s) {
        return v[s];
    };

    __host__ __device__ constexpr auto getValue(this_type& self, array_index_type const& s) { return self.at(s); };
    __host__ __device__ constexpr auto getValue(this_type const& self, array_index_type const& s) {
        return self.at(s);
    };

    template <typename TOP, typename... Others, size_t... IND>
    __host__ __device__ constexpr auto _invoke_helper(Expression<TOP, Others...> const& expr,
                                                      std::index_sequence<IND...>, array_index_type const& s) {
        return TOP::eval(getValue(std::get<IND>(expr.m_args_), s)...);
    }

    template <typename TOP, typename... Others>
    __host__ __device__ constexpr auto getValue(Expression<TOP, Others...> const& expr, array_index_type const& s) {
        return _invoke_helper(expr, std::index_sequence_for<Others...>(), s);
    }

   private:
#pragma STDC_FENV_ACCESS on
    void show_fe_exceptions() {
        if (std::fetestexcept(FE_ALL_EXCEPT) & FE_INVALID) { RUNTIME_ERROR << ("FE_INVALID is raised") << std::endl; }
        std::feclearexcept(FE_ALL_EXCEPT);
    }
};

template <typename V, int NDIMS, typename SFC>
V Array<V, NDIMS, SFC>::m_snan_ = std::numeric_limits<V>::signaling_NaN();
template <typename V, int NDIMS, typename SFC>
V Array<V, NDIMS, SFC>::m_null_ = 0;

namespace traits {
template <typename T, int N, typename SFC>
struct reference<Array<T, N, SFC>> {
    typedef Array<T, N, SFC> const& type;
};
}
// template <typename V, int NDIMS, typename SFC>
// nTuple<V, 3> Array<nTuple<V, 3>, NDIMS, SFC>::m_snan_{std::numeric_limits<V>::signaling_NaN(),
//                                                      std::numeric_limits<V>::signaling_NaN(),
//                                                      std::numeric_limits<V>::signaling_NaN()};
// template <typename V, int NDIMS, typename SFC>
// nTuple<V, 3> Array<nTuple<V, 3>, NDIMS, SFC>::m_null_{0, 0, 0};

template <typename TL, int NL, typename SFC>
std::ostream& operator<<(std::ostream& os, Array<TL, NL, SFC> const& lhs) {
    return lhs.Print(os, 0);
};

template <typename TL, int NL, typename SFC>
std::istream& operator>>(std::istream& is, Array<TL, NL, SFC>& lhs) {
    UNIMPLEMENTED;
    return is;
};

#define _SP_DEFINE_ARRAY_BINARY_OPERATOR(_OP_, _NAME_)                                             \
    template <typename TL, int NL, typename TR, typename SFC>                                      \
    auto operator _OP_(Array<TL, NL, SFC> const& lhs, TR const& rhs) {                             \
        return Expression<simpla::tags::_NAME_, Array<TL, NL, SFC>, TR>(lhs, rhs);                 \
    };                                                                                             \
    template <typename TL, typename TR, int NR, typename SFC>                                      \
    auto operator _OP_(TL const& lhs, Array<TR, NR, SFC> const& rhs) {                             \
        return Expression<simpla::tags::_NAME_, TL, Array<TR, NR, SFC>>(lhs, rhs);                 \
    };                                                                                             \
    template <typename TL, int NL, typename... TR, typename SFC>                                   \
    auto operator _OP_(Array<TL, NL, SFC> const& lhs, Expression<TR...> const& rhs) {              \
        return Expression<simpla::tags::_NAME_, Array<TL, NL, SFC>, Expression<TR...>>(lhs, rhs);  \
    };                                                                                             \
    template <typename... TL, typename TR, int NR, typename SFC>                                   \
    auto operator _OP_(Expression<TL...> const& lhs, Array<TR, NR, SFC> const& rhs) {              \
        return Expression<simpla::tags::_NAME_, Expression<TL...>, Array<TR, NR, SFC>>(lhs, rhs);  \
    };                                                                                             \
    template <typename TL, int NL, typename TR, int NR, typename SFC>                              \
    auto operator _OP_(Array<TL, NL, SFC> const& lhs, Array<TR, NR, SFC> const& rhs) {             \
        return Expression<simpla::tags::_NAME_, Array<TL, NL, SFC>, Array<TR, NR, SFC>>(lhs, rhs); \
    };

#define _SP_DEFINE_ARRAY_UNARY_OPERATOR(_OP_, _NAME_)                     \
    template <typename TL, int NL, typename SFC>                          \
    auto operator _OP_(Array<TL, NL, SFC> const& lhs) {                   \
        return Expression<simpla::tags::_NAME_, Array<TL, NL, SFC>>(lhs); \
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

template <typename TL, int NL, typename SFC>
auto operator<<(Array<TL, NL, SFC> const& lhs, int n) {
    return Expression<simpla::tags::bitwise_left_shift, Array<TL, NL>, int>(lhs, n);
};

template <typename TL, int NL, typename SFC>
auto operator>>(Array<TL, NL, SFC> const& lhs, int n) {
    return Expression<simpla::tags::bitwise_right_shifit, Array<TL, NL>, int>(lhs, n);
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

#define _SP_DEFINE_ARRAY_BINARY_FUNCTION(_NAME_)                                                     \
    template <typename TL, int NL, typename TR, typename SFC>                                        \
    auto _NAME_(Array<TL, NL, SFC> const& lhs, TR const& rhs) {                                      \
        return Expression<simpla::tags::_NAME_, Array<TL, NL, SFC>, TR>(lhs, rhs);                   \
    };                                                                                               \
    template <typename TL, typename TR, int NR, typename SFC>                                        \
    auto _NAME_(TL const& lhs, Array<TR, NR, SFC> const& rhs) {                                      \
        return Expression<simpla::tags::_NAME_, TL, Array<TR, NR, SFC>>(lhs, rhs);                   \
    };                                                                                               \
    template <typename TL, int NL, typename TR, typename SFC>                                        \
    auto _NAME_(Array<TL, NL, SFC> const& lhs, Expression<TR> const& rhs) {                          \
        return Expression<simpla::tags::_NAME_, Array<TL, NL, SFC>, Expression<TR>>(lhs, rhs);       \
    };                                                                                               \
    template <typename TL, typename TR, int NR, typename SFC>                                        \
    auto _NAME_(Expression<TL> const& lhs, Array<TR, NR, SFC> const& rhs) {                          \
        return Expression<simpla::tags::_NAME_, Expression<TL>, Array<TR, NR, SFC>>(lhs, rhs);       \
    };                                                                                               \
    template <typename TL, int NL, typename TR, int NR, typename SFCL, typename SFCR>                \
    auto _NAME_(Array<TL, NL, SFCL> const& lhs, Array<TR, NR, SFCR> const& rhs) {                    \
        return Expression<simpla::tags::_NAME_, Array<TL, NL, SFCL>, Array<TR, NR, SFCR>>(lhs, rhs); \
    };

#define _SP_DEFINE_ARRAY_UNARY_FUNCTION(_NAME_)                         \
    template <typename T, int N, typename SFC>                          \
    auto _NAME_(Array<T, N, SFC> const& lhs) {                          \
        return Expression<simpla::tags::_NAME_, Array<T, N, SFC>>(lhs); \
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

#define _SP_DEFINE_ARRAY_COMPOUND_OP(_OP_)                                                        \
    template <typename TL, int NL, typename TR, typename SFC>                                     \
    Array<TL, NL, SFC>& operator _OP_##=(Array<TL, NL, SFC>& lhs, TR const& rhs) {                \
        lhs = lhs _OP_ rhs;                                                                       \
        return lhs;                                                                               \
    }                                                                                             \
    template <typename TL, int NL, typename... TR, typename SFC>                                  \
    Array<TL, NL, SFC>& operator _OP_##=(Array<TL, NL, SFC>& lhs, Expression<TR...> const& rhs) { \
        lhs = lhs _OP_ rhs;                                                                       \
        return lhs;                                                                               \
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

#define _SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR(_OP_, _NAME_, _REDUCTION_)                                        \
    template <typename TL, int NL, typename TR, typename SFC>                                                      \
    bool operator _OP_(Array<TL, NL, SFC> const& lhs, TR const& rhs) {                                             \
        return traits::reduction<_REDUCTION_>(Expression<simpla::tags::_NAME_, Array<TL, NL, SFC>, TR>(lhs, rhs)); \
    };                                                                                                             \
    template <typename TL, typename TR, int NR, typename SFC>                                                      \
    bool operator _OP_(TL const& lhs, Array<TR, NR, SFC> const& rhs) {                                             \
        return traits::reduction<_REDUCTION_>(Expression<simpla::tags::_NAME_, TL, Array<TR, NR, SFC>>(lhs, rhs)); \
    };                                                                                                             \
    template <typename TL, int NL, typename... TR, typename SFC>                                                   \
    bool operator _OP_(Array<TL, NL, SFC> const& lhs, Expression<TR...> const& rhs) {                              \
        return traits::reduction<_REDUCTION_>(                                                                     \
            Expression<simpla::tags::_NAME_, Array<TL, NL, SFC>, Expression<TR...>>(lhs, rhs));                    \
    };                                                                                                             \
    template <typename... TL, typename TR, int NR, typename SFC>                                                   \
    bool operator _OP_(Expression<TL...> const& lhs, Array<TR, NR, SFC> const& rhs) {                              \
        return traits::reduction<_REDUCTION_>(                                                                     \
            Expression<simpla::tags::_NAME_, Expression<TL...>, Array<TR, NR, SFC>>(lhs, rhs));                    \
    };                                                                                                             \
    template <typename TL, int NL, typename TR, int NR, typename SFCL, typename SFCR>                              \
    bool operator _OP_(Array<TL, NL, SFCL> const& lhs, Array<TR, NR, SFCR> const& rhs) {                           \
        return traits::reduction<_REDUCTION_>(                                                                     \
            Expression<simpla::tags::_NAME_, Array<TL, NL, SFCL>, Array<TR, NR, SFCR>>(lhs, rhs));                 \
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
