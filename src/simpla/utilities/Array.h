//
// Created by salmon on 16-12-28.
//

#ifndef SIMPLA_ARRAY_H
#define SIMPLA_ARRAY_H

#include <simpla/SIMPLA_config.h>
#include <initializer_list>
#include <limits>
#include <memory>
#include <tuple>

#include "ExpressionTemplate.h"
#include "Log.h"
#include "SFC.h"
#include "memory.h"
namespace simpla {

template <typename V, typename SFC = ZSFC<3>>
class Array {
   public:
    typedef V value_type;
    typedef typename SFC::array_index_box_type array_index_box_type;

   private:
    typedef Array<value_type, SFC> this_type;
    SFC m_sfc_;
    std::shared_ptr<value_type> m_holder_ = nullptr;
    value_type* m_host_data_ = nullptr;
    value_type* m_data_ = nullptr;

   public:
    Array() = default;
    virtual ~Array() = default;

    Array(this_type const& other)
        : m_sfc_(other.m_sfc_), m_data_(other.m_data_), m_holder_(other.m_holder_), m_host_data_(other.m_host_data_) {}

    Array(this_type&& other) noexcept
        : m_sfc_(other.m_sfc_), m_data_(other.m_data_), m_holder_(other.m_holder_), m_host_data_(other.m_host_data_) {}

    template <typename... Args>
    explicit Array(Args&&... args) : m_data_(nullptr), m_sfc_(std::forward<Args>(args)...) {}

    template <typename... Args>
    explicit Array(value_type* d, Args&&... args) : m_data_(d), m_sfc_(std::forward<Args>(args)...) {}

    void swap(this_type& other) {
        std::swap(m_holder_, other.m_holder_);
        std::swap(m_host_data_, other.m_host_data_);
        std::swap(m_data_, other.m_data_);
        m_sfc_.swap(other.m_sfc_);
    }

    Array& operator=(this_type&& other) {
        this_type(std::forward<this_type>(other)).swap(*this);
        return *this;
    };

    void DoSetUp() {
        if (m_data_ == nullptr) {
            m_holder_ = spMakeShared<value_type>(m_data_, m_sfc_.size());
            m_host_data_ = m_data_;
            m_data_ = m_holder_.get();
        }
    }
    void SetSpaceFillingCurve(SFC const& s) { m_sfc_ = s; }
    SFC const& GetSpaceFillingCurve() const { return m_sfc_; }

    int GetNDIMS() const { return SFC::ndims; }
    bool empty() const { return m_data_ == nullptr; }
    std::type_info const& value_type_info() const { return typeid(value_type); }
    size_type size() const { return m_sfc_.size(); }

    void reset(std::shared_ptr<value_type> const& d = nullptr) {
        m_holder_ = d;
        m_host_data_ = m_holder_.get();
    }

    void SetData(std::shared_ptr<value_type> const& d) {
        m_holder_ = d;
        m_host_data_ = m_holder_.get();
    }

    std::shared_ptr<value_type>& GetData() { return m_holder_; }
    std::shared_ptr<value_type> const& GetData() const { return m_holder_; }

    value_type* get() { return m_data_; }
    value_type* get() const { return m_data_; }

    template <typename Other, typename... Args>
    void Assign(Other const& other, Args&&... args) {
        at(std::forward<Args>(args)...) = this_type::getValue(other, std::forward<Args>(args)...);
    };

    template <typename... Args>
    this_type Shift(Args&&... args) const {
        this_type res(*this);
        res.Shift(std::forward<Args>(args)...);
        return std::move(res);
    }

    this_type operator[](typename SFC::array_index_type const& IX) const { return Shift(IX); }

    void SetUndefined() { Fill(std::numeric_limits<V>::signaling_NaN()); }
    void Clear() { Fill(0); }
    void Fill(value_type v) {
        DoSetUp();
        spMemoryFill(m_data_, v, m_sfc_.size());
    }
    void DeepCopy(value_type const* other) {
        DoSetUp();
        spMemoryCopy(m_data_, other, m_sfc_.size());
    }

    this_type& operator=(this_type const& rhs) {
        DoSetUp();
        m_sfc_.Foreach(*this, [=] __host__ __device__(auto&&... s) {
            return this_type::getValue(rhs, std::forward<decltype(s)>(s)...);
        });
        return (*this);
    }

    template <typename TR>
    this_type& operator=(TR const& rhs) {
        DoSetUp();
        m_sfc_.Foreach(*this, [=] __host__ __device__(auto&&... s) {
            return this_type::getValue(rhs, std::forward<decltype(s)>(s)...);
        });
        return (*this);
    }

    std::ostream& Print(std::ostream& os, int indent = 0) const { return m_sfc_.Print(os, m_holder_.get(), indent); }

    __host__ __device__ value_type& operator[](size_type s) { return m_data_[s]; }

    __host__ __device__ value_type const& operator[](size_type s) const { return m_data_[s]; }

    template <typename... Args>
    __host__ __device__ value_type& at(Args&&... args) {
        return m_data_[m_sfc_.hash(std::forward<Args>(args)...)];
    }
    template <typename... Args>
    __host__ __device__ value_type const& at(Args&&... args) const {
        return m_data_[m_sfc_.hash(std::forward<Args>(args)...)];
    }
    template <typename... Args>
    __host__ __device__ value_type& operator()(Args&&... args) {
        return at(std::forward<Args>(args)...);
    }
    template <typename... Args>
    __host__ __device__ value_type const& operator()(Args&&... args) const {
        return at(std::forward<Args>(args)...);
    }

    template <typename... Args>
    __host__ __device__ static constexpr value_type const& getValue(value_type const& v, Args&&... args) {
        return v;
    };

    template <typename U, typename RSFC, typename... Args>
    __host__ __device__ static constexpr U const& getValue(Array<U, RSFC> const& other, Args&&... args) {
        return other.at(std::forward<Args>(args)...);
    };

    template <typename... Args>
    __host__ __device__ static constexpr value_type const& getValue(this_type const& other, Args&&... args) {
        return other.at(std::forward<Args>(args)...);
    };

    template <typename TOP, typename... Others, size_t... IND, typename... Args>
    __host__ __device__ static constexpr auto _invoke_helper(Expression<TOP, Others...> const& expr,
                                                             std::index_sequence<IND...>, Args&&... args) {
        return TOP::eval(getValue(std::get<IND>(expr.m_args_), std::forward<Args>(args)...)...);
    }

    template <typename TOP, typename... Others, typename... Args>
    __host__ __device__ static constexpr auto getValue(Expression<TOP, Others...> const& expr, Args&&... args) {
        return _invoke_helper(expr, std::index_sequence_for<Others...>(), std::forward<Args>(args)...);
    }
};

namespace traits {
template <typename... T>
struct reference<Array<T...>> {
    typedef Array<T...> const& type;
};
}

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

#define _SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR(_OP_, _NAME_, _REDUCTION_)                                            \
    template <typename... TL, typename TR>                                                                             \
    bool operator _OP_(Array<TL...> const& lhs, TR const& rhs) {                                                       \
        return traits::reduction<_REDUCTION_>(Expression<simpla::tags::_NAME_, Array<TL...>, TR>(lhs, rhs));           \
    };                                                                                                                 \
    template <typename TL, typename... TR>                                                                             \
    bool operator _OP_(TL const& lhs, Array<TR...> const& rhs) {                                                       \
        return traits::reduction<_REDUCTION_>(Expression<simpla::tags::_NAME_, TL, Array<TR...>>(lhs, rhs));           \
    };                                                                                                                 \
    template <typename... TL, typename... TR>                                                                          \
    bool operator _OP_(Array<TL...> const& lhs, Expression<TR...> const& rhs) {                                        \
        return traits::reduction<_REDUCTION_>(                                                                         \
            Expression<simpla::tags::_NAME_, Array<TL...>, Expression<TR...>>(lhs, rhs));                              \
    };                                                                                                                 \
    template <typename... TL, typename... TR>                                                                          \
    bool operator _OP_(Expression<TL...> const& lhs, Array<TR...> const& rhs) {                                        \
        return traits::reduction<_REDUCTION_>(                                                                         \
            Expression<simpla::tags::_NAME_, Expression<TL...>, Array<TR...>>(lhs, rhs));                              \
    };                                                                                                                 \
    template <typename... TL, typename... TR>                                                                          \
    bool operator _OP_(Array<TL...> const& lhs, Array<TR...> const& rhs) {                                             \
        return traits::reduction<_REDUCTION_>(Expression<simpla::tags::_NAME_, Array<TL...>, Array<TR...>>(lhs, rhs)); \
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
