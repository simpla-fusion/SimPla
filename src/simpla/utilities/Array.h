//
// Created by salmon on 16-12-28.
//

#ifndef SIMPLA_ARRAY_H
#define SIMPLA_ARRAY_H

#include <simpla/concept/Printable.h>
#include <cstring>
#include "ExpressionTemplate.h"
#include "FancyStream.h"
#include "Log.h"
#include "MemoryPool.h"
#include "Range.h"
#include "nTuple.h"
#include "sp_def.h"
namespace simpla {

struct ArrayIndexShift {
    int dim_num = 0;
    index_type value = 0;
};
inline ArrayIndexShift operator+(ArrayIndexShift const& l, index_type s) {
    return ArrayIndexShift{l.dim_num, l.value - s};
}
inline ArrayIndexShift operator-(ArrayIndexShift const& l, index_type s) {
    return ArrayIndexShift{l.dim_num, l.value + s};
}

static const ArrayIndexShift I{0, 0};
static const ArrayIndexShift J{1, 0};
static const ArrayIndexShift K{2, 0};

namespace detail {
template <int N, typename TFun>
void ForeachND(std::tuple<nTuple<index_type, N>, nTuple<index_type, N>> const& inner_box, TFun const& fun) {
    UNIMPLEMENTED;
    //    nTuple<index_type, N> idx;
    //    idx = std::get<0>(inner_box);
    //
    //    while (1) {
    //        fun(idx);
    //
    //        ++idx[N - 1];
    //        for (int rank = N - 1; rank > 0; --rank) {
    //            if (idx[rank] >= std::get<1>(inner_box)[rank]) {
    //                idx[rank] = std::get<0>(inner_box)[rank];
    //                ++idx[rank - 1];
    //            }
    //        }
    //        if (idx[0] >= std::get<1>(inner_box)[0]) break;
    //    }
}

template <typename TFun>
void ForeachND(std::tuple<nTuple<index_type, 1>, nTuple<index_type, 1>> const& inner_box, TFun const& fun,
               bool is_fast_first = false) {
    index_type ib = std::get<0>(inner_box)[0];
    index_type ie = std::get<1>(inner_box)[0];
#pragma omp parallel for
    for (index_type i = ib; i < ie; ++i) { fun(nTuple<index_type, 1>{i}); }
}
template <typename TFun>
void ForeachND(std::tuple<nTuple<index_type, 2>, nTuple<index_type, 2>> const& inner_box, TFun const& fun,
               bool is_fast_first = false) {
    index_type ib = std::get<0>(inner_box)[0];
    index_type ie = std::get<1>(inner_box)[0];
    index_type jb = std::get<0>(inner_box)[1];
    index_type je = std::get<1>(inner_box)[1];
    if (is_fast_first) {
#pragma omp parallel for
        for (index_type j = jb; j < je; ++j)
            for (index_type i = ib; i < ie; ++i) { fun(nTuple<index_type, 2>{i, j}); }
    } else {
#pragma omp parallel for
        for (index_type i = ib; i < ie; ++i)
            for (index_type j = jb; j < je; ++j) { fun(nTuple<index_type, 2>{i, j}); }
    }
}

template <typename TFun>
void ForeachND(std::tuple<nTuple<index_type, 3>, nTuple<index_type, 3>> const& inner_box, TFun const& fun,
               bool is_fast_first = false) {
    index_type ib = std::get<0>(inner_box)[0];
    index_type ie = std::get<1>(inner_box)[0];
    index_type jb = std::get<0>(inner_box)[1];
    index_type je = std::get<1>(inner_box)[1];
    index_type kb = std::get<0>(inner_box)[2];
    index_type ke = std::get<1>(inner_box)[2];

    if (is_fast_first) {
        //#pragma omp parallel for
        for (index_type k = kb; k < ke; ++k)
            for (index_type j = jb; j < je; ++j)
                for (index_type i = ib; i < ie; ++i) { fun(nTuple<index_type, 3>{i, j, k}); }

    } else {
        //#pragma omp parallel for
        for (index_type i = ib; i < ie; ++i)
            for (index_type j = jb; j < je; ++j)
                for (index_type k = kb; k < ke; ++k) { fun(nTuple<index_type, 3>{i, j, k}); }
    }
}

template <typename TFun>
void ForeachND(std::tuple<nTuple<index_type, 4>, nTuple<index_type, 4>> const& inner_box, TFun const& fun,
               bool is_fast_first = false) {
    index_type ib = std::get<0>(inner_box)[0];
    index_type ie = std::get<1>(inner_box)[0];
    index_type jb = std::get<0>(inner_box)[1];
    index_type je = std::get<1>(inner_box)[1];
    index_type kb = std::get<0>(inner_box)[2];
    index_type ke = std::get<1>(inner_box)[2];
    index_type lb = std::get<0>(inner_box)[3];
    index_type le = std::get<1>(inner_box)[3];

    if (is_fast_first) {
#pragma omp parallel for
        for (index_type l = lb; l < le; ++l)
            for (index_type k = kb; k < ke; ++k)
                for (index_type j = jb; j < je; ++j)
                    for (index_type i = ib; i < ie; ++i) fun(nTuple<index_type, 4>{i, j, k, l});

    } else {
#pragma omp parallel for
        for (index_type i = ib; i < ie; ++i)
            for (index_type j = jb; j < je; ++j)
                for (index_type k = kb; k < ke; ++k)
                    for (index_type l = lb; l < le; ++l) { fun(nTuple<index_type, 4>{i, j, k, l}); }
    }
}
template <typename TIdx>
size_type Hash(std::tuple<nTuple<index_type, 2>, nTuple<index_type, 2>> const& box, TIdx const& idx) {
    return static_cast<size_type>(((idx[1] - std::get<0>(box)[1]) +
                                   (idx[0] - std::get<0>(box)[0]) * (std::get<1>(box)[1] - std::get<0>(box)[1])));
};
template <typename TIdx>
size_type Hash(std::tuple<nTuple<index_type, 3>, nTuple<index_type, 3>> const& box, TIdx const& idx) {
    return static_cast<size_type>((idx[2] - std::get<0>(box)[2]) +
                                  ((idx[1] - std::get<0>(box)[1]) +
                                   (idx[0] - std::get<0>(box)[0]) * (std::get<1>(box)[1] - std::get<0>(box)[1])) *
                                      (std::get<1>(box)[2] - std::get<0>(box)[2]));
};
template <typename TIdx>
size_type Hash(std::tuple<nTuple<index_type, 4>, nTuple<index_type, 4>> const& box, TIdx const& idx) {
    return static_cast<size_type>((idx[3] - std::get<0>(box)[3]) +
                                  ((idx[2] - std::get<0>(box)[2]) +
                                   ((idx[1] - std::get<0>(box)[1]) +
                                    (idx[0] - std::get<0>(box)[0]) * (std::get<1>(box)[1] - std::get<0>(box)[1])) *
                                       (std::get<1>(box)[2] - std::get<0>(box)[2])) *
                                      (std::get<1>(box)[3] - std::get<0>(box)[3]));
};

template <int N, typename TIdx>
size_type Hash(std::tuple<nTuple<index_type, N>, nTuple<index_type, N>> const& box, TIdx const& idx) {
    size_type res = idx[0] - std::get<0>(box)[0];
    for (int i = 1; i < N; ++i) {
        res *= (std::get<1>(box)[i - 1] - std::get<0>(box)[i - 1]);
        res += idx[i] - std::get<0>(box)[i];
    }
    return res;
};
};

template <typename V, int NDIMS>
struct Array : public concept::Printable {
   private:
    typedef Array<V, NDIMS> this_type;

   public:
    typedef V value_type;
    static const int ndims = NDIMS;
    typedef std::true_type prefer_pass_by_reference;
    typedef std::false_type is_expression;
    typedef std::true_type is_array;

    typedef nTuple<index_type, NDIMS> m_index_tuple;
    typedef std::tuple<m_index_tuple, m_index_tuple> m_index_box_type;

   private:
    std::shared_ptr<value_type> m_data_ = nullptr;
    bool m_array_order_fast_first_ = false;
    m_index_box_type m_index_box_;

    index_type m_offset_ = 0;
    m_index_tuple m_strides_;

    size_type m_size_ = 0;

   public:
    Array() {}

    Array(this_type const& other)
        : m_data_(other.m_data_),
          m_index_box_(other.m_index_box_),
          m_array_order_fast_first_(other.m_array_order_fast_first_) {
        SetUp();
    }

    explicit Array(std::initializer_list<index_type> const& l) {
        for (int i = 0; i < NDIMS; ++i) {
            std::get<0>(m_index_box_)[i] = 0;
            std::get<1>(m_index_box_)[i] = 1;
        }
        int count = 0;
        for (auto const& v : l) {
            if (count >= NDIMS) { break; }
            std::get<1>(m_index_box_)[count] = v;
            ++count;
        }
        std::get<0>(m_index_box_) = std::get<0>(m_index_box_);
        std::get<1>(m_index_box_) = std::get<1>(m_index_box_);
        SetUp();
    }
    explicit Array(m_index_box_type const& b, std::shared_ptr<value_type> const& d = nullptr,
                   bool array_order_fast_first = false)
        : m_index_box_(b), m_data_(d), m_array_order_fast_first_(array_order_fast_first) {
        SetUp();
    }

    void SetUp() {
        if (m_array_order_fast_first_) {
            m_strides_[0] = 1;
            m_offset_ = -std::get<0>(m_index_box_)[0];
            for (int i = 1; i < NDIMS; ++i) {
                m_strides_[i] =
                    m_strides_[i - 1] * (std::get<1>(m_index_box_)[i - 1] - std::get<0>(m_index_box_)[i - 1]);
                m_offset_ -= std::get<0>(m_index_box_)[i] * m_strides_[i];
            }
        } else {
            m_strides_[NDIMS - 1] = 1;
            m_offset_ = -std::get<0>(m_index_box_)[NDIMS - 1];
            for (int i = NDIMS - 2; i >= 0; --i) {
                m_strides_[i] =
                    m_strides_[i + 1] * (std::get<1>(m_index_box_)[i + 1] - std::get<0>(m_index_box_)[i + 1]);
                m_offset_ -= std::get<0>(m_index_box_)[i] * m_strides_[i];
            }
        }
        m_size_ = 1;
        for (int i = 0; i < NDIMS; ++i) { m_size_ *= (std::get<1>(m_index_box_)[i] - std::get<0>(m_index_box_)[i]); }
        if (m_data_ == nullptr && m_size_ > 0) { m_data_ = sp_alloc_array<value_type>(m_size_); }
    }

    template <typename... U>
    Array(Expression<U...> const& expr) {
        Foreach(simpla::tags::_assign(), expr);
    }

    virtual ~Array() {}
    void swap(this_type& other) {
        std::swap(m_data_, other.m_data_);
        std::swap(m_array_order_fast_first_, other.m_array_order_fast_first_);
        std::swap(m_index_box_, other.m_index_box_);
        SetUp();
        other.SetUp();
    };

    template <typename... Others>
    this_type operator()(ArrayIndexShift const& IX, Others&&... others) const {
        this_type res(*this);
        res.Shift(IX, std::forward<Others>(others)...);
        return std::move(res);
    }
    void Shift(ArrayIndexShift const& IX) {
        std::get<0>(m_index_box_)[IX.dim_num] += IX.value;
        std::get<1>(m_index_box_)[IX.dim_num] += IX.value;
        SetUp();
    }
    template <typename... Others>
    void Shift(ArrayIndexShift const& IX, Others&&... others) {
        Shift(IX);
        Shift(std::forward<Others>(others)...);
    }
    void Shift(m_index_tuple const& offset) {
        std::get<0>(m_index_box_) += offset;
        std::get<1>(m_index_box_) += offset;
        SetUp();
    }
    virtual bool empty() const { return m_data_ == nullptr; }
    virtual std::type_info const& value_type_info() const { return typeid(value_type); }
    virtual int GetNDIMS() const { return NDIMS; }

    m_index_box_type const& GetIndexBox() const { return m_index_box_; }

    std::shared_ptr<value_type>& GetPointer() { return m_data_; }
    std::shared_ptr<value_type> const& GetPointer() const { return m_data_; }
    virtual void* GetRawPointer() { return m_data_.get(); }
    virtual void* GetRawPointer() const { return m_data_.get(); }

    void SetData(std::shared_ptr<value_type> const& d) const { m_data_ = d; }

    //    size_type hash(m_index_tuple const& idx) const {
    //        size_type s = 0;
    //        for (int i = 0; i < NDIMS; ++i) {
    //            //            if (idx[i] >= std::get<1>(m_index_box_)[i] || idx[i] < std::get<0>(m_index_box_)[i]) {
    //            //                OUT_OF_RANGE << idx[0] << "," << idx[1] << "," << idx[2] << " in " << m_index_box_
    //            <<
    //            //                std::endl;
    //            //            }
    //            s += ((idx[i] - std::get<0>(m_index_box_)[i]) %
    //                  (std::get<1>(m_index_box_)[i] - std::get<0>(m_index_box_)[i])) *
    //                 m_strides_[i];
    //        }
    //
    //        return s;
    //    }

    value_type& at(m_index_tuple const& idx) { return m_data_.get()[dot(m_strides_, idx) + m_offset_]; }

    value_type const& at(m_index_tuple const& idx) const {
        return m_data_.get()[(dot(m_strides_, idx) + m_offset_) % m_size_];
    }

    value_type& operator[](m_index_tuple const& idx) { return at(idx); }

    value_type const& operator[](m_index_tuple const& idx) const { return at(idx); }

    template <typename... TID>
    value_type& at(index_type i0, TID&&... s) {
        return at(m_index_tuple{i0, std::forward<TID>(s)...});
    }
    template <typename... TID>
    value_type const& at(index_type i0, TID&&... s) const {
        return at(m_index_tuple{i0, std::forward<TID>(s)...});
    }
    template <typename... TID>
    value_type& operator()(index_type ix, TID&&... s) {
        return at(m_index_tuple{ix, std::forward<TID>(s)...});
    }

    template <typename... TID>
    value_type const& operator()(index_type idx, TID&&... s) const {
        return at(m_index_tuple{idx, std::forward<TID>(s)...});
    }

    this_type& operator=(this_type const& rhs) {
        Foreach(simpla::tags::_assign(), rhs);
        return (*this);
    }

    template <typename TR>
    this_type& operator=(TR const& rhs) {
        Foreach(simpla::tags::_assign(), rhs);
        return (*this);
    }

    size_type size() const { return m_size_; }

    void Clear() {
        SetUp();
        memset(m_data_.get(), 0, size() * sizeof(value_type));
    }

    std::ostream& Print(std::ostream& os, int indent = 0) const {
        os << "Print Array " << m_index_box_ << std::endl;

        simpla::detail::ForeachND(m_index_box_, [&](m_index_tuple const& idx) {
            if (idx[NDIMS - 1] == std::get<0>(m_index_box_)[NDIMS - 1]) {
                os << "{" << at(idx);
            } else {
                os << "," << at(idx);
            }
            if (idx[NDIMS - 1] == std::get<1>(m_index_box_)[NDIMS - 1] - 1) { os << "}" << std::endl; }
        });

        return os;
    }

   private:
   public:
    template <typename TOP, typename... Others>
    void Foreach(TOP const& op, Others&&... others) {
        if (size() <= 0) { return; }
        simpla::detail::ForeachND(m_index_box_, [&](m_index_tuple const& idx) {
            op(at(idx), getValue(std::forward<Others>(others), idx)...);
        });
    };

    template <typename TFun>
    void Foreach(TFun const& op,
                 ENABLE_IF(simpla::concept::is_callable<TFun(m_index_tuple const&, value_type&)>::value)) {
        if (size() <= 0) { return; }
        SetUp();
        simpla::detail::ForeachND(m_index_box_, [&](m_index_tuple const& idx) { op(idx, at(idx)); });
    };
    template <typename TFun>
    void Foreach(TFun const& op,
                 ENABLE_IF(simpla::concept::is_callable<TFun(m_index_tuple const&, value_type const&)>::value)) const {
        if (size() <= 0) { return; }
        simpla::detail::ForeachND(m_index_box_, [&](m_index_tuple const& idx) { op(idx, at(idx)); });
    };

   public:
    static constexpr value_type const& getValue(value_type const& v, m_index_tuple const& s) { return v; };
    static constexpr decltype(auto) getValue(this_type& self, m_index_tuple const& s) { return self.at(s); };
    static constexpr decltype(auto) getValue(this_type const& self, m_index_tuple const& s) { return self.at(s); };

    template <typename TOP, typename... Others, int... IND>
    static constexpr decltype(auto) _invoke_helper(Expression<TOP, Others...> const& expr, int_sequence<IND...>,
                                                   m_index_tuple const& s) {
        return TOP::eval(getValue(std::get<IND>(expr.m_args_), s)...);
    }

    template <typename TOP, typename... Others>
    static constexpr decltype(auto) getValue(Expression<TOP, Others...> const& expr, m_index_tuple const& s) {
        return _invoke_helper(expr, int_sequence_for<Others...>(), s);
    }
};

#define _SP_DEFINE_ARRAY_BINARY_OPERATOR(_NAME_, _OP_)                                                   \
    template <typename TL, int NL, typename TR>                                                          \
    auto operator _OP_(Array<TL, NL> const& lhs, TR const& rhs) {                                        \
        return Expression<simpla::tags::_NAME_, const Array<TL, NL>, TR const>(lhs, rhs);                \
    };                                                                                                   \
    template <typename TL, typename TR, int NR>                                                          \
    auto operator _OP_(TL const& lhs, Array<TR, NR> const& rhs) {                                        \
        return Expression<simpla::tags::_NAME_, TL const, const Array<TR, NR>>(lhs, rhs);                \
    };                                                                                                   \
    template <typename TL, int NL, typename... TR>                                                       \
    auto operator _OP_(Array<TL, NL> const& lhs, Expression<TR...> const& rhs) {                         \
        return Expression<simpla::tags::_NAME_, const Array<TL, NL>, Expression<TR...> const>(lhs, rhs); \
    };                                                                                                   \
    template <typename... TL, typename TR, int NR>                                                       \
    auto operator _OP_(Expression<TL...> const& lhs, Array<TR, NR> const& rhs) {                         \
        return Expression<simpla::tags::_NAME_, Expression<TL...> const, const Array<TR, NR>>(lhs, rhs); \
    };                                                                                                   \
    template <typename TL, int NL, typename TR, int NR>                                                  \
    auto operator _OP_(Array<TL, NL> const& lhs, Array<TR, NR> const& rhs) {                             \
        return Expression<simpla::tags::_NAME_, const Array<TL, NL>, const Array<TR, NR>>(lhs, rhs);     \
    };

#define _SP_DEFINE_ARRAY_UNARY_OPERATOR(_NAME_, _OP_)                      \
    template <typename TL, int NL>                                         \
    auto operator _OP_(Array<TL, NL> const& lhs) {                         \
        return Expression<simpla::tags::_NAME_, const Array<TL, NL>>(lhs); \
    };

_SP_DEFINE_ARRAY_BINARY_OPERATOR(addition, +)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(subtraction, -)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(multiplication, *)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(division, /)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(modulo, %)

_SP_DEFINE_ARRAY_UNARY_OPERATOR(bitwise_not, ~)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(bitwise_xor, ^)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(bitwise_and, &)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(bitwise_or, |)

template <typename TL, int NL>
auto operator<<(Array<TL, NL> const& lhs, int n) {
    return Expression<simpla::tags::bitwise_left_shift, const Array<TL, NL>, int>(lhs, n);
};

template <typename TL, int NL>
auto operator>>(Array<TL, NL> const& lhs, int n) {
    return Expression<simpla::tags::bitwise_right_shifit, const Array<TL, NL>, int>(lhs, n);
};
//_SP_DEFINE_ARRAY_BINARY_OPERATOR(bitwise_left_shift, <<)
//_SP_DEFINE_ARRAY_BINARY_OPERATOR(bitwise_right_shift, >>)

_SP_DEFINE_ARRAY_UNARY_OPERATOR(unary_plus, +)
_SP_DEFINE_ARRAY_UNARY_OPERATOR(unary_minus, -)

_SP_DEFINE_ARRAY_UNARY_OPERATOR(logical_not, !)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(logical_and, &&)
_SP_DEFINE_ARRAY_BINARY_OPERATOR(logical_or, ||)

#undef _SP_DEFINE_ARRAY_BINARY_OPERATOR
#undef _SP_DEFINE_ARRAY_UNARY_OPERATOR

#define _SP_DEFINE_ARRAY_BINARY_FUNCTION(_NAME_)                                                      \
    template <typename TL, int NL, typename TR>                                                       \
    auto _NAME_(Array<TL, NL> const& lhs, TR const& rhs) {                                            \
        return Expression<simpla::tags::_NAME_, const Array<TL, NL>, const TR>(lhs, rhs);             \
    };                                                                                                \
    template <typename TL, typename TR, int NR>                                                       \
    auto _NAME_(TL const& lhs, Array<TR, NR> const& rhs) {                                            \
        return Expression<simpla::tags::_NAME_, const TL, const Array<TR, NR>>(lhs, rhs);             \
    };                                                                                                \
    template <typename TL, int NL, typename TR>                                                       \
    auto _NAME_(Array<TL, NL> const& lhs, Expression<TR> const& rhs) {                                \
        return Expression<simpla::tags::_NAME_, const Array<TL, NL>, const Expression<TR>>(lhs, rhs); \
    };                                                                                                \
    template <typename TL, typename TR, int NR>                                                       \
    auto _NAME_(Expression<TL> const& lhs, Array<TR, NR> const& rhs) {                                \
        return Expression<simpla::tags::_NAME_, const Expression<TL>, const Array<TR, NR>>(lhs, rhs); \
    };                                                                                                \
    template <typename TL, int NL, typename TR, int NR>                                               \
    auto _NAME_(Array<TL, NL> const& lhs, Array<TR, NR> const& rhs) {                                 \
        return Expression<simpla::tags::_NAME_, const Array<TL, NL>, const Array<TR, NR>>(lhs, rhs);  \
    };

#define _SP_DEFINE_ARRAY_UNARY_FUNCTION(_NAME_)                          \
    template <typename T, int N>                                         \
    auto _NAME_(Array<T, N> const& lhs) {                                \
        return Expression<simpla::tags::_NAME_, const Array<T, N>>(lhs); \
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

#define _SP_DEFINE_ARRAY_COMPOUND_OP(_OP_)                                              \
    template <typename TL, int NL, typename TR>                                         \
    Array<TL, NL>& operator _OP_##=(Array<TL, NL>& lhs, TR const& rhs) {                \
        lhs = lhs _OP_ rhs;                                                             \
        return lhs;                                                                     \
    }                                                                                   \
    template <typename TL, int NL, typename... TR>                                      \
    Array<TL, NL>& operator _OP_##=(Array<TL, NL>& lhs, Expression<TR...> const& rhs) { \
        lhs = lhs _OP_ rhs;                                                             \
        return lhs;                                                                     \
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

#define _SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR(_NAME_, _REDUCTION_, _OP_)                                       \
    template <typename TL, int NL, typename TR>                                                                   \
    auto operator _OP_(Array<TL, NL> const& lhs, TR const& rhs) {                                                 \
        return reduction<_REDUCTION_>(Expression<simpla::tags::_NAME_, const Array<TL, NL>, const TR>(lhs, rhs)); \
    };                                                                                                            \
    template <typename TL, typename TR, int NR>                                                                   \
    auto operator _OP_(TL const& lhs, Array<TR, NR> const& rhs) {                                                 \
        return reduction<_REDUCTION_>(Expression<simpla::tags::_NAME_, const TL, const Array<TR, NR>>(lhs, rhs)); \
    };                                                                                                            \
    template <typename TL, int NL, typename... TR>                                                                \
    auto operator _OP_(Array<TL, NL> const& lhs, Expression<TR...> const& rhs) {                                  \
        return reduction<_REDUCTION_>(                                                                            \
            Expression<simpla::tags::_NAME_, const Array<TL, NL>, const Expression<TR...>>(lhs, rhs));            \
    };                                                                                                            \
    template <typename... TL, typename TR, int NR>                                                                \
    auto operator _OP_(Expression<TL...> const& lhs, Array<TR, NR> const& rhs) {                                  \
        return reduction<_REDUCTION_>(                                                                            \
            Expression<simpla::tags::_NAME_, const Expression<TL...>, const Array<TR, NR>>(lhs, rhs));            \
    };                                                                                                            \
    template <typename TL, int NL, typename TR, int NR>                                                           \
    auto operator _OP_(Array<TL, NL> const& lhs, Array<TR, NR> const& rhs) {                                      \
        return reduction<_REDUCTION_>(                                                                            \
            Expression<simpla::tags::_NAME_, const Array<TL, NL>, const Array<TR, NR>>(lhs, rhs));                \
    };

_SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR(not_equal_to, simpla::tags::logical_or, !=)
_SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR(equal_to, simpla::tags::logical_and, ==)
_SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR(less, simpla::tags::logical_and, <)
_SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR(greater, simpla::tags::logical_and, >)
_SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR(less_equal, simpla::tags::logical_and, <=)
_SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR(greater_equal, simpla::tags::logical_and, >=)

#undef _SP_DEFINE_ARRAY_BINARY_BOOLEAN_OPERATOR

}  // namespace simpla{
#endif  // SIMPLA_ARRAY_H
