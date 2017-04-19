//
// Created by salmon on 16-12-28.
//

#ifndef SIMPLA_ARRAY_H
#define SIMPLA_ARRAY_H

#include <simpla/concept/Printable.h>
#include <simpla/utilities/FancyStream.h>
#include <simpla/utilities/Log.h>
#include <simpla/utilities/Range.h>
#include <simpla/utilities/sp_def.h>
#include <cstring>

#include "Algebra.h"
#include "Arithmetic.h"
#include "Expression.h"
#include "nTuple.h"
#include "nTupleExt.h"

//#ifdef NDEBUG
#include <simpla/utilities/MemoryPool.h>
//#endif

namespace simpla {
namespace algebra {
template <typename V, int NDIMS>
struct ArrayView;
namespace declare {
template <typename V, int NDIMS>
struct Array_ : public ArrayView<V, NDIMS> {
   private:
    typedef Array_<V, NDIMS> this_type;
    typedef ArrayView<V, NDIMS> base_type;

   public:
    Array_() : base_type() {}
    Array_(Array_ const& other) : base_type(other) {}
    template <typename... Args>
    explicit Array_(Args&&... args) : base_type(std::forward<Args>(args)...) {}

    template <typename T>
    explicit Array_(std::initializer_list<T> const& l) : base_type(l) {}

    virtual ~Array_() {}

    using base_type::operator=;
    using base_type::operator[];
    using base_type::operator();
    using base_type::ndims;
    using base_type::at;
    using base_type::swap;

    //    Array_<V, NDIMS> view(index_type const* il, index_type const* iu) { return Array_<V, NDIMS>(*this, il, iu); };
    //    Array_<const V, NDIMS> view(index_type const* il, index_type const* iu) const { return Array_<V, NDIMS>(*this,
    //    il, iu); };
};
}  // namespace declare

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
void ForeachND(std::tuple<nTuple<index_type, 1>, nTuple<index_type, 1>> const& inner_box, TFun const& fun) {
    for (index_type i = std::get<0>(inner_box)[0], ie = std::get<1>(inner_box)[0]; i < ie; ++i)
        fun(nTuple<index_type, 1>{i});
}
template <typename TFun>
void ForeachND(std::tuple<nTuple<index_type, 2>, nTuple<index_type, 2>> const& inner_box, TFun const& fun) {
    for (index_type i = std::get<0>(inner_box)[0], ie = std::get<1>(inner_box)[0]; i < ie; ++i)
        for (index_type j = std::get<0>(inner_box)[1], je = std::get<1>(inner_box)[1]; j < je; ++j)
            fun(nTuple<index_type, 2>{i, j});
}

template <typename TFun>
void ForeachND(std::tuple<nTuple<index_type, 3>, nTuple<index_type, 3>> const& inner_box, TFun const& fun) {
    for (index_type i = std::get<0>(inner_box)[0], ie = std::get<1>(inner_box)[0]; i < ie; ++i)
        for (index_type j = std::get<0>(inner_box)[1], je = std::get<1>(inner_box)[1]; j < je; ++j)
            for (index_type k = std::get<0>(inner_box)[2], ke = std::get<1>(inner_box)[2]; k < ke; ++k) {
                fun(nTuple<index_type, 3>{i, j, k});
            }
}

template <typename TFun>
void ForeachND(std::tuple<nTuple<index_type, 4>, nTuple<index_type, 4>> const& inner_box, TFun const& fun) {
    for (index_type i = std::get<0>(inner_box)[0], ie = std::get<1>(inner_box)[0]; i < ie; ++i)
        for (index_type j = std::get<0>(inner_box)[1], je = std::get<1>(inner_box)[1]; j < je; ++j)
            for (index_type k = std::get<0>(inner_box)[2], ke = std::get<1>(inner_box)[2]; k < ke; ++k)
                for (index_type l = std::get<0>(inner_box)[3], le = std::get<1>(inner_box)[3]; l < le; ++l) {
                    fun(nTuple<index_type, 4>{i, j, k, l});
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
struct ArrayView : public concept::Printable {
   private:
    typedef ArrayView<V, NDIMS> this_type;

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

   public:
    ArrayView() {
        std::get<0>(m_index_box_) = 0;
        std::get<1>(m_index_box_) = 0;
    }

    ArrayView(this_type const& other)
        : m_data_(other.m_data_),
          m_index_box_(other.m_index_box_),
          m_array_order_fast_first_(other.m_array_order_fast_first_) {
        SetUp();
    }

    explicit ArrayView(std::initializer_list<index_type> const& l) {
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
    ArrayView(m_index_box_type const& b, std::shared_ptr<value_type> const& d = nullptr,
              bool array_order_fast_first = false)
        : m_index_box_(b), m_data_(d), m_array_order_fast_first_(array_order_fast_first) {
        SetUp();
    }

    ArrayView(index_type const* in_low, index_type const* in_up, std::shared_ptr<value_type> const& d = nullptr,
              bool array_order_fast_first = false)
        : m_data_(d), m_array_order_fast_first_(array_order_fast_first) {
        for (int i = 0; i < NDIMS; ++i) {
            std::get<0>(m_index_box_)[i] = in_low[i];
            std::get<1>(m_index_box_)[i] = in_up[i];
        }
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
        if (m_data_ == nullptr && size() > 0) { m_data_ = sp_alloc_array<value_type>(size()); }
    }

    template <typename... U>
    ArrayView(declare::Expression<U...> const& expr) {
        Foreach(tags::_assign(), expr);
    }

    virtual ~ArrayView() {}
    void swap(this_type& other) {
        std::swap(m_data_, other.m_data_);
        std::swap(m_array_order_fast_first_, other.m_array_order_fast_first_);
        std::swap(m_index_box_, other.m_index_box_);

        SetUp();
        other.SetUp();
    };

    template <typename... Others>
    declare::Array_<value_type, NDIMS> operator()(ArrayIndexShift const& IX, Others&&... others) const {
        declare::Array_<value_type, NDIMS> res(*this);
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

    template <typename... TID>
    value_type& at(index_type i0, TID&&... s) {
        return m_data_.get()[vec_dot(m_strides_, m_index_tuple{i0, std::forward<TID>(s)...}) + m_offset_];
    }

    template <typename... TID>
    value_type const& at(index_type i0, TID&&... s) const {
        return m_data_.get()[vec_dot(m_strides_, m_index_tuple{i0, std::forward<TID>(s)...}) + m_offset_];
    }

    value_type& at(m_index_tuple const& idx) { return m_data_.get()[vec_dot(m_strides_, idx) + m_offset_]; }
    value_type const& at(m_index_tuple const& idx) const { return m_data_.get()[vec_dot(m_strides_, idx) + m_offset_]; }

    template <typename TID>
    value_type& operator[](TID s) {
        return at(s);
    }

    template <typename TID>
    value_type const& operator[](TID s) const {
        return at(s);
    }

    template <typename... TID>
    value_type& operator()(index_type i0, TID&&... s) {
        return at(i0, std::forward<TID>(s)...);
    }

    template <typename... TID>
    value_type const& operator()(index_type i0, TID&&... s) const {
        return at(i0, std::forward<TID>(s)...);
    }

    this_type& operator=(this_type const& rhs) {
        Foreach(tags::_assign(), rhs);
        return (*this);
    }

    template <typename TR>
    this_type& operator=(TR const& rhs) {
        Foreach(tags::_assign(), rhs);
        return (*this);
    }

    size_type size() const {
        size_type res = 1;
        for (int i = 0; i < NDIMS; ++i) { res *= (std::get<1>(m_index_box_)[i] - std::get<0>(m_index_box_)[i]); }
        return res;
    }

    void Clear() {
        SetUp();
        memset(m_data_.get(), 0, size() * sizeof(value_type));
    }

    std::ostream& Print(std::ostream& os, int indent = 0) const {
        //        nTuple<size_type, NDIMS> m_dims_;
        ////        CHECK(std::get<0>(m_index_box_));
        ////        CHECK(std::get<1>(m_index_box_));
        //        for (int i = 0; i < NDIMS; ++i) {
        //            m_dims_[i] = std::get<1>(m_index_box_)[i] - std::get<0>(m_index_box_)[i];
        //        }
        //
        //        printNdArray(os, m_data_.get(), NDIMS, &m_dims_[0]);

        os << "Print Array " << m_index_box_ << std::endl;

        detail::ForeachND(m_index_box_, [&](m_index_tuple const& idx) {
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
    //    template <typename TOP, typename... Others>
    //    void Foreach(TOP const& op, Others&&... others) {
    //        if (size() <= 0) { return; }
    //        detail::ForeachND(m_index_box_, [&](m_index_tuple const& idx) {
    //            op(at(idx), getValue(std::forward<Others>(others), idx)...);
    //        });
    //    };

    template <typename TFun>
    void Foreach(TFun const& op,
                 ENABLE_IF(simpla::concept::is_callable<TFun(m_index_tuple const&, value_type&)>::value)) {
        if (size() <= 0) { return; }
        SetUp();
        detail::ForeachND(m_index_box_, [&](m_index_tuple const& idx) { op(idx, at(idx)); });
    };
    template <typename TFun>
    void Foreach(TFun const& op,
                 ENABLE_IF(simpla::concept::is_callable<TFun(m_index_tuple const&, value_type const&)>::value)) const {
        if (size() <= 0) { return; }
        detail::ForeachND(m_index_box_, [&](m_index_tuple const& idx) { op(idx, at(idx)); });
    };

   public:
    static constexpr value_type const& getValue(value_type const& v, m_index_tuple const& s) { return v; };
    static constexpr decltype(auto) getValue(this_type& self, m_index_tuple const& s) { return self.at(s); };
    static constexpr decltype(auto) getValue(this_type const& self, m_index_tuple const& s) { return self.at(s); };

    template <typename TOP, typename... Others, int... IND>
    static constexpr decltype(auto) _invoke_helper(declare::Expression<TOP, Others...> const& expr,
                                                   int_sequence<IND...>, m_index_tuple const& s) {
        return TOP::eval(getValue(std::get<IND>(expr.m_args_), s)...);
    }

    template <typename TOP, typename... Others>
    static constexpr decltype(auto) getValue(declare::Expression<TOP, Others...> const& expr, m_index_tuple const& s) {
        return _invoke_helper(expr, int_sequence_for<Others...>(), s);
    }
};

}  // namespace algebra{

template <typename V, int NDIMS>
using Array = simpla::algebra::declare::Array_<V, NDIMS>;
}  // namespace simpla{
#endif  // SIMPLA_ARRAY_H
