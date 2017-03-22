//
// Created by salmon on 16-12-28.
//

#ifndef SIMPLA_ARRAY_H
#define SIMPLA_ARRAY_H

#include <simpla/SIMPLA_config.h>
#include <cstring>

#include <simpla/concept/Splittable.h>
#include <simpla/mpl/Range.h>
#include <simpla/mpl/macro.h>
#include <simpla/toolbox/FancyStream.h>
#include <simpla/toolbox/Log.h>

#include "Algebra.h"
#include "Arithmetic.h"
#include "Expression.h"
#include "nTuple.h"
#include "nTupleExt.h"

//#ifdef NDEBUG
#include <simpla/toolbox/MemoryPool.h>
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

    Array_<V, NDIMS> view(index_type const* il, index_type const* iu) { return Array_<V, NDIMS>(*this, il, iu); };

    Array_<const V, NDIMS> view(index_type const* il, index_type const* iu) const {
        return Array_<V, NDIMS>(*this, il, iu);
    };
};
}  // namespace declare

struct ArrayIndexShift {
    int dim_num = 0;
    index_type value = 0;
};
ArrayIndexShift operator+(ArrayIndexShift const& l, index_type s) { return ArrayIndexShift{l.dim_num, l.value - s}; }
ArrayIndexShift operator-(ArrayIndexShift const& l, index_type s) { return ArrayIndexShift{l.dim_num, l.value + s}; }

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
            for (index_type k = std::get<0>(inner_box)[1], ke = std::get<1>(inner_box)[1]; k < ke; ++k) {
                fun(nTuple<index_type, 3>{i, j, k});
            }
}

template <typename TFun>
void ForeachND(std::tuple<nTuple<index_type, 4>, nTuple<index_type, 4>> const& inner_box, TFun const& fun) {
    for (index_type i = std::get<0>(inner_box)[0], ie = std::get<1>(inner_box)[0]; i < ie; ++i)
        for (index_type j = std::get<0>(inner_box)[1], je = std::get<1>(inner_box)[1]; j < je; ++j)
            for (index_type k = std::get<0>(inner_box)[1], ke = std::get<1>(inner_box)[1]; k < ke; ++k)
                for (index_type l = std::get<0>(inner_box)[1], le = std::get<1>(inner_box)[1]; l < le; ++l) {
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

   private:
    typedef nTuple<index_type, NDIMS> m_index_tuple;
    typedef std::tuple<m_index_tuple, m_index_tuple> m_index_box_type;
    m_index_box_type m_inner_index_box_;
    m_index_box_type m_outer_index_box_;
    std::shared_ptr<value_type> m_data_ = nullptr;

   public:
    ArrayView() {}

    explicit ArrayView(std::initializer_list<index_type> const& l) {
        for (int i = 0; i < NDIMS; ++i) {
            std::get<0>(m_inner_index_box_)[i] = 0;
            std::get<1>(m_inner_index_box_)[i] = 1;
        }
        int count = 0;
        for (auto const& v : l) {
            if (count >= NDIMS) { break; }
            std::get<1>(m_inner_index_box_)[count] = v;
            ++count;
        }
        std::get<0>(m_outer_index_box_) = std::get<0>(m_inner_index_box_);
        std::get<1>(m_outer_index_box_) = std::get<1>(m_inner_index_box_);
        Update();
    }
    ArrayView(m_index_box_type const& b, std::shared_ptr<value_type> const& d = nullptr)
        : m_inner_index_box_(b), m_outer_index_box_(b), m_data_(d) {
        Update();
    }
    ArrayView(m_index_box_type const& b_in, m_index_box_type const& b_out,
              std::shared_ptr<value_type> const& d = nullptr)
        : m_inner_index_box_(b_in), m_outer_index_box_(b_out), m_data_(d) {
        Update();
    }

    template <typename... U>
    ArrayView(declare::Expression<U...> const& expr) {
        Foreach(tags::_assign(), expr);
    }

    virtual ~ArrayView() {}

    template <typename... Others>
    this_type operator()(ArrayIndexShift const& IX, Others&&... others) const {
        this_type res(*this);
        res.Shift(IX, std::forward<Others>(others)...);
        return std::move(res);
    }
    void Shift(ArrayIndexShift const& IX) {
        std::get<0>(m_inner_index_box_)[IX.dim_num] += IX.value;
        std::get<1>(m_inner_index_box_)[IX.dim_num] += IX.value;
        std::get<0>(m_outer_index_box_)[IX.dim_num] += IX.value;
        std::get<1>(m_outer_index_box_)[IX.dim_num] += IX.value;
    }
    template <typename... Others>
    void Shift(ArrayIndexShift const& IX, Others&&... others) {
        Shift(IX);
        Shift(std::forward<Others>(others)...);
    }
    void Shift(m_index_tuple const& offset) {
        std::get<0>(m_inner_index_box_) += offset;
        std::get<1>(m_inner_index_box_) += offset;
        std::get<0>(m_outer_index_box_) += offset;
        std::get<1>(m_outer_index_box_) += offset;
    }
    virtual bool empty() const { return m_data_ == nullptr; }
    virtual std::type_info const& value_type_info() const { return typeid(value_type); }
    virtual int GetNDIMS() const { return NDIMS; }

    virtual index_type const* GetInnerLowerIndex() const { return &std::get<0>(m_inner_index_box_)[0]; };
    virtual index_type const* GetInnerUpperIndex() const { return &std::get<1>(m_inner_index_box_)[0]; };
    virtual index_type const* GetOuterLowerIndex() const { return &std::get<0>(m_outer_index_box_)[0]; };
    virtual index_type const* GetOuterUpperIndex() const { return &std::get<1>(m_outer_index_box_)[0]; };
    virtual void const* GetRawData() const { return m_data_.get(); };
    virtual void* GetRawData() { return m_data_.get(); };

    //    declare::Array_<V, NDIMS> operator()(nTuple<index_type, NDIMS> const& offset) {
    //        declare::Array_<V, NDIMS> res(this);
    //        res.Shift(offset);
    //        return std::move(res);
    //    };
    void Update() {
        if (m_data_ == nullptr) { m_data_ = sp_alloc_array<value_type>(full_size()); }
    };

    std::shared_ptr<value_type>& GetData() { return m_data_; }
    std::shared_ptr<value_type> const& GetData() const { return m_data_; }
    void SetData(std::shared_ptr<value_type> const& d) const { m_data_ = d; }
    m_index_box_type const& GetIndexBox() const { return m_inner_index_box_; }
    m_index_box_type const& GetInnerIndexBox() const { return m_inner_index_box_; }
    m_index_box_type const& GetOuterIndexBox() const { return m_outer_index_box_; }

    template <typename... TID>
    value_type& at(index_type i0, TID&&... s) {
        return m_data_.get()[detail::Hash(m_outer_index_box_, m_index_tuple{i0, std::forward<TID>(s)...})];
    }

    template <typename... TID>
    value_type const& at(index_type i0, TID&&... s) const {
        return m_data_.get()[detail::Hash(m_outer_index_box_, m_index_tuple{i0, std::forward<TID>(s)...})];
    }

    value_type& at(m_index_tuple const& idx) { return m_data_.get()[detail::Hash(m_outer_index_box_, idx)]; }
    value_type const& at(m_index_tuple const& idx) const {
        return m_data_.get()[detail::Hash(m_outer_index_box_, idx)];
    }

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
    size_type full_size() const {
        size_type res = 1;
        for (int i = 0; i < NDIMS; ++i) {
            res *= (std::get<1>(m_outer_index_box_)[i] - std::get<0>(m_outer_index_box_)[i]);
        }
        return res;
    }
    size_type size() const {
        size_type res = 1;
        for (int i = 0; i < NDIMS; ++i) {
            res *= (std::get<1>(m_inner_index_box_)[i] - std::get<0>(m_inner_index_box_)[i]);
        }
        return res;
    }

    void Clear() {
        Update();
        memset(m_data_.get(), 0, full_size() * sizeof(value_type));
    }

    std::ostream& Print(std::ostream& os, int indent = 0) const {
        //        nTuple<size_type, NDIMS> m_dims_;
        ////        CHECK(std::get<0>(m_outer_index_box_));
        ////        CHECK(std::get<1>(m_outer_index_box_));
        //        for (int i = 0; i < NDIMS; ++i) {
        //            m_dims_[i] = std::get<1>(m_outer_index_box_)[i] - std::get<0>(m_outer_index_box_)[i];
        //        }
        //
        //        printNdArray(os, m_data_.get(), NDIMS, &m_dims_[0]);

        detail::ForeachND(m_inner_index_box_, [&](m_index_tuple const& idx) {
            if (idx[NDIMS - 1] == std::get<0>(m_inner_index_box_)[NDIMS - 1]) {
                os << "{" << at(idx);
            } else {
                os << "," << at(idx);
            }
            if (idx[NDIMS - 1] == std::get<1>(m_inner_index_box_)[NDIMS - 1] - 1) { os << "}" << std::endl; }
        });

        return os;
    }

   private:
   public:
    template <typename TOP, typename... Others>
    void Foreach(TOP const& op, Others&&... others) {
        Update();
        detail::ForeachND(m_inner_index_box_, [&](m_index_tuple const& idx) {
            op(at(idx), getValue(std::forward<Others>(others), idx)...);
        });
    };

    template <typename TOP>
    void Foreach(TOP const& op) {
        detail::ForeachND(m_inner_index_box_, [&](m_index_tuple const& idx) { op(at(idx)); });
    };

    void swap(this_type& other) {
        std::swap(m_data_, other.m_data_);
        std::swap(m_inner_index_box_, other.m_inner_index_box_);
        std::swap(m_outer_index_box_, other.m_outer_index_box_);
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
