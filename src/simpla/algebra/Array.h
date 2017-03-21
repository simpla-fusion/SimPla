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

//#ifdef NDEBUG
#include <simpla/toolbox/MemoryPool.h>
//#endif

namespace simpla {
namespace algebra {

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

    ArrayView(m_index_box_type const& b, std::shared_ptr<value_type> const& d = nullptr)
        : m_inner_index_box_(b), m_outer_index_box_(b), m_data_(d) {
        Update();
    }
    ArrayView(m_index_box_type const& b_in, m_index_box_type const& b_out,
              std::shared_ptr<value_type> const& d = nullptr)
        : m_inner_index_box_(b_in), m_outer_index_box_(b_out), m_data_(d) {
        Update();
    }

    ArrayView(this_type const& other, m_index_tuple const& offset)
        : m_inner_index_box_(other.m_inner_index_box_),
          m_outer_index_box_(other.m_outer_index_box_),
          m_data_(other.m_data_) {
        Shift(offset);
    }
    template <typename... U>
    ArrayView(declare::Expression<U...> const& expr) {
        Foreach(tags::_assign(), expr);
    }

    virtual ~ArrayView() {}
    virtual bool empty() const { return m_data_ == nullptr; }
    virtual std::type_info const& value_type_info() const { return typeid(value_type); }
    virtual int GetNDIMS() const { return NDIMS; }
    virtual index_type const* GetInnerLowerIndex() const { return &std::get<0>(m_inner_index_box_)[0]; };
    virtual index_type const* GetInnerUpperIndex() const { return &std::get<1>(m_inner_index_box_)[0]; };
    virtual index_type const* GetOuterLowerIndex() const { return &std::get<0>(m_outer_index_box_)[0]; };
    virtual index_type const* GetOuterUpperIndex() const { return &std::get<1>(m_outer_index_box_)[0]; };
    virtual void const* GetRawData() const { return m_data_.get(); };
    virtual void* GetRawData() { return m_data_.get(); };

    void Update() {
        if (m_data_ == nullptr) {
            m_data_ = sp_alloc_array<value_type>(full_size());
            //#ifdef NDEBUG
            //#else
            //                std::shared_ptr<V>(new value_type[size()]);
            //#endif
        }
    };

    std::shared_ptr<value_type>& GetData() { return m_data_; }
    std::shared_ptr<value_type> const& GetData() const { return m_data_; }
    void SetData(std::shared_ptr<value_type> const& d) const { m_data_ = d; }
    m_index_box_type const& GetIndexBox() const { return m_inner_index_box_; }
    m_index_box_type const& GetInnerIndexBox() const { return m_inner_index_box_; }
    m_index_box_type const& GetOuterIndexBox() const { return m_outer_index_box_; }
    void Shift(m_index_tuple const& offset) {
        std::get<0>(m_inner_index_box_) += offset;
        std::get<1>(m_inner_index_box_) += offset;
        std::get<0>(m_outer_index_box_) += offset;
        std::get<1>(m_outer_index_box_) += offset;
    }

    this_type operator()(m_index_tuple const& offset) { return this_type(*this, offset); }

    template <typename... TID>
    value_type& at(TID&&... s) {
        return m_data_.get()[hash(std::forward<TID>(s)...)];
    }

    template <typename... TID>
    value_type const& at(TID&&... s) const {
        return m_data_.get()[hash(std::forward<TID>(s)...)];
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
    value_type& operator()(TID&&... s) {
        return at(std::forward<TID>(s)...);
    }

    template <typename... TID>
    value_type const& operator()(TID&&... s) const {
        return at(std::forward<TID>(s)...);
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

    template <typename... Others>
    size_type hash(Others&&... others) const {
        return hash(m_index_tuple(std::forward<Others>(others)...));
    }
    size_type hash(std::initializer_list<index_type> const& idx) const { return hash(m_index_tuple(idx)); }
    size_type hash(m_index_tuple const& idx) const {
        size_type res = idx[0];
        for (int i = 1; i < NDIMS; ++i) {
            res *= (std::get<1>(m_outer_index_box_)[i - 1] - std::get<0>(m_outer_index_box_)[i - 1]);
            res += idx[i] - std::get<0>(m_outer_index_box_)[i];
        }
        return res;
    }

    //    std::ostream& Print(std::ostream& os, int indent = 0) const {
    //        printNdArray(os, m_data_, NDIMS, m_dims_);
    //        return os;
    //    }

   private:
    template <typename TFun>
    void traversal_nd(index_type const* lower, index_type const* upper, TFun const& fun) {
        /// FIXME: need parallelism
        index_type idx[NDIMS];
        for (int i = 0; i < NDIMS; ++i) { idx[i] = lower[i]; }

        while (1) {
            fun(idx);

            ++idx[NDIMS - 1];
            for (int rank = NDIMS - 1; rank > 0; --rank) {
                if (idx[rank] >= upper[rank]) {
                    idx[rank] = lower[rank];
                    ++idx[rank - 1];
                }
            }
            if (idx[0] >= upper[0]) break;
        }
    }

   public:
    template <typename TOP, typename... Others>
    void Foreach(TOP const& op, Others&&... others) {
        Update();
        traversal_nd(&std::get<0>(m_inner_index_box_)[0], &std::get<1>(m_inner_index_box_)[0],
                     [&](index_type const* idx) { op(at(idx), getValue(std::forward<Others>(others), idx)...); });
    };

    template <typename TOP>
    void Apply(TOP const& op) {
        traversal_nd(&std::get<0>(m_inner_index_box_)[0], &std::get<1>(m_inner_index_box_)[0],
                     [&](index_type const* idx) { op(at(idx)); });
    };

    void swap(this_type& other) {
        std::swap(m_data_, other.m_data_);
        std::swap(m_inner_index_box_, other.m_inner_index_box_);
        std::swap(m_outer_index_box_, other.m_outer_index_box_);
    };

    //    this_type split(concept::tags::split const& split) {
    //        this_type other(*this);
    //        size_type max_dims = 0;
    //        int n = 0;
    //        for (int i = 0; i < NDIMS; ++i) {
    //            if (m_dims_[i] > max_dims) {
    //                n = i;
    //                max_dims = m_dims_[i];
    //            }
    //        }
    //        other.m_upper_[n] =
    //            other.m_lower_[n] + (other.m_upper_[n] - other.m_lower_[n]) * split.left() / (split.left() +
    //            split.right());
    //        m_lower_[n] = other.m_upper_[n];
    //
    //        return std::move(other);
    //    }

   public:
    template <typename T>
    static constexpr decltype(auto) getValue(T& v) {
        return v;
    };

    static constexpr decltype(auto) getValue(this_type& self, index_type const* s) { return self.at(s); };
    static constexpr decltype(auto) getValue(this_type const& self, index_type const* s) { return self.at(s); };
    template <typename T, typename I0>
    static constexpr decltype(auto) getValue(T& v, I0 const* s,
                                             ENABLE_IF((simpla::concept::is_indexable<T, I0>::value))) {
        return ((getValue(v[*s], s + 1)));
    }

    template <typename T, typename I0>
    static constexpr T& getValue(T& v, I0 const* s, ENABLE_IF((!simpla::concept::is_indexable<T, I0>::value))) {
        return v;
    };

    template <typename TOP, typename... Others, int... index, typename... Idx>
    static constexpr decltype(auto) _invoke_helper(declare::Expression<TOP, Others...> const& expr,
                                                   int_sequence<index...>, Idx&&... s) {
        return ((TOP::eval(getValue(std::get<index>(expr.m_args_), std::forward<Idx>(s)...)...)));
    }

    template <typename TOP, typename... Others, typename... Idx>
    static constexpr decltype(auto) getValue(declare::Expression<TOP, Others...> const& expr, Idx&&... s) {
        return ((_invoke_helper(expr, int_sequence_for<Others...>(), std::forward<Idx>(s)...)));
    }

    template <typename TOP, typename... Others, int... index>
    static decltype(auto) _invoke_helper(declare::Expression<TOP, Others...> const& expr, int_sequence<index...>,
                                         index_type const* s) {
        return ((expr.m_op_(getValue(std::get<index>(expr.m_args_), s)...)));
    }

    template <typename TOP, typename... Others>
    static decltype(auto) getValue(declare::Expression<TOP, Others...> const& expr, index_type const* s) {
        return ((_invoke_helper(expr, int_sequence_for<Others...>(), s)));
    }
};

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
}  // namespace algebra{

template <typename V, int NDIMS>
using Array = simpla::algebra::declare::Array_<V, NDIMS>;
}  // namespace simpla{
#endif  // SIMPLA_ARRAY_H
