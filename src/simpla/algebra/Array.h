//
// Created by salmon on 16-12-28.
//

#ifndef SIMPLA_ARRAY_H
#define SIMPLA_ARRAY_H

#include <simpla/SIMPLA_config.h>
#include <cstring>

#include <simpla/concept/Splittable.h>
#include <simpla/data/all.h>
#include <simpla/mpl/Range.h>
#include <simpla/mpl/macro.h>
#include <simpla/toolbox/FancyStream.h>
#include <simpla/toolbox/Log.h>

#include "Algebra.h"
#include "Arithmetic.h"
#include "Expression.h"

#ifdef NDEBUG
#include <simpla/toolbox/MemoryPool.h>
#endif

namespace simpla {
namespace algebra {

template <typename V, int NDIMS>
struct ArrayView : public concept::Printable {
   private:
    typedef ArrayView<V, NDIMS> this_type;

   public:
    typedef V value_type;

    typedef V* pod_type;
    static constexpr bool SLOW_FIRST = true;
    typedef std::integral_constant<bool, SLOW_FIRST> is_slow_first;
    typedef std::true_type prefer_pass_by_reference;
    typedef std::false_type is_expression;
    typedef std::true_type is_array;

   private:
    size_type m_dims_[NDIMS];
    index_type m_lower_[NDIMS];
    index_type m_upper_[NDIMS];

    value_type* m_data_;
    std::shared_ptr<value_type> m_data_holder_;

   public:
    ArrayView() { initialize(); }

    template <typename... TID>
    explicit ArrayView(TID&&... idx) : m_data_(nullptr), m_data_holder_(nullptr) {
        initialize(std::forward<TID>(idx)...);
    }

    template <typename... TID>
    ArrayView(value_type* d, TID&&... idx)
        : m_data_(d), m_data_holder_(d, simpla::tags::do_nothing()) {
        initialize(std::forward<TID>(idx)...);
    }

    template <typename... TID>
    ArrayView(std::shared_ptr<V> const& d, TID&&... idx) : m_data_holder_(d), m_dims_{idx...} {}

    template <typename... U>
    ArrayView(declare::Expression<U...> const& expr) {
        apply(tags::_assign(), expr);
    }

    virtual ~ArrayView() {}

    ArrayView(this_type const& other)
        : ArrayView(other.m_data_holder_, other.m_dims_, other.m_lower_, other.m_upper_) {}

    ArrayView(this_type&& other)
        : ArrayView(std::move(other.m_data_holder_), other.m_dims_, other.m_lower_,
                    other.m_upper_) {}

    ArrayView(this_type& other, concept::tags::split const& s) : ArrayView(other.split(s)) {}

    virtual std::type_info const& value_type_info() const { return typeid(value_type); }

    virtual void* data() { return reinterpret_cast<void*>(m_data_); }

    virtual void const* data() const { return reinterpret_cast<void const*>(m_data_); }

    std::shared_ptr<value_type>& data_holder() { return m_data_holder_; }

    std::shared_ptr<value_type> const& data_holder() const { return m_data_holder_; }

    size_type const* dims() const { return m_dims_; }

    index_type const* lower() const { return m_lower_; }

    index_type const* upper() const { return m_upper_; }

    int ndims() const { return NDIMS; }

    template <typename... TID>
    value_type& at(TID&&... s) {
        return m_data_[hash(m_dims_, m_lower_, std::forward<TID>(s)...)];
    }

    template <typename... TID>
    value_type const& at(TID&&... s) const {
        return m_data_[hash(m_dims_, m_lower_, std::forward<TID>(s)...)];
    }

    template <typename TID>
    value_type& operator[](TID s) {
        return m_data_[s];
    }

    template <typename TID>
    value_type const& operator[](TID s) const {
        return m_data_[s];
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
        apply(tags::_assign(), rhs);
        return (*this);
    }

    template <typename TR>
    this_type& operator=(TR const& rhs) {
        apply(tags::_assign(), rhs);
        return (*this);
    }

    size_type size(size_type const* dims) const {
        size_type res = 1;
        for (int i = 0; i < NDIMS; ++i) { res *= dims[i]; }
        return res;
    }

    void deploy() {
        if (!m_data_holder_) {
            m_data_holder_ =
#ifdef NDEBUG
                sp_alloc_array<V>(size(m_dims_));
#else
                std::shared_ptr<V>(new V[size(m_dims_)]);
#endif
        }
        m_data_ = m_data_holder_.get();
    };

    void clear() {
        deploy();

        memset(m_data_, 0, size(dims()) * sizeof(value_type));
    }

    void reset() {
        m_data_holder_.reset();
        m_data_ = nullptr;

        for (int i = 0; i < NDIMS; ++i) {
            m_dims_[i] = 0;
            m_lower_[i] = 0;
            m_upper_[i] = 0;
        }
    }

    typedef std::integral_constant<bool, true> slow_first_t;

    typedef std::integral_constant<bool, false> fast_first_t;

    template <bool array_order>
    static size_type hash_(std::integral_constant<bool, array_order>, size_type const* dims,
                           index_type const* offset) {
        return 0;
    }

    template <bool array_order>
    static size_type hash_(std::integral_constant<bool, array_order>, size_type const* dims,
                           index_type const* offset, index_type s) {
        return s;
    }

    // fast first

    template <typename... TID>
    static size_type hash_(fast_first_t, size_type const* dims, index_type const* offset,
                           index_type i0, TID&&... idx) {
        return i0 +
               hash_(fast_first_t(), dims, offset, std::forward<TID>(idx)...) *
                   dims[NDIMS - sizeof...(TID)-1];
    }

    static size_type hash_(fast_first_t, int_const<NDIMS>, size_type const* dims,
                           index_type const* offset, index_type const* i) {
        return 0;
    }

    template <int N>
    static size_type hash_(fast_first_t, int_const<N>, size_type const* dims,
                           index_type const* offset, index_type const* i) {
#ifndef NDEBUG
        ASSERT(i[N] - offset[N] < dims[N]);
#endif
        return i[N] - offset[N] +
               hash_(fast_first_t(), int_const<N + 1>(), dims, offset, i) * dims[N];
    }
    // slow first

    template <typename... TID>
    static size_type hash_(slow_first_t, size_type const* dims, index_type const* offset,
                           index_type s, index_type i1, TID&&... idx) {
        return hash_(slow_first_t(), dims, offset, s * dims[NDIMS - (sizeof...(TID) + 1)] + i1 -
                                                       offset[NDIMS - (sizeof...(TID) + 1)],
                     std::forward<TID>(idx)...);
    }

    static size_type hash_(slow_first_t, int_const<NDIMS>, size_type const* dims,
                           index_type const* offset, index_type const* i) {
        return 0;
    }

    template <int N>
    static size_type hash_(slow_first_t, int_const<N>, size_type const* dims,
                           index_type const* offset, index_type const* i) {
#ifndef NDEBUG
        ASSERT(i[NDIMS - N - 1] - offset[NDIMS - N - 1] >= 0);
#endif

        return i[NDIMS - N - 1] - offset[NDIMS - N - 1] +
               hash_(slow_first_t(), int_const<N + 1>(), dims, offset, i) * dims[NDIMS - N - 1];
    }

    template <typename... TID>
    static size_type hash(size_type const* dims, index_type const* offset, index_type s,
                          TID&&... idx) {
        static_assert(NDIMS == ((sizeof...(TID) + 1)), "illegal index number! NDIMS=");
        return hash_(std::integral_constant<bool, SLOW_FIRST>(), dims, offset, s,
                     std::forward<TID>(idx)...);
    }

    static size_type hash(size_type const* dims, index_type const* offset, index_type const* s) {
        return hash_(std::integral_constant<bool, SLOW_FIRST>(), int_const<0>(), dims, offset, s);
    }

    std::ostream& print(std::ostream& os, int indent = 0) const {
        printNdArray(os, m_data_, NDIMS, m_dims_);
        return os;
    }

    void initialize(size_type const* dims = nullptr, const index_type* lower = nullptr,
                    const index_type* upper = nullptr) {
        for (int i = 0; i < NDIMS; ++i) {
            m_dims_[i] = dims == nullptr ? 1 : dims[i];
            m_lower_[i] = lower == nullptr ? 0 : lower[i];
            m_upper_[i] = upper == nullptr ? static_cast<index_type>(m_dims_[i]) : upper[i];
        }
    }

    template <typename T0>
    void copy_(T0* dest){};

    template <typename U, typename T0, typename... Others>
    void copy_(U* dest, T0 const& s0, Others&&... others) {
        dest[0] = static_cast<U>(s0);
        copy_(dest + 1, std::forward<Others>(others)...);
    };

    template <typename... TID>
    void initialize(size_type s0, TID&&... idx) {
        size_type dims[NDIMS];
        dims[0] = s0;
        copy_(dims + 1, std::forward<TID>(idx)...);
        initialize(dims);
    }

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
    void apply(TOP const& op, Others&&... others) {
        deploy();
        traversal_nd(m_lower_, m_upper_, [&](index_type const* idx) {
            //             op(at(idx), calculator::get_value(std::forward<Others>(others), idx)...);
        });
    };

    template <typename TOP>
    void apply(TOP const& op) {
        traversal_nd(m_lower_, m_upper_, [&](index_type const* idx) { op(at(idx)); });
    };

    void swap(this_type& other) {
        std::swap(m_data_, other.m_data_);
        std::swap(m_data_holder_, other.m_data_holder_);

        for (int i = 0; i < NDIMS; ++i) {
            std::swap(m_dims_[i], other.m_dims_[i]);
            std::swap(m_lower_[i], other.m_lower_[i]);
            std::swap(m_upper_[i], other.m_upper_[i]);
        }
    };

    this_type split(concept::tags::split const& split) {
        this_type other(*this);
        size_type max_dims = 0;
        int n = 0;
        for (int i = 0; i < NDIMS; ++i) {
            if (m_dims_[i] > max_dims) {
                n = i;
                max_dims = m_dims_[i];
            }
        }
        other.m_upper_[n] =
            other.m_lower_[n] +
            (other.m_upper_[n] - other.m_lower_[n]) * split.left() / (split.left() + split.right());
        m_lower_[n] = other.m_upper_[n];

        return std::move(other);
    }

   public:
    template <typename T>
    static constexpr T& get_value(T& v) {
        return v;
    };

    static constexpr value_type& get_value(this_type& self, index_type const* s) {
        return self.m_data_[hash(self.m_dims_, self.m_lower_, s)];
    };

    static constexpr value_type const& get_value(this_type const& self, index_type const* s) {
        return self.m_data_[hash(self.m_dims_, self.m_lower_, s)];
    };

    template <typename T, typename I0>
    static constexpr auto get_value(T& v, I0 const* s,
                                    ENABLE_IF((simpla::concept::is_indexable<T, I0>::value))) {
        return ((get_value(v[*s], s + 1)));
    }

    template <typename T, typename I0>
    static constexpr T& get_value(T& v, I0 const* s,
                                  ENABLE_IF((!simpla::concept::is_indexable<T, I0>::value))) {
        return v;
    };

    template <typename TOP, typename... Others, int... index, typename... Idx>
    static constexpr auto _invoke_helper(declare::Expression<TOP, Others...> const& expr,
                                         int_sequence<index...>, Idx&&... s) {
        return ((TOP::eval(get_value(std::get<index>(expr.m_args_), std::forward<Idx>(s)...)...)));
    }

    template <typename TOP, typename... Others, typename... Idx>
    static constexpr auto get_value(declare::Expression<TOP, Others...> const& expr, Idx&&... s) {
        return ((_invoke_helper(expr, int_sequence_for<Others...>(), std::forward<Idx>(s)...)));
    }

    template <typename TOP, typename... Others, int... index>
    static auto _invoke_helper(declare::Expression<TOP, Others...> const& expr,
                               int_sequence<index...>, index_type const* s) {
        return ((expr.m_op_(get_value(std::get<index>(expr.m_args_), s)...)));
    }

    template <typename TOP, typename... Others>
    static auto get_value(declare::Expression<TOP, Others...> const& expr, index_type const* s) {
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


    Array_<V, NDIMS> view(index_type const* il, index_type const* iu) {
        return Array_<V, NDIMS>(*this, il, iu);
    };

    Array_<const V, NDIMS> view(index_type const* il, index_type const* iu) const {
        return Array_<V, NDIMS>(*this, il, iu);
    };
};
}  // namespace declare

namespace traits {
template <typename T, int I>
struct rank<declare::Array_<T, I>> : public int_const<I> {};

template <typename T, int I>
struct sub_type<declare::Array_<T, I>> {
    typedef std::conditional_t<I == 0, T, declare::Array_<T, I - 1>> type;
};

template <typename T>
struct pod_type<declare::Array_<T, 0>> {
    typedef pod_type_t<T> type;
};
template <typename T, int I>
struct pod_type<declare::Array_<T, I>> {
    typedef pod_type_t<declare::Array_<T, I - 1>>* type;
};

}  // namespace traits

}  // namespace algebra{

template <typename V, int NDIMS>
using Array = simpla::algebra::declare::Array_<V, NDIMS>;
}  // namespace simpla{
#endif  // SIMPLA_ARRAY_H
