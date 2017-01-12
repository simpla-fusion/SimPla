//
// Created by salmon on 17-1-8.
//

#ifndef SIMPLA_NDARRAY_H
#define SIMPLA_NDARRAY_H
#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Splittable.h>
#include <simpla/data/all.h>
#include <simpla/mpl/Range.h>
#include <simpla/mpl/macro.h>
#include <simpla/toolbox/Log.h>
#include <simpla/toolbox/FancyStream.h>
#include <cstring>
#include <memory>

#include "Algebra.h"
#include "Arithmetic.h"
#include "Expression.h"

namespace simpla {
namespace algebra {

namespace declare {
template <typename V, size_type NDIMS>
struct ndArray_;
}
namespace traits {
template <typename T, size_type I>
struct reference<declare::ndArray_<T, I>> {
    typedef declare::ndArray_<T, I>& type;
};

template <typename T, size_type I>
struct reference<const declare::ndArray_<T, I>> {
    typedef declare::ndArray_<T, I> const& type;
};

template <typename T, size_type I>
struct rank<declare::ndArray_<T, I>> : public int_const<I> {};

// template<typename V, size_type I>
// struct extents<declare::ndArray_<V, I> > : public int_sequence<I...> {};

template <typename T, size_type I>
struct value_type<declare::ndArray_<T, I>> {
    typedef T type;
};

template <typename T, size_type I>
struct sub_type<declare::ndArray_<T, I>> {
    typedef std::conditional_t<I == 0, T, declare::ndArray_<T, I - 1>> type;
};

template <typename T>
struct pod_type<declare::ndArray_<T, 0>> {
    typedef pod_type_t<T> type;
};
template <typename T, size_type I>
struct pod_type<declare::ndArray_<T, I>> {
    typedef pod_type_t<declare::ndArray_<T, I - 1>>* type;
};

}  // namespace traits

namespace declare {
template <typename V, size_type NDIMS>
struct ndArray_ {
   private:
    typedef ndArray_<V, NDIMS> this_type;
    typedef calculus::calculator<this_type> calculator;

   public:
    typedef V value_type;
    typedef traits::sub_type_t<this_type> sub_type;

    std::vector<sub_type> m_data_;

   public:
    ndArray_() {}

    template <typename... TID>
    explicit ndArray_(size_type N, TID&&... idx) : m_data_(N) {
        calculator::initialize(this, std::forward<TID>(idx)...);
    }

    template <typename... TID>
    ndArray_(value_type* d, TID&&... idx)
        : m_data_(d), m_data_holder_(d, simpla::tags::do_nothing()) {
        calculator::initialize(this, std::forward<TID>(idx)...);
    }

    template <typename... TID>
    ndArray_(std::shared_ptr<V> const& d, TID&&... idx) : m_data_holder_(d), m_dims_{idx...} {}

    template <typename... U>
    ndArray_(Expression<U...> const& expr) {
        calculator::apply((*this), tags::_assign(), expr);
    }

    ~ndArray_() {}

    ndArray_(this_type const& other)
        : ndArray_(other.m_data_holder_, other.m_dims_, other.m_lower_, other.m_upper_) {}

    ndArray_(this_type&& other)
        : ndArray_(std::move(other.m_data_holder_), other.m_dims_, other.m_lower_, other.m_upper_) {
    }

    ndArray_(this_type& other, concept::tags::split const& split)
        : m_data_(nullptr), m_data_holder_(nullptr) {
        calculator::split(split, other, *this);
    }

    virtual std::type_info const& value_type_info() const { return typeid(value_type); }

    virtual void* data() { return reinterpret_cast<void*>(m_data_); }

    virtual void const* data() const { return reinterpret_cast<void const*>(m_data_); }

    std::shared_ptr<value_type>& data_holder() { return m_data_holder_; }

    std::shared_ptr<value_type> const& data_holder() const { return m_data_holder_; }

    size_type const* dims() const { return m_dims_; }

    index_type const* lower() const { return m_lower_; }

    index_type const* upper() const { return m_upper_; }

    size_type ndims() const { return NDIMS; }

    template <typename... TID>
    value_type& at(TID&&... s) {
        return m_data_[calculator::hash(m_dims_, m_lower_, std::forward<TID>(s)...)];
    }

    template <typename... TID>
    value_type const& at(TID&&... s) const {
        return m_data_[calculator::hash(m_dims_, m_lower_, std::forward<TID>(s)...)];
    }

    template <typename TID>
    inline value_type& operator[](TID s) {
        return m_data_[s];
    }

    template <typename TID>
    inline value_type const& operator[](TID s) const {
        return m_data_[s];
    }

    template <typename... TID>
    inline value_type& operator()(TID&&... s) {
        return at(std::forward<TID>(s)...);
    }

    template <typename... TID>
    inline value_type const& operator()(TID&&... s) const {
        return at(std::forward<TID>(s)...);
    }

    virtual std::ostream& print(std::ostream& os, int indent = 0) const {
        return calculator::print(*this, os, indent);
    }

    void deploy() { calculator::deploy(*this); }

    void clear() { calculator::clear(*this); }

    void reset() { calculator::reset(*this); }

    void swap(this_type& other) { calculator::swap(*this, other); }

    inline this_type& operator=(this_type const& rhs) {
        calculator::apply((*this), tags::_assign(), rhs);
        return (*this);
    }

    template <typename TR>
    inline this_type& operator=(TR const& rhs) {
        calculator::apply((*this), tags::_assign(), rhs);
        return (*this);
    }

    template <typename TFun>
    void apply(TFun const& fun) {
        calculator::apply((*this), fun);
    }

    template <typename TFun>
    void apply(TFun const& fun) const {
        calculator::apply((*this), fun);
    }
};
}  // namespace declare

namespace calculus {
namespace st = simpla::traits;

template <typename V, size_type NDIMS>
struct calculator<declare::ndArray_<V, NDIMS>> {
    typedef declare::ndArray_<V, NDIMS> self_type;
    typedef traits::value_type_t<self_type> value_type;

    static inline size_type size(size_type const* dims) {
        size_type res = 1;
        for (int i = 0; i < NDIMS; ++i) { res *= dims[i]; }
        return res;
    }

    static void deploy(self_type& self) {
        if (!self.m_data_holder_) {
            self.m_data_holder_ =
#ifdef NDEBUG
                sp_alloc_array<V>(size(self.m_dims_));
#else
                std::shared_ptr<V>(new V[size(self.m_dims_)]);
#endif
        }
        self.m_data_ = self.m_data_holder_.get();
    };

    static void clear(self_type& self) {
        deploy(self);

        memset(self.m_data_, 0, size(self.dims()) * sizeof(value_type));
    }

    static void reset(self_type& self) {
        self.m_data_holder_.reset();
        self.m_data_ = nullptr;

        for (int i = 0; i < NDIMS; ++i) {
            self.m_dims_[i] = 0;
            self.m_lower_[i] = 0;
            self.m_upper_[i] = 0;
        }
    }

    typedef std::integral_constant<bool, true> slow_first_t;

    typedef std::integral_constant<bool, false> fast_first_t;

    template <bool array_order>
    static inline size_type hash_(std::integral_constant<bool, array_order>, size_type const* dims,
                                  index_type const* offset) {
        return 0;
    }

    template <bool array_order>
    static inline size_type hash_(std::integral_constant<bool, array_order>, size_type const* dims,
                                  index_type const* offset, index_type s) {
        return s;
    }

    // fast first

    template <typename... TID>
    static inline size_type hash_(fast_first_t, size_type const* dims, index_type const* offset,
                                  index_type i0, TID&&... idx) {
        return i0 +
               hash_(fast_first_t(), dims, offset, std::forward<TID>(idx)...) *
                   dims[NDIMS - sizeof...(TID)-1];
    }

    static inline size_type hash_(fast_first_t, int_const<NDIMS>, size_type const* dims,
                                  index_type const* offset, index_type const* i) {
        return 0;
    }

    template <size_type N>
    static inline size_type hash_(fast_first_t, int_const<N>, size_type const* dims,
                                  index_type const* offset, index_type const* i) {
#ifndef NDEBUG
        ASSERT(i[N] - offset[N] < dims[N]);
#endif
        return i[N] - offset[N] +
               hash_(fast_first_t(), int_const<N + 1>(), dims, offset, i) * dims[N];
    }
    // slow first

    template <typename... TID>
    static inline size_type hash_(slow_first_t, size_type const* dims, index_type const* offset,
                                  index_type s, index_type i1, TID&&... idx) {
        return hash_(slow_first_t(), dims, offset, s * dims[NDIMS - (sizeof...(TID) + 1)] + i1 -
                                                       offset[NDIMS - (sizeof...(TID) + 1)],
                     std::forward<TID>(idx)...);
    }

    static inline size_type hash_(slow_first_t, int_const<NDIMS>, size_type const* dims,
                                  index_type const* offset, index_type const* i) {
        return 0;
    }

    template <size_type N>
    static inline size_type hash_(slow_first_t, int_const<N>, size_type const* dims,
                                  index_type const* offset, index_type const* i) {
#ifndef NDEBUG
        ASSERT(i[NDIMS - N - 1] - offset[NDIMS - N - 1] >= 0);
#endif

        return i[NDIMS - N - 1] - offset[NDIMS - N - 1] +
               hash_(slow_first_t(), int_const<N + 1>(), dims, offset, i) * dims[NDIMS - N - 1];
    }

    template <typename... TID>
    static inline size_type hash(size_type const* dims, index_type const* offset, index_type s,
                                 TID&&... idx) {
        static_assert(NDIMS == ((sizeof...(TID) + 1)), "illegal index number! NDIMS=");
        return hash_(std::integral_constant<bool>(), dims, offset, s, std::forward<TID>(idx)...);
    }

    static inline size_type hash(size_type const* dims, index_type const* offset,
                                 index_type const* s) {
        return hash_(std::integral_constant<bool>(), int_const<0>(), dims, offset, s);
    }

   public:
    template <typename T>
    static constexpr inline T& get_value(T& v) {
        return v;
    };

    static constexpr inline value_type& get_value(self_type& self, index_type const* s) {
        return self.m_data_[hash(self.m_dims_, self.m_lower_, s)];
    };

    static constexpr inline value_type const& get_value(self_type const& self,
                                                        index_type const* s) {
        return self.m_data_[hash(self.m_dims_, self.m_lower_, s)];
    };

    template <typename T, typename I0>
    static constexpr inline auto get_value(T& v, I0 const* s,
                                           ENABLE_IF((st::is_indexable<T, I0>::value))) {
        return ((get_value(v[*s], s + 1)));
    }

    template <typename T, typename I0>
    static constexpr inline T& get_value(T& v, I0 const* s,
                                         ENABLE_IF((!st::is_indexable<T, I0>::value))) {
        return v;
    };

   private:
    //    template<typename T, typename ...Args> static constexpr inline T &
    //    get_value_(std::integral_constant<bool, false> const &, T &v, Args
    //    &&...)
    //    {
    //        return v;
    //    }
    //
    //
    //    template<typename T, typename I0, typename ...Idx> static constexpr
    //    inline st::remove_extents_t<T, I0, Idx...> &
    //    get_value_(std::integral_constant<bool, true> const &, T &v, I0 const
    //    &s0, Idx &&...idx)
    //    {
    //        return get_value(v[s0], std::forward<Idx>(idx)...);
    //    };
    // public:
    //    template<typename T, typename I0, typename ...Idx> static constexpr
    //    inline st::remove_extents_t<T, I0, Idx...> &
    //    get_value(T &v, I0 const &s0, Idx &&...idx)
    //    {
    //        return get_value_(std::integral_constant<bool, st::is_indexable<T,
    //        I0>::value>(),
    //                          v, s0, std::forward<Idx>(idx)...);
    //    };
    //
    //    template<typename T, size_type N> static constexpr inline T &
    //    get_value(declare::nTuple_<T, N> &v, size_type const &s0) { return
    //    v[s0]; };
    //
    //    template<typename T, size_type N> static constexpr inline T const &
    //    get_value(declare::nTuple_<T, N> const &v, size_type const &s0) {
    //    return v[s0]; };
   public:
    template <typename TOP, typename... Others, size_type... index, typename... Idx>
    static constexpr inline auto _invoke_helper(declare::Expression<TOP, Others...> const& expr,
                                                index_sequence<index...>, Idx&&... s) {
        return ((TOP::eval(get_value(std::get<index>(expr.m_args_), std::forward<Idx>(s)...)...)));
    }

    template <typename TOP, typename... Others, typename... Idx>
    static constexpr inline auto get_value(declare::Expression<TOP, Others...> const& expr,
                                           Idx&&... s) {
        return ((_invoke_helper(expr, index_sequence_for<Others...>(), std::forward<Idx>(s)...)));
    }

    template <typename TOP, typename... Others, size_type... index>
    static inline auto _invoke_helper(declare::Expression<TOP, Others...> const& expr,
                                      index_sequence<index...>, index_type const* s) {
        return ((expr.m_op_(get_value(std::get<index>(expr.m_args_), s)...)));
    }

    template <typename TOP, typename... Others>
    static inline auto get_value(declare::Expression<TOP, Others...> const& expr,
                                 index_type const* s) {
        return ((_invoke_helper(expr, index_sequence_for<Others...>(), s)));
    }

    template <typename TFun>
    static void traversal_nd(index_type const* lower, index_type const* upper, TFun const& fun) {
        /// FIXME: need parallelism
        index_type idx[NDIMS];
        for (int i = 0; i < NDIMS; ++i) { idx[i] = lower[i]; }

        while (1) {
            fun(idx);

            ++idx[NDIMS - 1];
            for (size_type rank = NDIMS - 1; rank > 0; --rank) {
                if (idx[rank] >= upper[rank]) {
                    idx[rank] = lower[rank];
                    ++idx[rank - 1];
                }
            }
            if (idx[0] >= upper[0]) break;
        }
    }

    template <typename TOP, typename... Others>
    static inline void apply(self_type& self, TOP const& op, Others&&... others) {
        deploy(self);
        traversal_nd(self.m_lower_, self.m_upper_, [&](index_type const* idx) {
            op(get_value(self, idx), get_value(std::forward<Others>(others), idx)...);
        });
    };

    template <typename TOP>
    static inline void apply(self_type& self, TOP const& op) {
        traversal_nd(self.m_lower_, self.m_upper_,
                     [&](index_type const* idx) { op(get_value(self, idx)); });
    };

    template <typename TOP>
    static inline void apply(self_type const& self, TOP const& op) {
        traversal_nd(self.m_lower_, self.m_upper_,
                     [&](index_type const* idx) { op(get_value(self, idx)); });
    };

    static inline void swap(self_type& self, self_type& other) {
        std::swap(self.m_data_, other.m_data_);
        std::swap(self.m_data_holder_, other.m_data_holder_);

        for (int i = 0; i < NDIMS; ++i) {
            std::swap(self.m_dims_, other.m_dims_);
            std::swap(self.m_lower_, other.m_lower_);
            std::swap(self.m_upper_, other.m_upper_);
        }
    };

    static inline void split(concept::tags::split const& split, self_type& other, self_type& self) {
        self_type(other).swap(self);
        size_type max_dims = 0;
        int n = 0;
        for (int i = 0; i < NDIMS; ++i) {
            if (self.m_dims_[i] > max_dims) {
                n = i;
                max_dims = self.m_dims_[i];
            }
        }
        other.m_upper_[n] =
            other.m_lower_[n] +
            (other.m_upper_[n] - other.m_lower_[n]) * split.left() / (split.left() + split.right());
        self.m_lower_[n] = other.m_upper_[n];
    }

    static declare::ndArray_<V, NDIMS> view(self_type& self, index_type const* il,
                                            index_type const* iu) {
        return declare::ndArray_<V, NDIMS>(self.m_data_holder_, self.m_dims_, il, iu);
    };

    static declare::ndArray_<const V, NDIMS> view(self_type const& self, index_type const* il,
                                                  index_type const* iu) {
        return declare::ndArray_<V, NDIMS>(self.m_data_, self.m_dims_, il, iu);
    };

    static std::ostream& print(self_type const& self, std::ostream& os, int indent = 0) {
        printNdndArray(os, self.m_data_, NDIMS, self.m_dims_);
        return os;
    }

    static void initialize(self_type* self, size_type const* dims = nullptr,
                           const index_type* lower = nullptr, const index_type* upper = nullptr) {
        for (int i = 0; i < NDIMS; ++i) {
            self->m_dims_[i] = dims == nullptr ? 1 : dims[i];
            self->m_lower_[i] = lower == nullptr ? 0 : lower[i];
            self->m_upper_[i] =
                upper == nullptr ? static_cast<index_type>(self->m_dims_[i]) : upper[i];
        }
    }

    template <typename T0>
    static void copy_(T0* dest){};

    template <typename U, typename T0, typename... Others>
    static void copy_(U* dest, T0 const& s0, Others&&... others) {
        dest[0] = static_cast<U>(s0);
        copy_(dest + 1, std::forward<Others>(others)...);
    };

    //
    template <typename... TID>
    static void initialize(self_type* self, index_type s0, TID&&... idx) {
        size_type dims[NDIMS];
        dims[0] = s0;
        copy_(dims + 1, std::forward<TID>(idx)...);
        initialize(self, dims);
    }
};
}  // namespace calculus{
}  // namespace algebra{
}  // namespace simpla{
#endif  // SIMPLA_NDARRAY_H
