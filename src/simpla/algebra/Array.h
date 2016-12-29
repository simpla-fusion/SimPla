//
// Created by salmon on 16-12-28.
//

#ifndef SIMPLA_ARRAY_H
#define SIMPLA_ARRAY_H

#include <simpla/SIMPLA_config.h>
#include <cstring>

#include "Algebra.h"
#include "Arithmetic.h"
#include "Expression.h"
#include <simpla/mpl/range.h>
#include <simpla/concept/Splittable.h>

#ifdef NDEBUG
#   include <simpla/toolbox/MemoryPool.h>
#endif

namespace simpla { namespace algebra
{

namespace declare { template<typename V, size_type NDIMS, bool SLOW_FIRST> struct Array_; }
namespace traits
{
template<typename T, size_type I>
struct reference<declare::Array_<T, I> > { typedef declare::Array_<T, I> &type; };

template<typename T, size_type I>
struct reference<const declare::Array_<T, I> > { typedef declare::Array_<T, I> const &type; };

template<typename T, size_type I>
struct rank<declare::Array_<T, I> > : public index_const<I> {};

//template<typename V, size_type I>
//struct extents<declare::Array_<V, I> > : public index_sequence<I...> {};



template<typename T, size_type I>
struct value_type<declare::Array_<T, I> > { typedef T type; };

template<typename T, size_type I>
struct sub_type<declare::Array_<T, I> > { typedef std::conditional_t<I == 0, T, declare::Array_<T, I - 1> > type; };


template<typename T>
struct pod_type<declare::Array_<T, 0> > { typedef pod_type_t <T> type; };
template<typename T, size_type I>
struct pod_type<declare::Array_<T, I> > { typedef pod_type_t <declare::Array_<T, I - 1>> *type; };

}//namespace traits




namespace declare
{
template<typename V, size_type NDIMS, bool SLOW_FIRST>
struct Array_
{
private:
    typedef Array_<V, NDIMS, SLOW_FIRST> this_type;
    typedef calculus::calculator <this_type> calculator;
public:
    typedef V value_type;

    typedef V *pod_type;

    static constexpr bool is_slow_first = SLOW_FIRST;

    size_type m_dims_[NDIMS];
    size_type m_lower_[NDIMS];
    size_type m_upper_[NDIMS];

    V *m_data_;

    std::shared_ptr<V> m_data_holder_;
public:
    friend calculator;

    Array_() { calculator::initialize(this); }


    template<typename ...TID>
    explicit Array_(TID &&...idx):m_data_(nullptr), m_data_holder_(nullptr)
    {
        calculator::initialize(this, std::forward<TID>(idx)...);
    }

    template<typename ...TID>
    Array_(value_type *d, TID &&...idx):m_data_(d), m_data_holder_(d, simpla::tags::do_nothing())
    {
        calculator::initialize(this, std::forward<TID>(idx)...);
    }

    template<typename ...TID> Array_(std::shared_ptr<V> const &d, TID &&...idx):m_data_holder_(d), m_dims_{idx...} {}

    template<typename ...U>
    Array_(Expression<U...> const &expr) { calculator::apply(tags::_assign(), (*this), expr); }

    ~Array_() {}

    Array_(this_type const &other)
            : Array_(other.m_data_holder_, other.m_dims_, other.m_lower_, other.m_upper_) {}

    Array_(this_type &&other)
            : Array_(std::move(other.m_data_holder_), other.m_dims_, other.m_lower_, other.m_upper_) {}

    Array_(this_type &other, concept::tags::split const &split) : m_data_(nullptr), m_data_holder_(nullptr)
    {
        calculator::split(split, other, *this);
    }

    value_type *data() { return m_data_; }

    value_type const *data() const { return m_data_; }

    std::shared_ptr<value_type> &data_holder() { return m_data_holder_; }

    std::shared_ptr<value_type> const &data_holder() const { return m_data_holder_; }

    size_type const *dims() const { return m_dims_; }

    size_type const *lower() const { return m_lower_; }

    size_type const *upper() const { return m_upper_; }

    size_type ndims() const { return NDIMS; }

    template<typename ...TID> value_type &
    at(TID &&...s) { return m_data_[calculator::hash(m_dims_, m_lower_, std::forward<TID>(s)...)]; }

    template<typename ...TID> value_type const &
    at(TID &&...s) const { return m_data_[calculator::hash(m_dims_, m_lower_, std::forward<TID>(s)...)]; }

    template<typename TID>
    inline value_type &
    operator[](TID s) { return at(s); }

    template<typename TID>
    inline value_type const &
    operator[](TID s) const { return at(s); }


    template<typename ...TID> value_type &
    operator()(TID &&...s) { return at(std::forward<TID>(s)...); }

    template<typename ...TID> value_type const &
    operator()(TID &&...s) const { return at(std::forward<TID>(s)...); }


    void deploy() { calculator::deploy(*this); }

    void clear() { calculator::clear(*this); }

    void swap(this_type &other) { calculator::swap(*this, other); }

    inline this_type &
    operator=(this_type const &rhs)
    {
        calculator::apply(tags::_assign(), (*this), rhs);
        return (*this);
    }

    template<typename TR> inline this_type &
    operator=(TR const &rhs)
    {
        calculator::apply(tags::_assign(), (*this), rhs);
        return (*this);
    }

    template<typename TFun> void apply(TFun const &fun) { calculator::apply(fun, (*this)); }

    template<typename TFun> void apply(TFun const &fun) const { calculator::apply(fun, (*this)); }

    Array_<value_type, NDIMS> view(size_type const *il, size_type const *iu)
    {
        return calculator::view((*this), il, iu);
    }

    Array_<const value_type, NDIMS> view(size_type const *il, size_type const *iu) const
    {
        return calculator::view((*this), il, iu);
    }
};
} // namespace declare
namespace calculus
{
namespace st=simpla::traits;

template<typename V, size_type NDIMS, bool SLOW_FIRST>
struct calculator<declare::Array_<V, NDIMS, SLOW_FIRST> >
{

    typedef declare::Array_<V, NDIMS, SLOW_FIRST> self_type;
    typedef traits::value_type_t<self_type> value_type;

    static inline size_type size(size_type const *dims)
    {
        size_type res = 1;
        for (int i = 0; i < NDIMS; ++i) { res *= dims[i]; }
        return res;
    }

    static void deploy(declare::Array_<V, NDIMS> &self)
    {
        if (!self.m_data_holder_)
        {
            self.m_data_holder_ =
            #ifdef NDEBUG
                    sp_alloc_array<V>(size(self.m_dims_));
            #else
                    std::shared_ptr<V>(new V[size(self.m_dims_)]);
            #endif
        }
        self.m_data_ = self.m_data_holder_.get();
    };


    typedef std::integral_constant<bool, true> slow_first_t;

    typedef std::integral_constant<bool, false> fast_first_t;

    template<bool array_order> static inline size_type
    hash_(std::integral_constant<bool, array_order>, size_type const *dims, size_type const *offset) { return 0; }

    template<bool array_order> static inline size_type
    hash_(std::integral_constant<bool, array_order>, size_type const *dims, size_type const *offset,
          size_type s) { return s; }

    //fast first

    template<typename ...TID> static inline size_type
    hash_(fast_first_t, size_type const *dims, size_type const *offset, size_type i0, TID &&...idx)
    {
        return i0 + hash_(fast_first_t(), dims, offset, std::forward<TID>(idx)...) * dims[NDIMS - sizeof...(TID) - 1];
    }

    static inline size_type
    hash_(fast_first_t, index_const<NDIMS>, size_type const *dims, size_type const *offset,
          size_type const *i) { return 0; }


    template<size_type N> static inline size_type
    hash_(fast_first_t, index_const<N>, size_type const *dims, size_type const *offset, size_type const *i)
    {
        return i[N] - offset[N] + hash_(fast_first_t(), index_const<N + 1>(), dims, offset, i) * dims[N];
    }
    //slow first

    template<typename ...TID> static inline size_type
    hash_(slow_first_t, size_type const *dims, size_type const *offset, size_type s, size_type i1, TID &&...idx)
    {
        return hash_(slow_first_t(), dims, offset,
                     s * dims[NDIMS - (sizeof...(TID) + 1)] + i1 - offset[NDIMS - (sizeof...(TID) + 1)],
                     std::forward<TID>(idx)...);
    }

    static inline size_type
    hash_(slow_first_t, index_const<NDIMS>, size_type const *dims, size_type const *offset,
          size_type const *i) { return 0; }

    template<size_type N> static inline size_type
    hash_(slow_first_t, index_const<N>, size_type const *dims, size_type const *offset, size_type const *i)
    {
        return i[NDIMS - N - 1] - offset[NDIMS - N - 1] +
               hash_(slow_first_t(), index_const<N + 1>(), dims, offset, i) * dims[NDIMS - N - 1];
    }

    template<typename ...TID> static inline size_type
    hash(size_type const *dims, size_type const *offset, size_type s, TID &&...idx)
    {
        static_assert(NDIMS == (sizeof...(TID) + 1), "illegal index number!");
        return hash_(std::integral_constant<bool, SLOW_FIRST>(), dims, offset, s, std::forward<TID>(idx)...);
    }

    static inline size_type hash(size_type const *dims, size_type const *offset, size_type const *s)
    {
        auto t = hash_(std::integral_constant<bool, SLOW_FIRST>(), index_const<0>(), dims, offset, s);
        return t;
    }


public:
    template<typename T> static constexpr inline T &
    get_value(T &v) { return v; };

    static constexpr inline value_type &
    get_value(self_type &self, size_type const *s) { return self.m_data_[hash(self.m_dims_, self.m_lower_, s)]; };

    static constexpr inline value_type const &
    get_value(self_type const &self, size_type const *s) { return self.m_data_[hash(self.m_dims_, self.m_lower_, s)]; };

    template<typename T, typename I0> static constexpr inline st::remove_all_extents_t<T, I0> &
    get_value(T &v, I0 const *s,
              ENABLE_IF((st::is_indexable<T, I0>::value))) { return get_value(v[*s], s + 1); };

    template<typename T, typename I0> static constexpr inline T &
    get_value(T &v, I0 const *s, ENABLE_IF((!st::is_indexable<T, I0>::value))) { return v; };
private:
//    template<typename T, typename ...Args> static constexpr inline T &
//    get_value_(std::integral_constant<bool, false> const &, T &v, Args &&...)
//    {
//        return v;
//    }
//
//
//    template<typename T, typename I0, typename ...Idx> static constexpr inline st::remove_extents_t<T, I0, Idx...> &
//    get_value_(std::integral_constant<bool, true> const &, T &v, I0 const &s0, Idx &&...idx)
//    {
//        return get_value(v[s0], std::forward<Idx>(idx)...);
//    };
//public:
//    template<typename T, typename I0, typename ...Idx> static constexpr inline st::remove_extents_t<T, I0, Idx...> &
//    get_value(T &v, I0 const &s0, Idx &&...idx)
//    {
//        return get_value_(std::integral_constant<bool, st::is_indexable<T, I0>::value>(),
//                          v, s0, std::forward<Idx>(idx)...);
//    };
//
//    template<typename T, size_type N> static constexpr inline T &
//    get_value(declare::nTuple_<T, N> &v, size_type const &s0) { return v[s0]; };
//
//    template<typename T, size_type N> static constexpr inline T const &
//    get_value(declare::nTuple_<T, N> const &v, size_type const &s0) { return v[s0]; };
public:
    template<typename TOP, typename ...Others, size_type ... index, typename ...Idx> static constexpr inline auto
    _invoke_helper(declare::Expression<TOP, Others...> const &expr, index_sequence<index...>, Idx &&... s)
    DECL_RET_TYPE((TOP::eval(get_value(std::get<index>(expr.m_args_), std::forward<Idx>(s)...)...)))

    template<typename TOP, typename   ...Others, typename ...Idx> static constexpr inline auto
    get_value(declare::Expression<TOP, Others...> const &expr, Idx &&... s)
    DECL_RET_TYPE((_invoke_helper(expr, index_sequence_for<Others...>(), std::forward<Idx>(s)...)))


    template<typename TOP, typename ...Others, size_type ... index> static constexpr inline auto
    _invoke_helper(declare::Expression<TOP, Others...> const &expr, index_sequence<index...>, size_type const *s)
    DECL_RET_TYPE((TOP::eval(get_value(std::get<index>(expr.m_args_), s)...)))

    template<typename TOP, typename   ...Others> static constexpr inline auto
    get_value(declare::Expression<TOP, Others...> const &expr, size_type const *s)
    DECL_RET_TYPE((_invoke_helper(expr, index_sequence_for<Others...>(), s)))

    template<typename TFun> static void
    traversal_nd(size_type const *lower, size_type const *upper, TFun const &fun)
    {
/// FIXME: need parallelism
        size_type idx[NDIMS];
        for (int i = 0; i < NDIMS; ++i) { idx[i] = lower[i]; }

        while (1)
        {
            fun(idx);

            ++idx[NDIMS - 1];
            for (size_type rank = NDIMS - 1; rank > 0; --rank)
            {
                if (idx[rank] >= upper[rank])
                {
                    idx[rank] = lower[rank];
                    ++idx[rank - 1];
                }
            }
            if (idx[0] >= upper[0]) break;
        }
    }

    static constexpr inline
    void clear(self_type &self)
    {
        deploy(self);
        memset(self.m_data_, static_cast<int>(size(self.m_dims_) * sizeof(value_type)), 0);
    };

    template<typename TOP, typename ...Others> static constexpr inline
    void apply(TOP const &op, self_type &self, Others &&...others)
    {
        deploy(self);
        traversal_nd(
                self.m_lower_, self.m_upper_,
                [&](size_type const *idx)
                {
                    TOP::eval(get_value(self, idx), get_value(std::forward<Others>(others), idx)...);
                });
    };


    template<typename TOP> static constexpr inline
    void apply(TOP const &op, self_type &self)
    {
        traversal_nd(self.m_lower_, self.m_upper_, [&](size_type const *idx) { op(get_value(self, idx)); });
    };

    template<typename TOP> static constexpr inline
    void apply(TOP const &op, self_type const &self)
    {
        traversal_nd(self.m_lower_, self.m_upper_, [&](size_type const *idx) { op(get_value(self, idx)); });
    };

    static constexpr inline
    void swap(self_type &self, self_type &other)
    {
        std::swap(self.m_data_, other.m_data_);
        std::swap(self.m_data_holder_, other.m_data_holder_);

        for (int i = 0; i < NDIMS; ++i)
        {
            std::swap(self.m_dims_, other.m_dims_);
            std::swap(self.m_lower_, other.m_lower_);
            std::swap(self.m_upper_, other.m_upper_);

        }

    };

    static constexpr inline
    void split(concept::tags::split const &split, self_type &other, self_type &self)
    {
        self_type(other).swap(self);
        size_type max_dims = 0;
        int n = 0;
        for (int i = 0; i < NDIMS; ++i)
        {
            if (self.m_dims_[i] > max_dims)
            {
                n = i;
                max_dims = self.m_dims_[i];
            }
        }
        other.m_upper_[n] = other.m_lower_[n] +
                            (other.m_upper_[n] - other.m_lower_[n]) * split.left() / (split.left() + split.right());
        self.m_lower_[n] = other.m_upper_[n];
    }

    declare::Array_<V, NDIMS> view(self_type &self, size_type const *il, size_type const *iu)
    {
        return declare::Array_<V, NDIMS>(self.m_data_holder_, self.m_dims_, il, iu);
    };

    declare::Array_<const V, NDIMS> view(self_type const &self, size_type const *il, size_type const *iu)
    {
        return declare::Array_<V, NDIMS>(self.m_data_, self.m_dims_, il, iu);
    };

    static void
    initialize(self_type *self, size_type const *dims = nullptr, size_type const *lower = nullptr,
               size_type const *upper = nullptr)
    {
        for (int i = 0; i < NDIMS; ++i)
        {
            self->m_dims_[i] = dims == nullptr ? 1 : dims[i];
            self->m_lower_[i] = lower == nullptr ? 0 : lower[i];
            self->m_upper_[i] = upper == nullptr ? self->m_dims_[i] : upper[i];
        }
    }

    template<typename T0> static void copy_(T0 *dest) {};

    template<typename U, typename T0, typename ...Others> static void
    copy_(U *dest, T0 const &s0, Others &&...others)
    {
        dest[0] = static_cast<U>(s0);
        copy_(dest + 1, std::forward<Others>(others)...);
    };

    template<typename ...TID> static void
    initialize(self_type *self, size_type s0, TID &&...idx)
    {
        size_type dims[NDIMS];
        dims[0] = s0;
        copy_(dims + 1, std::forward<TID>(idx)...);
        initialize(self, dims);
    }
};
}//namespace calculus{


}}//namespace simpla{namespace algebra{namespace declare{
#endif //SIMPLA_ARRAY_H
