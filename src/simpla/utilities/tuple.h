//
// Created by salmon on 17-6-22.
//

#ifndef SIMPLA_TUPLE_H
#define SIMPLA_TUPLE_H
namespace simpla {
template <class... T>
class tuple;
struct null_type;

template <>
class tuple<> {
   public:
    std::nullptr_t m_first_ = nullptr;
    std::nullptr_t m_second_ = nullptr;
    /*! \p tuple's no-argument constructor initializes each element.
     */
    inline __host__ __device__ tuple(void) {}

    __host__ __device__ tuple(const tuple<>& t0) {}

    inline __host__ __device__ tuple& operator=(const tuple<>& k) { return *this; }

    inline __host__ __device__ void swap(tuple& t) {}
};
template <typename First, class... Others>
class tuple<First, Others...> {
   public:
    First m_first_;
    tuple<Others...> m_second_;
    /*! \p tuple's no-argument constructor initializes each element.
     */

    template <typename U0, typename... U>
    __host__ __device__ tuple(U0&& first, U&&... others)
        : m_first_(std::forward<U0>(first)), m_second_(std::forward<U>(others)...) {}

    template <typename... U>
    __host__ __device__ tuple(const tuple<U...>& t0) {}

    template <class... U>
    inline __host__ __device__ tuple& operator=(const tuple<U...>& k) {
        return *this;
    }

    inline __host__ __device__ void swap(tuple& t) {}
};

/*! \p swap swaps the contents of two <tt>tuple</tt>s.
 *
 *  \param x The first \p tuple to swap.
 *  \param y The second \p tuple to swap.
 */
template <typename... Args>
inline __host__ __device__ void swap(tuple<Args...>& x, tuple<Args...>& y) {}

/*! \cond
 */

template <typename... Args>
__host__ __device__ inline tuple<Args...> make_tuple(Args&&... args) {
    return tuple<Args...>(std::forward<Args>(args)...);
};

template <typename... Args>
__host__ __device__ inline tuple<Args&...> tie(Args&&... args) {
    return tuple<Args&...>(std::forward<Args>(args)...);
};

__host__ __device__ inline bool operator==(const null_type&, const null_type&);

__host__ __device__ inline bool operator>=(const null_type&, const null_type&);

__host__ __device__ inline bool operator<=(const null_type&, const null_type&);

__host__ __device__ inline bool operator!=(const null_type&, const null_type&);

__host__ __device__ inline bool operator<(const null_type&, const null_type&);

__host__ __device__ inline bool operator>(const null_type&, const null_type&);
//
// namespace detail {
//
//    template <int N, typename... T>
//    auto& get(tuple<T...>& t) {
//        return t.first;
//    };
//}
template <int N, typename... T>
auto& get(tuple<T...>& t) {
    return t.first;
};

/*! \endcond
 */

/*! \} // tuple
 */

/*! \} // utility
 */
}
#endif  // SIMPLA_TUPLE_H
