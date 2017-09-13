//
// Created by salmon on 17-6-24.
//

#ifndef SIMPLA_Z_SFC_H
#define SIMPLA_Z_SFC_H

#include "simpla/SIMPLA_config.h"

#include <algorithm>
#include <cmath>
#include <cstddef>  // for size_t
#include <iomanip>
#include <limits>
#include <tuple>
#include "simpla/algebra/EntityId.h"
#include "simpla/algebra/nTuple.ext.h"
#include "simpla/algebra/nTuple.h"
#include "simpla/utilities/memory.h"

namespace simpla {
template <typename V, typename SFC>
struct Array;
template <int NDIMS>
class ZSFC {
    typedef ZSFC<NDIMS> this_type;

   public:
    typedef nTuple<index_type, NDIMS> array_index_type;
    typedef std::tuple<array_index_type, array_index_type> array_index_box_type;
    static constexpr int ndims = NDIMS;
    array_index_box_type m_index_box_{{0, 0, 0}, {1, 1, 1}};
    array_index_type m_strides_{0, 0, 0};
    size_type m_size_ = 0;
    bool m_array_order_fast_first_ = false;

    ZSFC() = default;
    ~ZSFC() = default;

    ZSFC(this_type const& other)
        : m_index_box_(other.m_index_box_),
          m_strides_(other.m_strides_),
          m_size_(other.m_size_),
          m_array_order_fast_first_(other.m_array_order_fast_first_) {}
    ZSFC(this_type&& other)
    noexcept
        : m_index_box_(other.m_index_box_),
          m_strides_(other.m_strides_),
          m_size_(other.m_size_),
          m_array_order_fast_first_(other.m_array_order_fast_first_) {}

    template <typename RHS>
    this_type Overlap(RHS const& rhs) const;

    //    ZSFC(std::initializer_list<index_type> const& l) {
    //        for (int i = 0; i < NDIMS; ++i) {
    //            std::get<0>(m_index_box_)[i] = 0;
    //            std::get<1>(m_index_box_)[i] = 1;
    //        }
    //        int count = 0;
    //        for (auto const& v : l) {
    //            if (count >= NDIMS) { break; }
    //            std::get<1>(m_index_box_)[count] = v;
    //            ++count;
    //        }
    //        //        std::get<0>(m_index_box_) = std::get<0>(m_index_box_);
    //        //        std::get<1>(m_index_box_) = std::get<1>(m_index_box_);
    //        DoSetUp();
    //    }

    explicit ZSFC(array_index_box_type const& b, bool array_order_fast_first = false)
        : m_index_box_(b), m_array_order_fast_first_(array_order_fast_first) {
        Update();
    }

    this_type& operator=(this_type const& other) {
        this_type(other).swap(*this);
        return *this;
    };
    this_type& operator=(this_type&& other) noexcept {
        this_type(other).swap(*this);
        return *this;
    };
    void swap(ZSFC& other) {
        m_index_box_.swap(other.m_index_box_);
        m_strides_.swap(other.m_strides_);
        std::swap(m_size_, other.m_size_);
        std::swap(m_array_order_fast_first_, other.m_array_order_fast_first_);
    }
    bool empty() const { return m_size_ == 0; }
    void reset() {
        std::get<0>(m_index_box_) = 0;
        std::get<1>(m_index_box_) = 0;
        m_strides_ = 0;
        m_size_ = 0;
    }

    void reset(index_box_type const& b) {
        m_index_box_ = b;
        Update();
    }

    void Update() {
        if (m_array_order_fast_first_) {
            m_strides_[0] = 1;
            for (int i = 1; i < NDIMS; ++i) {
                m_strides_[i] =
                    m_strides_[i - 1] * (std::get<1>(m_index_box_)[i - 1] - std::get<0>(m_index_box_)[i - 1]);
            }
        } else {
            m_strides_[NDIMS - 1] = 1;
            for (int i = NDIMS - 2; i >= 0; --i) {
                m_strides_[i] =
                    m_strides_[i + 1] * (std::get<1>(m_index_box_)[i + 1] - std::get<0>(m_index_box_)[i + 1]);
            }
        }
        m_size_ = 1;
        for (int i = 0; i < NDIMS; ++i) { m_size_ *= (std::get<1>(m_index_box_)[i] - std::get<0>(m_index_box_)[i]); }
    };
    size_type GetNDIMS() const { return ndims; }
    size_type GetIndexBox(index_type* lo, index_type* hi) const {
        for (int i = 0; i < ndims; ++i) {
            lo[i] = std::get<0>(m_index_box_)[i];
            hi[i] = std::get<1>(m_index_box_)[i];
        }
        return ndims;
    }
    template <typename U, typename... Args>
    size_type Copy(U* dst, Args&&... src) const;
    //    template <typename U>
    //    size_type Copy(U* data, this_type const& other_sfc, U const* other, index_type const* lo = nullptr,
    //                   index_type const* hi = nullptr) const;

    size_type size() const { return m_size_; }
    index_box_type IndexBox() const { return m_index_box_; }
    void Shift(array_index_type const& offset) {
        std::get<0>(m_index_box_) += offset;
        std::get<1>(m_index_box_) += offset;
        Update();
    }

    __host__ __device__ constexpr inline index_type hash() const { return 0; }

    __host__ __device__ constexpr inline index_type hash(array_index_type const& idx) const {
        return hash(idx[0], idx[1], idx[2]);
    }

    __host__ __device__ inline index_type hash(EntityId s) const { return hash(s.x, s.y, s.z); }

    __host__ __device__ inline index_type hash(index_type const* idx) const { return hash(idx[0], idx[1], idx[2]); }

    __host__ __device__ inline index_type hash(index_type s0, index_type s1 = 0, index_type s2 = 0, index_type s3 = 0,
                                               index_type s4 = 0, index_type s5 = 0, index_type s6 = 0,
                                               index_type s7 = 0, index_type s8 = 0, index_type s9 = 0) const;

    __host__ __device__ constexpr inline bool in_box(array_index_type const& x) const;

    __host__ __device__ constexpr inline bool in_box(index_type s0, index_type s1 = 0, index_type s2 = 0,
                                                     index_type s3 = 0, index_type s4 = 0, index_type s5 = 0,
                                                     index_type s6 = 0, index_type s7 = 0, index_type s8 = 0,
                                                     index_type s9 = 0) const;

    template <typename V, typename... Args>
    __host__ __device__ V& Get(V* p, Args&&... args) const {
#ifdef ENABLE_BOUND_CHECK
        auto s = hash(std::forward<Args>(args)...);
        return (s < m_size_) ? p[s] : m_null_;
#else
        return p[hash(std::forward<Args>(args)...)];
#endif
    }

    template <typename TFun>
    void Foreach(const TFun& fun) const;

    template <typename value_type>
    std::ostream& Print(std::ostream& os, value_type const* v, int indent = 0) const;
};
template <int N>
template <typename U, typename... Args>
size_type ZSFC<N>::Copy(U* dst, Args&&... src) const {
    FIXME;

    return 0;
};

template <>
__host__ __device__ inline index_type ZSFC<3>::hash(index_type s0, index_type s1, index_type s2, index_type s3,
                                                    index_type s4, index_type s5, index_type s6, index_type s7,
                                                    index_type s8, index_type s9) const {
    return ((s0 - std::get<0>(m_index_box_)[0]) * m_strides_[0] + (s1 - std::get<0>(m_index_box_)[1]) * m_strides_[1] +
            (s2 - std::get<0>(m_index_box_)[2]) * m_strides_[2]);
}
template <>
__host__ __device__ inline constexpr bool ZSFC<3>::in_box(index_type s0, index_type s1, index_type s2, index_type s3,
                                                          index_type s4, index_type s5, index_type s6, index_type s7,
                                                          index_type s8, index_type s9) const {
    return (std::get<0>(m_index_box_)[0] <= s0) && (s0 < std::get<1>(m_index_box_)[0]) &&
           (std::get<0>(m_index_box_)[1] <= s1) && (s1 < std::get<1>(m_index_box_)[1]) &&
           (std::get<0>(m_index_box_)[2] <= s2) && (s2 < std::get<1>(m_index_box_)[2]);
}

template <>
__host__ __device__ constexpr inline bool ZSFC<3>::in_box(array_index_type const& idx) const {
    return in_box(idx[0], idx[1], idx[2]);
};

template <>
template <typename value_type>
std::ostream& ZSFC<3>::Print(std::ostream& os, value_type const* v, int indent) const {
    os << "Array<" << simpla::traits::type_name<value_type>::value() << ">" << m_index_box_;
    if (v != nullptr && size() < 20) {
        index_type ib = std::get<0>(m_index_box_)[0];
        index_type ie = std::get<1>(m_index_box_)[0];
        index_type jb = std::get<0>(m_index_box_)[1];
        index_type je = std::get<1>(m_index_box_)[1];
        index_type kb = std::get<0>(m_index_box_)[2];
        index_type ke = std::get<1>(m_index_box_)[2];

        for (index_type i = ib; i < ie; ++i)
            for (index_type j = jb; j < je; ++j) {
                os << "{" << std::setw(8) << v[hash(i, j, kb)];
                for (index_type k = kb + 1; k < ke; ++k) { os << "," << std::setw(8) << v[hash(i, j, k)]; }
                os << "}" << std::endl;
            }
    }
    return os;
}
// template <>
// template <typename U>
// size_type ZSFC<1>::Copy(U* data, this_type const& other_sfc, U const* other, index_type const* lo,
//                        index_type const* hi) const {
//    FIXME;
//    return 0;
//}
// template <>
// template <typename U>
// size_type ZSFC<2>::Copy(U* data, this_type const& other_sfc, U const* other, index_type const* lo,
//                        index_type const* hi) const {
//    FIXME;
//    return 0;
//}
// template <>
// template <typename U>
// size_type ZSFC<3>::Copy(U* data, this_type const& other_sfc, U const* other, index_type const* lo,
//                        index_type const* hi) const {
//    FIXME;
//    return 0;
//}
// template <int N>
// template <typename TV, typename TFun>
// void ZSFC<N>::Foreach(TV* d, TFun const& fun) const {
//    UNIMPLEMENTED;
//    //    nTuple<index_type, N> idx;
//    //    idx = std::get<0>(inner_box);
//    //
//    //    while (1) {
//    //        fun(idx);
//    //
//    //        ++idx[N - 1];
//    //        for (int rank = N - 1; rank > 0; --rank) {
//    //            if (idx[rank] >= std::get<1>(inner_box)[rank]) {
//    //                idx[rank] = std::get<0>(inner_box)[rank];
//    //                ++idx[rank - 1];
//    //            }
//    //        }
//    //        if (idx[0] >= std::get<1>(inner_box)[0]) break;
//    //    }
//}
// template <>
// template <typename TFun>
// void ZSFC<1>::Foreach(TV* d, TFun const& fun) const {
//    index_type ib = std::get<0>(m_index_box_)[0];
//    index_type ie = std::get<1>(m_index_box_)[0];
//    //#pragma omp parallel for
//    for (index_type i = ib; i < ie; ++i) { fun(nTuple<index_type, 1>{i}); }
//}
// template <>
// template <typename TFun>
// void ZSFC<2>::Foreach(TV* d, TFun const& fun) const {
//    index_type ib = std::get<0>(m_index_box_)[0];
//    index_type ie = std::get<1>(m_index_box_)[0];
//    index_type jb = std::get<0>(m_index_box_)[1];
//    index_type je = std::get<1>(m_index_box_)[1];
//    if (m_array_order_fast_first_) {
//        //#pragma omp parallel for
//        for (index_type j = jb; j < je; ++j)
//            for (index_type i = ib; i < ie; ++i) { fun(nTuple<index_type, 2>{i, j}); }
//    } else {
//        //#pragma omp parallel for
//        for (index_type i = ib; i < ie; ++i)
//            for (index_type j = jb; j < je; ++j) { fun(nTuple<index_type, 2>{i, j}); }
//    }
//}

namespace detail {

inline index_box_type overlap() {
    return index_box_type{{std::numeric_limits<index_type>::min(), std::numeric_limits<index_type>::min(),
                           std::numeric_limits<index_type>::min()},
                          {std::numeric_limits<index_type>::max(), std::numeric_limits<index_type>::max(),
                           std::numeric_limits<index_type>::max()}};
}
template <typename T>
index_box_type overlap(T const& a) {
    return overlap();
}
template <typename U>
index_box_type overlap(Array<U, ZSFC<3>> const& a) {
    return a.GetSpaceFillingCurve().IndexBox();
}

inline index_box_type overlap(index_box_type const& a) { return a; }
inline index_box_type overlap(index_box_type const& b0, index_box_type const& b1) {
    return index_box_type{
        {std::max(std::get<0>(b0)[0], std::get<0>(b1)[0]), std::max(std::get<0>(b0)[1], std::get<0>(b1)[1]),
         std::max(std::get<0>(b0)[2], std::get<0>(b1)[2])},
        {std::min(std::get<1>(b0)[0], std::get<1>(b1)[0]), std::min(std::get<1>(b0)[1], std::get<1>(b1)[1]),
         std::min(std::get<1>(b0)[2], std::get<1>(b1)[2])}};
}
template <typename TOP, typename... Args>
index_box_type overlap(Expression<TOP, Args...> const& expr);

template <typename Arg0, typename Arg1, typename... Others>
index_box_type overlap(Arg0 const& first, Arg1 const& arg1, Others&&... others) {
    return overlap(overlap(first), overlap(arg1, std::forward<Others>(others)...));
}
template <size_t... index, typename... Args>
index_box_type _overlap(std::index_sequence<index...>, std::tuple<Args...> const& expr) {
    return overlap(std::get<index>(expr)...);
}
template <typename TOP, typename... Args>
index_box_type overlap(Expression<TOP, Args...> const& expr) {
    return _overlap(std::index_sequence_for<Args...>(), expr.m_args_);
}

}  // namespace traits {
template <>
template <typename RHS>
ZSFC<3> ZSFC<3>::Overlap(RHS const& rhs) const {
    return ZSFC<3>(detail::overlap(m_index_box_, rhs));
};
#ifdef __CUDA__
template <typename TFUN>
__global__ void foreach_device(nTuple<index_type, 3> min, nTuple<index_type, 3> max, TFUN fun) {
    nTuple<index_type, 3> idx{min[0] + blockIdx.x * blockDim.x + threadIdx.x,
                              min[1] + blockIdx.y * blockDim.y + threadIdx.y,
                              min[2] + blockIdx.z * blockDim.z + threadIdx.z};
    if (idx[0] < max[0] && idx[1] < max[1] && idx[2] < max[2]) { fun(idx); }
};

#endif
template <>
template <typename TFun>
void ZSFC<3>::Foreach(const TFun& fun) const {
    index_type ib = std::get<0>(m_index_box_)[0];
    index_type ie = std::get<1>(m_index_box_)[0];
    index_type jb = std::get<0>(m_index_box_)[1];
    index_type je = std::get<1>(m_index_box_)[1];
    index_type kb = std::get<0>(m_index_box_)[2];
    index_type ke = std::get<1>(m_index_box_)[2];

#ifndef __CUDA__
    if (m_array_order_fast_first_) {
        //#pragma omp parallel for
        for (index_type k = kb; k < ke; ++k)
            for (index_type j = jb; j < je; ++j)
                for (index_type i = ib; i < ie; ++i) { fun(i, j, k); }

    } else {
        //#pragma omp parallel for
        for (index_type i = ib; i < ie; ++i)
            for (index_type j = jb; j < je; ++j)
                for (index_type k = kb; k < ke; ++k) { fun(i, j, k); }
    }
#else

    dim3 threadsPerBlock{4, 4, 4};

    dim3 numBlocks{static_cast<uint>(std::get<1>(m_index_box_)[0] - std::get<0>(m_index_box_)[0] + threadsPerBlock.x) /
                       threadsPerBlock.x,
                   static_cast<uint>(std::get<1>(m_index_box_)[1] - std::get<0>(m_index_box_)[1] + threadsPerBlock.y) /
                       threadsPerBlock.y,
                   static_cast<uint>(std::get<1>(m_index_box_)[2] - std::get<0>(m_index_box_)[2] + threadsPerBlock.z) /
                       threadsPerBlock.z};

    SP_CALL_DEVICE_KERNEL(foreach_device, numBlocks, threadsPerBlock, std::get<0>(m_index_box_),
                          std::get<1>(m_index_box_), fun);

#endif
}

// template <>
// constexpr inline bool ZSFC<3>::in_box(index_type x, index_type y, index_type z) const {
//    return (std::get<0>(m_index_box_)[0] <= x) && (x < std::get<1>(m_index_box_)[0]) &&
//           (std::get<0>(m_index_box_)[1] <= y) && (y < std::get<1>(m_index_box_)[1]) &&
//           (std::get<0>(m_index_box_)[2] <= z) && (z < std::get<1>(m_index_box_)[2]);
//}
// template <>
// template <typename TFun>
// void ZSFC<4>::Foreach(TFun const& fun) const {
//    index_type ib = std::get<0>(m_index_box_)[0];
//    index_type ie = std::get<1>(m_index_box_)[0];
//    index_type jb = std::get<0>(m_index_box_)[1];
//    index_type je = std::get<1>(m_index_box_)[1];
//    index_type kb = std::get<0>(m_index_box_)[2];
//    index_type ke = std::get<1>(m_index_box_)[2];
//    index_type lb = std::get<0>(m_index_box_)[3];
//    index_type le = std::get<1>(m_index_box_)[3];
//
//    if (m_array_order_fast_first_) {
//#pragma omp parallel for
//        for (index_type l = lb; l < le; ++l)
//            for (index_type k = kb; k < ke; ++k)
//                for (index_type j = jb; j < je; ++j)
//                    for (index_type i = ib; i < ie; ++i) fun(nTuple<index_type, 4>{i, j, k, l});
//
//    } else {
//#pragma omp parallel for
//        for (index_type i = ib; i < ie; ++i)
//            for (index_type j = jb; j < je; ++j)
//                for (index_type k = kb; k < ke; ++k)
//                    for (index_type l = lb; l < le; ++l) { fun(nTuple<index_type, 4>{i, j, k, l}); }
//    }
//}

}  // namespace simpla
#endif  // SIMPLA_Z_SFC_H
