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
    array_index_box_type m_halo_index_box_{{0, 0, 0}, {1, 1, 1}};

    array_index_type m_strides_{0, 0, 0};
    size_type m_size_ = 0;
    index_type m_offset_ = 0;
    bool m_array_order_fast_first_ = false;

    ZSFC() = default;
    ~ZSFC() = default;

    ZSFC(this_type const& other)
        : m_index_box_(other.m_index_box_),
          m_halo_index_box_(other.m_halo_index_box_),
          m_strides_(other.m_strides_),
          m_size_(other.m_size_),
          m_offset_(other.m_offset_),
          m_array_order_fast_first_(other.m_array_order_fast_first_) {}
    ZSFC(this_type&& other)
    noexcept
        : m_index_box_(other.m_index_box_),
          m_halo_index_box_(other.m_halo_index_box_),
          m_strides_(other.m_strides_),
          m_size_(other.m_size_),
          m_offset_(other.m_offset_),
          m_array_order_fast_first_(other.m_array_order_fast_first_) {}

    explicit ZSFC(array_index_box_type const& b, bool array_order_fast_first = false)
        : m_halo_index_box_(b), m_index_box_(b), m_array_order_fast_first_(array_order_fast_first) {
        Update();
    }
    this_type Overlap(std::nullptr_t) const { return *this; }
    template <typename RHS>
    this_type Overlap(RHS const& rhs) const;
    this_type Overlap(this_type const& rhs) const;
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
        m_halo_index_box_.swap(other.m_halo_index_box_);

        m_strides_.swap(other.m_strides_);
        std::swap(m_offset_, other.m_offset_);
        std::swap(m_size_, other.m_size_);
        std::swap(m_array_order_fast_first_, other.m_array_order_fast_first_);
    }

    bool empty() const { return m_size_ == 0; }
    void reset() {
        std::get<0>(m_index_box_) = 0;
        std::get<1>(m_index_box_) = 0;
        std::get<0>(m_halo_index_box_) = 0;
        std::get<1>(m_halo_index_box_) = 0;
        m_offset_ = 0;
        m_strides_ = 0;
        m_size_ = 0;
    }

    void reset(index_box_type const& b) {
        reset();
        m_halo_index_box_ = b;
        Update();
    }
    void reset(index_type const* lo, index_type const* hi) {
        for (int i = 0; i < NDIMS; ++i) {
            std::get<0>(m_halo_index_box_)[i] = lo[i];
            std::get<1>(m_halo_index_box_)[i] = hi[i];
        }

        Update();
    }
    void Update() {
        if (m_array_order_fast_first_) {
            m_strides_[0] = 1;
            for (int i = 1; i < NDIMS; ++i) {
                m_strides_[i] =
                    m_strides_[i - 1] * (std::get<1>(m_halo_index_box_)[i - 1] - std::get<0>(m_halo_index_box_)[i - 1]);
            }
        } else {
            m_strides_[NDIMS - 1] = 1;
            for (int i = NDIMS - 2; i >= 0; --i) {
                m_strides_[i] =
                    m_strides_[i + 1] * (std::get<1>(m_halo_index_box_)[i + 1] - std::get<0>(m_halo_index_box_)[i + 1]);
            }
        }
        m_size_ = 1;
        for (int i = 0; i < NDIMS; ++i) {
            m_size_ *= (std::get<1>(m_halo_index_box_)[i] - std::get<0>(m_halo_index_box_)[i]);
        }
        m_index_box_ = m_halo_index_box_;
    };
    size_type GetNDIMS() const { return static_cast<size_type>(ndims); }
    size_type GetIndexBox(index_type* lo, index_type* hi) const {
        for (int i = 0; i < ndims; ++i) {
            lo[i] = std::get<0>(m_index_box_)[i];
            hi[i] = std::get<1>(m_index_box_)[i];
        }
        return static_cast<size_type>(ndims);
    }
    size_type GetHaloIndexBox(index_type* lo, index_type* hi) const {
        for (int i = 0; i < ndims; ++i) {
            lo[i] = std::get<0>(m_halo_index_box_)[i];
            hi[i] = std::get<1>(m_halo_index_box_)[i];
        }
        return static_cast<size_type>(ndims);
    }
    auto GetIndexBox() const { return m_index_box_; }
    auto GetHaloIndexBox() const { return m_halo_index_box_; }

    template <typename LHS, typename RHS>
    size_type Copy(LHS& dst, RHS const& src) const;

    size_type size() const { return m_size_; }

    void Select(index_type const* lo, index_type const* hi) {
        for (int i = 0; i < ndims; ++i) {
            std::get<0>(m_index_box_)[i] = lo[i];
            std::get<1>(m_index_box_)[i] = hi[i];
        }
    }

    void Select(std::tuple<nTuple<index_type, NDIMS>, nTuple<index_type, NDIMS>> const& b) {
        Select(&std::get<0>(b)[0], &std::get<1>(b)[0]);
    }

    void Select(std::initializer_list<std::initializer_list<index_type>> const& b) {
        ASSERT(b.size() == 2);
        auto it = b.begin();
        std::vector<index_type> lo(*it);
        ++it;
        std::vector<index_type> hi(*it);

        for (auto i = lo.size(); i < ndims; ++i) { lo.push_back(std::get<0>(m_halo_index_box_)[i]); }
        for (auto i = hi.size(); i < ndims; ++i) { lo.push_back(std::get<1>(m_halo_index_box_)[i]); }

        Select(&lo[0], &hi[0]);
    }

    void Shift(index_type const* offset) {
        for (int i = 0; i < ndims; ++i) {
            std::get<0>(m_index_box_)[i] += offset[i];
            std::get<1>(m_index_box_)[i] += offset[i];
            std::get<0>(m_halo_index_box_)[i] += offset[i];
            std::get<1>(m_halo_index_box_)[i] += offset[i];
        }
        Update();
    }

    void Shift(std::initializer_list<index_type> const& idx) {
        std::vector<index_type> s;
        for (auto const& v : idx) { s.push_back(v); }
        for (auto i = s.size(); i < ndims; ++i) { s.push_back(0); }
        Shift(&s[0]);
    }

    void Shift(array_index_type const& offset) { Shift(&offset[0]); }

    template <typename... Args>
    this_type GetShift(Args&&... args) const {
        this_type res(*this);
        res.Shift(std::forward<Args>(args)...);
        return res;
    }
    template <typename... Args>
    this_type GetSelection(Args&&... args) const {
        this_type res(*this);
        res.Select(std::forward<Args>(args)...);
        return res;
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
    size_type Foreach(const TFun& fun) const;
};

template <>
__host__ __device__ inline index_type ZSFC<3>::hash(index_type s0, index_type s1, index_type s2, index_type s3,
                                                    index_type s4, index_type s5, index_type s6, index_type s7,
                                                    index_type s8, index_type s9) const {
    return ((s0 - std::get<0>(m_halo_index_box_)[0]) * m_strides_[0] +
            (s1 - std::get<0>(m_halo_index_box_)[1]) * m_strides_[1] +
            (s2 - std::get<0>(m_halo_index_box_)[2]) * m_strides_[2]);
}
template <>
__host__ __device__ inline constexpr bool ZSFC<3>::in_box(index_type s0, index_type s1, index_type s2, index_type s3,
                                                          index_type s4, index_type s5, index_type s6, index_type s7,
                                                          index_type s8, index_type s9) const {
    return (std::get<0>(m_halo_index_box_)[0] <= s0) && (s0 < std::get<1>(m_halo_index_box_)[0]) &&
           (std::get<0>(m_halo_index_box_)[1] <= s1) && (s1 < std::get<1>(m_halo_index_box_)[1]) &&
           (std::get<0>(m_halo_index_box_)[2] <= s2) && (s2 < std::get<1>(m_halo_index_box_)[2]);
}

template <>
__host__ __device__ constexpr inline bool ZSFC<3>::in_box(array_index_type const& idx) const {
    return in_box(idx[0], idx[1], idx[2]);
};

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
    return a.GetSpaceFillingCurve().GetIndexBox();
}

inline index_box_type overlap(index_box_type const& a) { return a; }
inline index_box_type overlap(index_box_type const& b0, index_box_type const& b1) {
    return index_box_type{{std::max(std::get<0>(b0)[0], std::get<0>(b1)[0]),   //
                           std::max(std::get<0>(b0)[1], std::get<0>(b1)[1]),   //
                           std::max(std::get<0>(b0)[2], std::get<0>(b1)[2])},  //
                          {std::min(std::get<1>(b0)[0], std::get<1>(b1)[0]),   //
                           std::min(std::get<1>(b0)[1], std::get<1>(b1)[1]),   //
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

}  // namespace detail {
template <>
template <typename RHS>
ZSFC<3> ZSFC<3>::Overlap(RHS const& rhs) const {
    ZSFC<3> res(m_halo_index_box_);
    res.Select(detail::overlap(m_index_box_, rhs));
    return res;
};
template <int NDIMS>
ZSFC<NDIMS> ZSFC<NDIMS>::Overlap(this_type const& rhs) const {
    ZSFC<NDIMS> res(m_halo_index_box_);
    res.Select(detail::overlap(m_index_box_, rhs.m_index_box_));
    return res;
}

template <>
template <typename LHS, typename RHS>
size_type ZSFC<3>::Copy(LHS& dst, RHS const& src) const {
    auto count = size();
    if (count == 0) { return count; }
    index_type ib = std::get<0>(m_index_box_)[0];
    index_type ie = std::get<1>(m_index_box_)[0];
    index_type jb = std::get<0>(m_index_box_)[1];
    index_type je = std::get<1>(m_index_box_)[1];
    index_type kb = std::get<0>(m_index_box_)[2];
    index_type ke = std::get<1>(m_index_box_)[2];
    count = static_cast<size_type>((ke - kb) * (je - jb) * (ie - ib));

    if (m_array_order_fast_first_) {
        //#pragma omp parallel for
        for (index_type k = kb; k < ke; ++k)
            for (index_type j = jb; j < je; ++j)
                for (index_type i = ib; i < ie; ++i) { dst(i, j, k) = src(i, j, k); }

    } else {
        //#pragma omp parallel for
        for (index_type i = ib; i < ie; ++i)
            for (index_type j = jb; j < je; ++j)
                for (index_type k = kb; k < ke; ++k) { dst(i, j, k) = src(i, j, k); }
    }
    return count;
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
size_type ZSFC<3>::Foreach(const TFun& fun) const {
    auto count = size();
    if (count == 0) { return count; }
    index_type ib = std::get<0>(m_index_box_)[0];
    index_type ie = std::get<1>(m_index_box_)[0];
    index_type jb = std::get<0>(m_index_box_)[1];
    index_type je = std::get<1>(m_index_box_)[1];
    index_type kb = std::get<0>(m_index_box_)[2];
    index_type ke = std::get<1>(m_index_box_)[2];
    count = static_cast<size_type>((ke - kb) * (je - jb) * (ie - ib));
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
    return count;
}

}  // namespace simpla
#endif  // SIMPLA_Z_SFC_H
