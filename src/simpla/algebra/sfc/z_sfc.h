//
// Created by salmon on 17-6-24.
//

#ifndef SIMPLA_Z_SFC_H
#define SIMPLA_Z_SFC_H

#include <algorithm>
#include <cmath>
#include <cstddef>  // for size_t
#include <iomanip>
#include <limits>
#include <tuple>
#include <utility>
#include "simpla/SIMPLA_config.h"
#include "simpla/algebra/EntityId.h"
#include "simpla/algebra/nTuple.ext.h"
#include "simpla/algebra/nTuple.h"
#include "simpla/utilities/memory.h"

namespace simpla {
enum enumArrayOrder { SLOW_FIRST = 0, FAST_FIRST = 1 };

template <typename V, typename SFC>
struct Array;
template <int NDIMS>
class ZSFC {
    typedef ZSFC<NDIMS> this_type;

   public:
    static constexpr int ndims = NDIMS;
    enum enumArrayOrder m_array_order_ = SLOW_FIRST;

    index_type m_index_min_[NDIMS] = {0};
    index_type m_index_max_[NDIMS] = {1};
    index_type m_shape_min_[NDIMS] = {0};
    index_type m_shape_max_[NDIMS] = {1};
    index_type m_strides_[NDIMS] = {1};

    ZSFC() = default;
    ~ZSFC() = default;

    ZSFC(this_type const& other) : m_array_order_(other.m_array_order_) {
        for (int i = 0; i < NDIMS; ++i) {
            m_index_min_[i] = other.m_index_min_[i];
            m_index_max_[i] = other.m_index_max_[i];
            m_shape_min_[i] = other.m_shape_min_[i];
            m_shape_max_[i] = other.m_shape_max_[i];
            m_strides_[i] = other.m_strides_[i];
        }
    }
    ZSFC(this_type&& other) noexcept : m_array_order_(other.m_array_order_) {
        for (int i = 0; i < NDIMS; ++i) {
            m_index_min_[i] = other.m_index_min_[i];
            m_index_max_[i] = other.m_index_max_[i];
            m_shape_min_[i] = other.m_shape_min_[i];
            m_shape_max_[i] = other.m_shape_max_[i];
            m_strides_[i] = other.m_strides_[i];
        }
    }

    ZSFC(index_type const* lo, index_type const* hi, enumArrayOrder array_order = SP_ARRAY_DEFAULT_ORDER)
        : m_array_order_(array_order) {
        for (int i = 0; i < NDIMS; ++i) {
            m_index_min_[i] = lo[i];
            m_index_max_[i] = hi[i];
            m_shape_min_[i] = lo[i];
            m_shape_max_[i] = hi[i];
            m_strides_[i] = 1;
        }

        if (m_array_order_ == FAST_FIRST) {
            m_strides_[0] = 1;
            for (int i = 1; i < NDIMS; ++i) {
                m_strides_[i] = m_strides_[i - 1] * (m_shape_max_[i - 1] - m_shape_min_[i - 1]);
            }
        } else {
            m_strides_[NDIMS - 1] = 1;
            for (int i = NDIMS - 2; i >= 0; --i) {
                m_strides_[i] = m_strides_[i + 1] * (m_shape_max_[i + 1] - m_shape_min_[i + 1]);
            }
        }
    }

    explicit ZSFC(std::tuple<nTuple<index_type, NDIMS>, nTuple<index_type, NDIMS>> const& d,
                  enumArrayOrder array_order = SP_ARRAY_DEFAULT_ORDER)
        : ZSFC(&std::get<0>(d)[0], &std::get<1>(d)[0], array_order) {}

    ZSFC(std::initializer_list<index_type> const& extents, enumArrayOrder array_order = SP_ARRAY_DEFAULT_ORDER) {
        index_type lo[NDIMS];
        index_type hi[NDIMS];
        for (int i = 0; i < NDIMS; ++i) { lo[i] = 0; }
        int count = 0;
        for (auto const& v : extents) { hi[count] = v; }
        reset(lo, hi, NDIMS);
    }
    ZSFC(std::initializer_list<std::initializer_list<index_type>> const& extents,
         enumArrayOrder array_order = SP_ARRAY_DEFAULT_ORDER) {
        auto const& lo_list = *extents.begin();
        auto const& hi_list = *(extents.begin() + 1);

        index_type lo[NDIMS];
        index_type hi[NDIMS];
        for (int i = 0; i < NDIMS; ++i) { lo[i] = 0; }
        int count = 0;
        for (auto const& v : lo_list) { lo[count] = v; }
        count = 0;
        for (auto const& v : hi_list) { hi[count] = v; }
        reset(lo, hi, NDIMS);
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
        std::swap(m_array_order_, other.m_array_order_);
        for (int i = 0; i < NDIMS; ++i) {
            std::swap(m_index_min_[i], other.m_index_min_[i]);
            std::swap(m_index_max_[i], other.m_index_max_[i]);
            std::swap(m_shape_min_[i], other.m_shape_min_[i]);
            std::swap(m_shape_max_[i], other.m_shape_max_[i]);
            std::swap(m_strides_[i], other.m_strides_[i]);
        }
    }

    template <int M>
    void reset(std::tuple<nTuple<index_type, M>, nTuple<index_type, M>> const& b) {
        reset(&std::get<0>(b)[0], &std::get<1>(b)[0], M);
    }
    void reset(index_type const* lo = nullptr, index_type const* hi = nullptr, int num_dims = NDIMS) {
        this_type(lo, hi, m_array_order_).swap(*this);
    }
    bool isSlowFirst() const { return m_array_order_ == SLOW_FIRST; };
    size_type GetNDIMS() const { return static_cast<size_type>(ndims); }
    size_type GetIndexBox(index_type* lo, index_type* hi) const {
        for (int i = 0; i < ndims; ++i) {
            if (lo != nullptr) { lo[i] = m_index_min_[i]; }
            if (hi != nullptr) { hi[i] = m_index_max_[i]; }
        }
        return static_cast<size_type>(ndims);
    }
    size_type GetShape(index_type* lo, index_type* hi) const {
        for (int i = 0; i < ndims; ++i) {
            if (lo != nullptr) { lo[i] = m_shape_min_[i]; }
            if (hi != nullptr) { hi[i] = m_shape_max_[i]; }
        }
        return static_cast<size_type>(ndims);
    }
    auto GetIndexBox() const {
        std::tuple<nTuple<index_type, NDIMS>, nTuple<index_type, NDIMS>> res;
        GetIndexBox(&std::get<0>(res)[0], &std::get<1>(res)[0]);
        return res;
    }
    auto GetShape() const {
        std::tuple<nTuple<index_type, NDIMS>, nTuple<index_type, NDIMS>> res;
        GetShape(&std::get<0>(res)[0], &std::get<1>(res)[0]);
        return res;
    }

    size_type size() const {
        index_type res = 1;
        for (int i = 0; i < ndims; ++i) { res *= m_index_max_[i] - m_index_min_[i]; }
        return res > 0 ? static_cast<size_type>(res) : 0;
    }
    size_type shape_size() const {
        index_type res = 1;
        for (int i = 0; i < ndims; ++i) { res *= m_shape_max_[i] - m_shape_min_[i]; }
        return res > 0 ? static_cast<size_type>(res) : 0;
    }
    void Select(index_type const* lo, index_type const* hi) {
        for (int i = 0; i < NDIMS; ++i) {
            m_index_min_[i] = (lo == nullptr) ? m_shape_min_[i] : std::max(lo[i], m_shape_min_[i]);
            m_index_max_[i] = (hi == nullptr) ? m_shape_max_[i] : std::min(hi[i], m_shape_max_[i]);
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

        for (auto i = lo.size(); i < ndims; ++i) { lo.push_back(m_shape_min_[i]); }
        for (auto i = hi.size(); i < ndims; ++i) { lo.push_back(m_shape_max_[i]); }

        Select(&lo[0], &hi[0]);
    }

    void Shift(index_type const* offset) {
        for (int i = 0; i < ndims; ++i) {
            m_index_min_[i] += offset[i];
            m_index_max_[i] += offset[i];
            m_shape_min_[i] += offset[i];
            m_shape_max_[i] += offset[i];
        }
    }

    void Shift(std::initializer_list<index_type> const& idx) {
        std::vector<index_type> s;
        for (auto const& v : idx) { s.push_back(v); }
        for (auto i = s.size(); i < ndims; ++i) { s.push_back(0); }
        Shift(&s[0]);
    }

    void Shift(nTuple<index_type, NDIMS> const& offset) { Shift(&offset[0]); }

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
    __host__ __device__ index_type hash(index_type const* s) const;
    __host__ __device__ bool in_box(index_type const* s) const;

    template <typename... Args>
    __host__ __device__ index_type hash(index_type i0, Args&&... args) const;
    template <typename... Args>
    __host__ __device__ bool in_box(index_type i0, Args&&... args) const;

    __host__ __device__ index_type hash(nTuple<index_type, NDIMS> const& idx) const { return hash(&idx[0]); }
    __host__ __device__ bool in_box(nTuple<index_type, NDIMS> const& idx) const { return in_box(&idx[0]); }

    this_type Overlap(std::nullptr_t) const { return *this; }
    template <typename RHS>
    this_type Overlap(RHS const& rhs) const;
    this_type Overlap(this_type const& rhs) const;

    template <typename TFun>
    size_type Foreach(const TFun& fun) const;
};

namespace detail {
template <size_type... I>
index_type hash_help(std::index_sequence<I...>, index_type const* lo, index_type const* strides,
                     index_type const* idx) {
    return utility::NSum((idx[I] - lo[I]) * strides[I]...);
}
template <size_type... I, typename... Args>
index_type hash_help(std::index_sequence<I...>, index_type const* lo, index_type const* strides, Args&&... args) {
    return utility::NSum((utility::NGet<I>(std::forward<Args>(args)...) - lo[I]) * strides[I]...);
}
template <size_type... I>
bool inbox_help(std::index_sequence<I...>, index_type const* lo, index_type const* hi, index_type const* idx) {
    return utility::NAnd(((idx[I] >= lo[I]) && idx[I] < hi[I])...);
}
template <size_type... I, typename... Args>
bool inbox_help(std::index_sequence<I...>, index_type const* lo, index_type const* hi, Args&&... args) {
    return utility::NAnd(((utility::NGet<I>(std::forward<Args>(args)...) >= lo[I]) &&
                          utility::NGet<I>(std::forward<Args>(args)...) < hi[I])...);
}
}

template <int NDIMS>
__host__ __device__ index_type ZSFC<NDIMS>::hash(index_type const* s) const {
    return detail::hash_help(std::make_index_sequence<NDIMS>(), m_shape_min_, m_strides_, s);
}

template <int NDIMS>
template <typename... Args>
__host__ __device__ index_type ZSFC<NDIMS>::hash(index_type i0, Args&&... args) const {
    return detail::hash_help(std::make_index_sequence<NDIMS>(), m_shape_min_, m_strides_, i0,
                             std::forward<Args>(args)...);
};
template <int NDIMS>
__host__ __device__ bool ZSFC<NDIMS>::in_box(index_type const* s) const {
    return detail::inbox_help(std::make_index_sequence<NDIMS>(), m_shape_min_, m_shape_max_, s);
}

template <int NDIMS>
template <typename... Args>
__host__ __device__ bool ZSFC<NDIMS>::in_box(index_type i0, Args&&... args) const {
    return detail::inbox_help(std::make_index_sequence<NDIMS>(), m_shape_min_, m_shape_max_, i0,
                              std::forward<Args>(args)...);
};

namespace detail {
template <int N, typename T>
std::tuple<nTuple<index_type, N>, nTuple<index_type, N>> overlap(T const&) {
    std::tuple<nTuple<index_type, N>, nTuple<index_type, N>> res;
    std::get<0>(res) = std::numeric_limits<index_type>::min();
    std::get<1>(res) = std::numeric_limits<index_type>::max();
    return res;
}

template <int N, typename U>
std::tuple<nTuple<index_type, N>, nTuple<index_type, N>> overlap(Array<U, ZSFC<N>> const& a) {
    return a.GetSpaceFillingCurve().GetIndexBox();
}
template <int N, int M>
std::tuple<nTuple<index_type, N>, nTuple<index_type, N>> overlap(
    std::tuple<nTuple<index_type, N>, nTuple<index_type, N>> const& a) {
    return a;
}
template <int N>
std::tuple<nTuple<index_type, N>, nTuple<index_type, N>> overlap(
    std::tuple<nTuple<index_type, N>, nTuple<index_type, N>> const& b0,
    std::tuple<nTuple<index_type, N>, nTuple<index_type, N>> const& b1) {
    std::tuple<nTuple<index_type, N>, nTuple<index_type, N>> res;

    for (int i = 0; i < N; ++i) {
        std::get<0>(res)[i] = std::max(std::get<0>(b0)[i], std::get<0>(b1)[i]);
        std::get<1>(res)[i] = std::min(std::get<1>(b0)[i], std::get<1>(b1)[i]);
    }

    return res;
}
template <int N, typename TOP, typename... Args>
std::tuple<nTuple<index_type, N>, nTuple<index_type, N>> overlap(Expression<TOP, Args...> const& expr);

template <int N, typename Arg0, typename Arg1, typename... Others>
auto overlap(Arg0 const& first, Arg1 const& arg1, Others&&... others) {
    return overlap<N>(overlap<N>(first), overlap<N>(arg1, std::forward<Others>(others)...));
}
template <int N, size_t... index, typename... Args>
auto _overlap(std::index_sequence<index...>, std::tuple<Args...> const& expr) {
    return overlap<N>(std::get<index>(expr)...);
}
template <int N, typename TOP, typename... Args>
std::tuple<nTuple<index_type, N>, nTuple<index_type, N>> overlap(Expression<TOP, Args...> const& expr) {
    return _overlap<N>(std::index_sequence_for<Args...>(), expr.m_args_);
}

}  // namespace detail {
template <int NDIMS>
template <typename RHS>
ZSFC<NDIMS> ZSFC<NDIMS>::Overlap(RHS const& rhs) const {
    ZSFC<NDIMS> res(*this);
    res.Select(detail::overlap<NDIMS>(GetShape(), rhs));
    return res;
};
template <int NDIMS>
ZSFC<NDIMS> ZSFC<NDIMS>::Overlap(this_type const& rhs) const {
    ZSFC<NDIMS> res(*this);
    res.Select(detail::overlap<NDIMS>(GetShape(), rhs.GetIndexBox()));
    return res;
}

// template <>
// template <typename LHS, typename RHS>
// size_type ZSFC<3>::Copy(LHS& dst, RHS const& src) const {
//    auto count = size();
//    if (count == 0) { return count; }
//    index_type ib = m_index_min_[0];
//    index_type ie = m_index_max_[0];
//    index_type jb = m_index_min_[1];
//    index_type je = m_index_max_[1];
//    index_type kb = m_index_min_[2];
//    index_type ke = m_index_max_[2];
//    count = static_cast<size_type>((ke - kb) * (je - jb) * (ie - ib));
//
//    if (m_array_order_) {
//#pragma omp parallel for
//        for (index_type k = kb; k < ke; ++k)
//            for (index_type j = jb; j < je; ++j)
//                for (index_type i = ib; i < ie; ++i) { dst(i, j, k) = src(i, j, k); }
//
//    } else {
//#pragma omp parallel for
//        for (index_type i = ib; i < ie; ++i)
//            for (index_type j = jb; j < je; ++j)
//                for (index_type k = kb; k < ke; ++k) { dst(i, j, k) = src(i, j, k); }
//    }
//    return count;
//};

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
    index_type ib = m_index_min_[0];
    index_type ie = m_index_max_[0];
    index_type jb = m_index_min_[1];
    index_type je = m_index_max_[1];
    index_type kb = m_index_min_[2];
    index_type ke = m_index_max_[2];
    count = static_cast<size_type>((ke - kb) * (je - jb) * (ie - ib));
#ifndef __CUDA__
    if (m_array_order_ == SLOW_FIRST) {
#pragma omp parallel for
        for (index_type k = kb; k < ke; ++k)
            for (index_type j = jb; j < je; ++j)
                for (index_type i = ib; i < ie; ++i) { fun(i, j, k); }

    } else {
#pragma omp parallel for
        for (index_type i = ib; i < ie; ++i)
            for (index_type j = jb; j < je; ++j)
                for (index_type k = kb; k < ke; ++k) { fun(i, j, k); }
    }
#else

    dim3 threadsPerBlock{4, 4, 4};

    dim3 numBlocks{static_cast<uint>(m_index_max_[0] - m_index_min_[0] + threadsPerBlock.x) / threadsPerBlock.x,
                   static_cast<uint>(m_index_max_[1] - m_index_min_[1] + threadsPerBlock.y) / threadsPerBlock.y,
                   static_cast<uint>(m_index_max_[2] - m_index_min_[2] + threadsPerBlock.z) / threadsPerBlock.z};

    SP_CALL_DEVICE_KERNEL(foreach_device, numBlocks, threadsPerBlock, m_index_min_, m_index_max_, fun);

#endif
    return count;
}

}  // namespace simpla
#endif  // SIMPLA_Z_SFC_H
