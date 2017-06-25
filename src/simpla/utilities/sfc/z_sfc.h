//
// Created by salmon on 17-6-24.
//

#ifndef SIMPLA_Z_SFC_H
#define SIMPLA_Z_SFC_H

#include <simpla/SIMPLA_config.h>
#include <tuple>
#include "../cuda/cuda.h"
#include "../nTuple.h"
namespace simpla {
template <typename V, int NDIMS, typename SFC>
struct Array;
template <int NDIMS>
class ZSFC {
    typedef ZSFC<NDIMS> this_type;

    bool m_array_order_fast_first_ = false;

   public:
    typedef nTuple<index_type, NDIMS> array_index_type;
    typedef std::tuple<array_index_type, array_index_type> array_index_box_type;
    array_index_box_type m_index_box_{{0, 0, 0}, {1, 1, 1}};
    array_index_type m_strides_{0, 0, 0};
    index_type m_offset_ = 0;
    size_type m_size_ = 0;
    bool m_is_fast_first_ = false;
    ZSFC() = default;
    ~ZSFC() = default;

    ZSFC(this_type const& other)
        : m_index_box_(other.m_index_box_), m_array_order_fast_first_(other.m_array_order_fast_first_) {
        DoSetUp();
    }
    ZSFC(this_type&& other)
    noexcept : m_index_box_(other.m_index_box_), m_array_order_fast_first_(other.m_array_order_fast_first_) {
        DoSetUp();
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
        m_index_box_.swap(m_index_box_);
        m_strides_.swap(other.m_strides_);
        std::swap(m_offset_, other.m_offset_);
        std::swap(m_size_, other.m_size_);
        std::swap(m_is_fast_first_, other.m_is_fast_first_);
    }

    ZSFC(std::initializer_list<index_type> const& l) {
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
        DoSetUp();
    }

    explicit ZSFC(array_index_box_type const& b, bool array_order_fast_first = false)
        : m_index_box_(b), m_array_order_fast_first_(array_order_fast_first) {
        DoSetUp();
    }

    void DoSetUp() {
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
        m_size_ = 1;
        for (int i = 0; i < NDIMS; ++i) { m_size_ *= (std::get<1>(m_index_box_)[i] - std::get<0>(m_index_box_)[i]); }
    };

    size_t size() const { return m_size_; }

    void Shift(array_index_type const& offset) {
        std::get<0>(m_index_box_) += offset;
        std::get<1>(m_index_box_) += offset;
        DoSetUp();
    }

    __host__ __device__ constexpr size_t hash(array_index_type const& idx) const {
        return dot(idx - m_offset_, m_strides_);
    }

    template <typename... Args>
    __host__ __device__ constexpr size_t hash(index_type s0, Args&&... idx) const {
        return hash(array_index_type{s0, std::forward<Args>(idx)...});
    }

    template <typename TFun>
    void Foreach(TFun const& f) const;

    template <typename T, typename TRhs>
    void Assign(T& lhs, TRhs const& rhs) const;

    template <typename value_type>
    std::ostream& Print(std::ostream& os, value_type const* v, int indent = 0) const;
};
template <int NDIMS>
template <typename value_type>
std::ostream& ZSFC<NDIMS>::Print(std::ostream& os, value_type const* v, int indent) const {
    os << "Array [" << typeid(value_type).name() << "]: " << m_index_box_ << std::endl;

    Foreach([&](array_index_type const& idx) {
        if (idx[NDIMS - 1] == std::get<0>(m_index_box_)[NDIMS - 1]) {
            os << "{" << v[hash(idx)];
        } else {
            os << "," << v[hash(idx)];
        }
        if (idx[NDIMS - 1] == std::get<1>(m_index_box_)[NDIMS - 1] - 1) { os << "}" << std::endl; }
    });

    return os;
}

template <int N>
template <typename TFun>
void ZSFC<N>::Foreach(TFun const& fun) const {
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
template <>
template <typename TFun>
void ZSFC<1>::Foreach(TFun const& fun) const {
    index_type ib = std::get<0>(m_index_box_)[0];
    index_type ie = std::get<1>(m_index_box_)[0];
#pragma omp parallel for
    for (index_type i = ib; i < ie; ++i) { fun(nTuple<index_type, 1>{i}); }
}
template <>
template <typename TFun>
void ZSFC<2>::Foreach(TFun const& fun) const {
    index_type ib = std::get<0>(m_index_box_)[0];
    index_type ie = std::get<1>(m_index_box_)[0];
    index_type jb = std::get<0>(m_index_box_)[1];
    index_type je = std::get<1>(m_index_box_)[1];
    if (m_is_fast_first_) {
#pragma omp parallel for
        for (index_type j = jb; j < je; ++j)
            for (index_type i = ib; i < ie; ++i) { fun(nTuple<index_type, 2>{i, j}); }
    } else {
#pragma omp parallel for
        for (index_type i = ib; i < ie; ++i)
            for (index_type j = jb; j < je; ++j) { fun(nTuple<index_type, 2>{i, j}); }
    }
}
template <>
template <typename TFun>
void ZSFC<3>::Foreach(TFun const& fun) const {
    index_type ib = std::get<0>(m_index_box_)[0];
    index_type ie = std::get<1>(m_index_box_)[0];
    index_type jb = std::get<0>(m_index_box_)[1];
    index_type je = std::get<1>(m_index_box_)[1];
    index_type kb = std::get<0>(m_index_box_)[2];
    index_type ke = std::get<1>(m_index_box_)[2];

    if (m_is_fast_first_) {
#pragma omp parallel for
        for (index_type k = kb; k < ke; ++k)
            for (index_type j = jb; j < je; ++j)
                for (index_type i = ib; i < ie; ++i) { fun(nTuple<index_type, 3>{i, j, k}); }

    } else {
#pragma omp parallel for
        for (index_type i = ib; i < ie; ++i)
            for (index_type j = jb; j < je; ++j)
                for (index_type k = kb; k < ke; ++k) { fun(nTuple<index_type, 3>{i, j, k}); }
    }
}
#ifdef __CUDA__
template <typename T, typename TRhs>
__global__ void assign(T lhs, TRhs rhs) {
    nTuple<index_type, 3> idx{blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y,
                              blockIdx.z * blockDim.z + threadIdx.z};
    lhs.at(idx) = T::getValue(rhs, idx);
};

#endif

template <>
template <typename T, typename TRhs>
void ZSFC<3>::Assign(T& lhs, TRhs const& rhs) const {
    index_type ib = std::get<0>(m_index_box_)[0];
    index_type ie = std::get<1>(m_index_box_)[0];
    index_type jb = std::get<0>(m_index_box_)[1];
    index_type je = std::get<1>(m_index_box_)[1];
    index_type kb = std::get<0>(m_index_box_)[2];
    index_type ke = std::get<1>(m_index_box_)[2];

#ifdef __CUDA__
    SP_CALL_DEVICE_KERNEL(assign, 2, 32, lhs, rhs);
#else
    if (m_is_fast_first_) {
#pragma omp parallel for
        for (index_type k = kb; k < ke; ++k)
            for (index_type j = jb; j < je; ++j)
                for (index_type i = ib; i < ie; ++i) {
                    nTuple<index_type, 3> idx{i, j, k};
                    lhs.at(idx) = T::getValue(rhs, idx);
                }

    } else {
#pragma omp parallel for
        for (index_type i = ib; i < ie; ++i)
            for (index_type j = jb; j < je; ++j)
                for (index_type k = kb; k < ke; ++k) {
                    nTuple<index_type, 3> idx{i, j, k};
                    lhs.at(idx) = T::getValue(rhs, idx);
                }
    }
#endif
}
template <>
template <typename TFun>
void ZSFC<4>::Foreach(TFun const& fun) const {
    index_type ib = std::get<0>(m_index_box_)[0];
    index_type ie = std::get<1>(m_index_box_)[0];
    index_type jb = std::get<0>(m_index_box_)[1];
    index_type je = std::get<1>(m_index_box_)[1];
    index_type kb = std::get<0>(m_index_box_)[2];
    index_type ke = std::get<1>(m_index_box_)[2];
    index_type lb = std::get<0>(m_index_box_)[3];
    index_type le = std::get<1>(m_index_box_)[3];

    if (m_is_fast_first_) {
#pragma omp parallel for
        for (index_type l = lb; l < le; ++l)
            for (index_type k = kb; k < ke; ++k)
                for (index_type j = jb; j < je; ++j)
                    for (index_type i = ib; i < ie; ++i) fun(nTuple<index_type, 4>{i, j, k, l});

    } else {
#pragma omp parallel for
        for (index_type i = ib; i < ie; ++i)
            for (index_type j = jb; j < je; ++j)
                for (index_type k = kb; k < ke; ++k)
                    for (index_type l = lb; l < le; ++l) { fun(nTuple<index_type, 4>{i, j, k, l}); }
    }
}
template <typename TIdx>
size_type Hash(std::tuple<nTuple<index_type, 2>, nTuple<index_type, 2>> const& box, TIdx const& idx) {
    return static_cast<size_type>(((idx[1] - std::get<0>(box)[1]) +
                                   (idx[0] - std::get<0>(box)[0]) * (std::get<1>(box)[1] - std::get<0>(box)[1])));
}
template <typename TIdx>
size_type Hash(std::tuple<nTuple<index_type, 3>, nTuple<index_type, 3>> const& box, TIdx const& idx) {
    return static_cast<size_type>((idx[2] - std::get<0>(box)[2]) +
                                  ((idx[1] - std::get<0>(box)[1]) +
                                   (idx[0] - std::get<0>(box)[0]) * (std::get<1>(box)[1] - std::get<0>(box)[1])) *
                                      (std::get<1>(box)[2] - std::get<0>(box)[2]));
}
template <int N, typename TIdx>
size_type Hash(std::tuple<nTuple<index_type, N>, nTuple<index_type, N>> const& box, TIdx const& idx) {
    size_type res = idx[0] - std::get<0>(box)[0];
    for (int i = 1; i < N; ++i) {
        res *= (std::get<1>(box)[i - 1] - std::get<0>(box)[i - 1]);
        res += idx[i] - std::get<0>(box)[i];
    }
    return res;
}

template <typename T, int N>
bool in_box(std::tuple<nTuple<T, N>, nTuple<T, N>> const& b, nTuple<T, N> const& idx) {
    bool res = true;
    for (int i = 0; i < N; ++i) { res = res && (std::get<0>(b)[i] <= idx[i]) && (idx[i] < std::get<1>(b)[i]); }
    return res;
};
}  // namespace simpla
#endif  // SIMPLA_Z_SFC_H
