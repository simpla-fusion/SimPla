//
// Created by salmon on 16-10-31.
//

#ifndef SIMPLA_ARRAYPATCH_H
#define SIMPLA_ARRAYPATCH_H

#include <type_traits>
#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/FancyStream.h>
#include <simpla/toolbox/Memory.h>
#include "HeavyData.h"

namespace simpla { namespace data
{
enum
{
    SLOW_FIRST = 0, //C_OLDER
    FAST_FIRST = 1//FORTRAN_ORDER
};
/** @ingroup data */
/**
 * @brief
 * @tparam V
 */
template<typename V>
class DataEntityNDArray : public HeavyData
{
SP_OBJECT_HEAD(DataEntityNDArray<V>, HeavyData);

public:

    typedef V value_type;

    explicit DataEntityNDArray()
            : m_data_(nullptr), m_holder_(nullptr), m_order_(SLOW_FIRST), m_ndims_(0), m_size_(0) {}


    template<typename ...Args>
    explicit DataEntityNDArray(value_type *p, Args &&...args)
            : m_data_(p), m_holder_(nullptr), m_order_(SLOW_FIRST), m_size_(0)
    {
        initialize(std::forward<Args>(args)...);
    }

    template<typename ...Args>
    explicit DataEntityNDArray(std::shared_ptr<value_type> const &p, Args &&...args)
            : m_holder_(p), m_data_(p.get()), m_order_(SLOW_FIRST), m_size_(0)
    {
        initialize(std::forward<Args>(args)...);
//        initialize(ndims, lo, hi, order, i_lo, i_hi);
    };


    DataEntityNDArray(this_type const &other) = delete;

    virtual ~DataEntityNDArray() {};

private:
    void initialize(int ndims, index_type const *lo, index_type const *hi, int order = SLOW_FIRST,
                    index_type const *i_lower = nullptr, index_type const *i_upper = nullptr)
    {
        m_order_ = order;
        m_ndims_ = ndims;
        m_size_ = 1;
        for (int i = 0; i < MAX_NDIMS_OF_ARRAY; ++i)
        {
            m_count_[i] = 1;
            m_start_[i] = 0;
            m_strides_[i] = 0;
            m_lower_[i] = 0;
            m_upper_[i] = 1;
        }

        for (int i = 0; i < m_ndims_; ++i)
        {
            m_start_[i] = lo[i];
            if (hi[i] > lo[i]) { m_count_[i] = static_cast<size_type>(hi[i] - lo[i]); }
            m_size_ *= m_count_[i];

            m_lower_[i] = (i_lower != nullptr && i_lower[i] >= lo[i]) ? i_lower[i] : lo[i];
            m_upper_[i] = (i_upper != nullptr && i_upper[i] <= hi[i]) ? i_upper[i] : hi[i];
        }

        if (m_order_ == SLOW_FIRST)
        {
            m_strides_[m_ndims_ - 1] = 1;
            for (int j = m_ndims_ - 2; j >= 0; --j) { m_strides_[j] = m_count_[j + 1] * m_strides_[j + 1]; }
        } else if (m_order_ == FAST_FIRST)
        {
            m_strides_[0] = 1;
            for (int j = 1; j < m_ndims_; ++j) { m_strides_[j] = m_count_[j - 1] * m_strides_[j - 1]; }
        }

    }

public:
    int ndims() const { return m_ndims_; }

    index_type const *start() const { return m_start_; }

    index_type const *index_lower() const { return m_lower_; }

    index_type const *index_upper() const { return m_upper_; }

    size_type const *count() const { return m_count_; }

    size_type size() const { return m_size_; }

    virtual bool is_valid() const
    {
        return (m_data_ != nullptr) &&
               (m_count_[0] >= 1) &&
               (m_count_[1] >= 1) &&
               (m_count_[2] >= 1) &&
               (m_count_[3] >= 1) &&
               (m_count_[4] >= 1);
    };


    virtual std::ostream &print(std::ostream &os, int indent = 1) const
    {
        os << "-- dims=["
           << m_count_[0] << ","
           << m_count_[1] << ","
           << m_count_[2] << ","
           << m_count_[3] << "," << "]"
           << std::endl;

        size_type r_count[m_ndims_];

        int r_ndims = 0;
        for (int i = 0; i < m_ndims_; ++i)
        {
            if (m_count_[i] > 1)
            {
                r_count[r_ndims] = m_count_[i];
                ++r_ndims;
            }
        }

        printNdArray(os, m_data_, r_ndims, r_count);

        return os;

    }

    virtual void load(DataTable const &, std::string const & = "") { UNIMPLEMENTED; };

    virtual void save(DataTable *, std::string const & = "") const { UNIMPLEMENTED; };

    virtual bool empty() const { return m_data_ == nullptr; }

    virtual bool is_null() const { return m_data_ == nullptr; };

    virtual void deploy()
    {
        if (empty())
        {
            if (m_holder_ == nullptr && m_size_ > 0) { m_holder_ = toolbox::MemoryHostAllocT<value_type>(m_size_); }
            m_data_ = m_holder_.get();
        }

    };

    virtual void destroy()
    {
        m_holder_.reset();
        m_data_ = nullptr;
    }

    virtual void update()
    {
        if (empty()) { deploy(); }
        if (m_data_ == nullptr && m_holder_ != nullptr) { m_data_ = m_holder_.get(); }
    }


    virtual void clear()
    {
        update();
        toolbox::MemorySet(m_data_, 0, m_size_ * sizeof(value_type));
    }

    virtual void deep_copy(this_type const &other)
    {
        update();
        toolbox::MemoryCopy(m_data_, other.m_data_, m_size_ * sizeof(value_type));
    }

    virtual void *data() { return m_data_; }

    virtual void const *data() const { return m_data_; }

    template<typename ...Args>
    value_type &get(Args &&...args) { return m_data_[hash(std::forward<Args>(args)...) % m_size_]; }

    template<typename ...Args>
    value_type const &get(Args &&...args) const { return m_data_[hash(std::forward<Args>(args)...) % m_size_]; }

//private:

    int m_ndims_;

    index_type m_start_[MAX_NDIMS_OF_ARRAY];
    size_type m_count_[MAX_NDIMS_OF_ARRAY];
    size_type m_strides_[MAX_NDIMS_OF_ARRAY];

    index_type m_lower_[MAX_NDIMS_OF_ARRAY];
    index_type m_upper_[MAX_NDIMS_OF_ARRAY];

    size_type m_size_ = 0;

    std::shared_ptr<value_type> m_holder_;
    value_type *m_data_ = nullptr;
    int m_order_ = SLOW_FIRST;

    inline constexpr size_type hash(mesh::MeshEntityId s) const
    {
        UNIMPLEMENTED;
        return 0;
//        return (x0 - m_start_[0] + m_count_[0] * 2) % m_count_[0] * m_strides_[0];
    }

    inline constexpr size_type hash(index_type x0) const
    {
        return (x0 - m_start_[0] + m_count_[0] * 2) % m_count_[0] * m_strides_[0];
    }

    inline constexpr size_type hash(index_type x0, index_type x1) const
    {
        return (x0 - m_start_[0] + m_count_[0] * 2) % m_count_[0] * m_strides_[0]
               + (x1 - m_start_[1] + m_count_[1] * 2) % m_count_[1] * m_strides_[1];
    }

    inline constexpr size_type hash(index_type x0, index_type x1, index_type x2) const
    {
        return (x0 - m_start_[0] + m_count_[0] * 2) % m_count_[0] * m_strides_[0]
               + (x1 - m_start_[1] + m_count_[1] * 2) % m_count_[1] * m_strides_[1]
               + (x2 - m_start_[2] + m_count_[2] * 2) % m_count_[2] * m_strides_[2];

//        return (x0 - m_start_[0]) * m_strides_[0] +
//               (x1 - m_start_[1]) * m_strides_[1] +
//               (x2 - m_start_[2]) * m_strides_[2];
    }

    inline constexpr size_type hash(index_type x0, index_type x1, index_type x2, index_type x3) const
    {


        ASSERT((x0 - m_start_[0] + m_count_[0] * 2) % m_count_[0] * m_strides_[0] +
               (x1 - m_start_[1] + m_count_[1] * 2) % m_count_[1] * m_strides_[1] +
               (x2 - m_start_[2] + m_count_[2] * 2) % m_count_[2] * m_strides_[2] +
               (x3 - m_start_[3] + m_count_[3] * 2) % m_count_[3] * m_strides_[3] < m_size_);

        return (x0 - m_start_[0] + m_count_[0] * 2) % m_count_[0] * m_strides_[0] +
               (x1 - m_start_[1] + m_count_[1] * 2) % m_count_[1] * m_strides_[1] +
               (x2 - m_start_[2] + m_count_[2] * 2) % m_count_[2] * m_strides_[2] +
               (x3 - m_start_[3] + m_count_[3] * 2) % m_count_[3] * m_strides_[3];

//        return (x0 - m_start_[0]) * m_strides_[0] +
//               (x1 - m_start_[1]) * m_strides_[1] +
//               (x2 - m_start_[2]) * m_strides_[2] +
//               (x3 - m_start_[3]) * m_strides_[3];
    }

    inline constexpr size_type hash(nTuple<index_type, 4> const &idx) const
    {
        return hash(idx[0], idx[1], idx[2], idx[3]);
    }

    inline constexpr size_type
    hash2(index_type const *start, size_type const *count) const { return 0; }

    template<typename ...Args>
    inline constexpr size_type
    hash2(index_type const *start, size_type const *count, size_type const *stride, index_type first,
          Args const &...others) const
    {
        return (first - start[0] + count[0] * 2) % count[0] * stride[0] +
               hash2(start + 1, count + 1, stride + 1, std::forward<Args>(others)...);
    }

    template<typename ...Args>
    inline constexpr size_type
    hash(index_type x0, index_type x1, index_type x2, index_type x3, index_type s4, Args const &...args) const
    {
        return hash(x0, x1, x2, x3) +
               hash2(m_start_ + 4, m_count_ + 4, m_strides_ + 4, s4, std::forward<Args>(args)...);
    }


public:


    template<typename TFUN>
    void foreach(TFUN const &fun, index_type const *gw = nullptr)
    {


        ASSERT(m_ndims_ <= 4);

        index_type ib = m_start_[0] + (gw != nullptr ? gw[0] : 0);
        index_type ie = m_start_[0] + m_count_[0] - (gw != nullptr ? gw[0] : 0);
        index_type jb = m_start_[1] + (gw != nullptr ? gw[1] : 0);
        index_type je = m_start_[1] + m_count_[1] - (gw != nullptr ? gw[1] : 0);
        index_type kb = m_start_[2] + (gw != nullptr ? gw[2] : 0);
        index_type ke = m_start_[2] + m_count_[2] - (gw != nullptr ? gw[2] : 0);
        index_type lb = m_start_[3] + (gw != nullptr ? gw[3] : 0);
        index_type le = m_start_[3] + m_count_[3] - (gw != nullptr ? gw[3] : 0);

        if (m_order_ == SLOW_FIRST)
        {
#pragma omp parallel for
            for (index_type i = ib; i < ie; ++i)
                for (index_type j = jb; j < je; ++j)
                    for (index_type k = kb; k < ke; ++k)
                        for (index_type l = lb; l < le; ++l)
                        {
                            fun(get(i, j, k, l), i, j, k, l);
                        }

        } else
        {
            for (index_type l = lb; l < le; ++l)
            {
#pragma omp parallel for
                for (index_type i = ib; i < ie; ++i)
                    for (index_type j = jb; j < je; ++j)
                        for (index_type k = kb; k < ke; ++k)
                        {
                            fun(get(i, j, k, l), i, j, k, l);
                        }
            }
        }
    };


    template<typename TFUN>
    void assign_ghost(TFUN const &fun)
    {
        ASSERT(m_ndims_ <= 4);

        index_type ib = m_lower_[0], ie = m_upper_[0];
        index_type jb = m_lower_[1], je = m_upper_[1];
        index_type kb = m_lower_[2], ke = m_upper_[2];
        index_type lb = m_lower_[3], le = m_upper_[3];
///******/
//        VERBOSE << "start ={" << m_start_[0] << "," << m_start_[1] << "," << m_start_[2] << "},"
//                << "count ={" << m_count_[0] << "," << m_count_[1] << "," << m_count_[2] << "},"
//                << "lower ={" << m_lower_[0] << "," << m_lower_[1] << "," << m_lower_[2] << "},"
//                << "upper ={" << m_upper_[0] << "," << m_upper_[1] << "," << m_upper_[2] << "}," << std::endl;
#pragma omp parallel for
        for (index_type i = m_start_[0]; i < m_lower_[0]; ++i)
            for (index_type j = m_start_[1]; j < m_start_[1] + m_count_[1]; ++j)
                for (index_type k = m_start_[2]; k < m_start_[2] + m_count_[2]; ++k)
                    for (index_type l = m_start_[3]; l < m_start_[3] + m_count_[3]; ++l)
                    {
                        get(i, j, k, l) = fun(i, j, k, l);
                    }

#pragma omp parallel for
        for (index_type i = m_upper_[0]; i < m_start_[0] + m_start_[0]; ++i)
            for (index_type j = m_start_[1]; j < m_start_[1] + m_count_[1]; ++j)
                for (index_type k = m_start_[2]; k < m_start_[2] + m_count_[2]; ++k)
                    for (index_type l = m_start_[3]; l < m_start_[3] + m_count_[3]; ++l)
                    {
                        get(i, j, k, l) = fun(i, j, k, l);
                    }
/******/

#pragma omp parallel for
        for (index_type j = m_lower_[1]; j < m_start_[1]; ++j)
            for (index_type i = m_lower_[0]; i < m_upper_[0]; ++i)
                for (index_type k = m_start_[2]; k < m_start_[2] + m_count_[2]; ++k)
                    for (index_type l = m_start_[3]; l < m_start_[3] + m_count_[3]; ++l)
                    {
                        get(i, j, k, l) = fun(i, j, k, l);
                    }

#pragma omp parallel for
        for (index_type j = m_upper_[1]; j < m_start_[1] + m_count_[1]; ++j)
            for (index_type i = m_lower_[0]; i < m_upper_[0]; ++i)
                for (index_type k = m_start_[2]; k < m_start_[2] + m_count_[2]; ++k)
                    for (index_type l = m_start_[3]; l < m_start_[3] + m_count_[3]; ++l)
                    {
                        get(i, j, k, l) = fun(i, j, k, l);
                    }
/******/

#pragma omp parallel for
        for (index_type k = m_lower_[2]; k < m_start_[2]; ++k)
            for (index_type i = m_lower_[0]; i < m_upper_[0]; ++i)
                for (index_type j = m_lower_[1]; j < m_upper_[1]; ++j)
                    for (index_type l = m_start_[3]; l < m_start_[3] + m_count_[3]; ++l)
                    {
                        get(i, j, k, l) = fun(i, j, k, l);
                    }

#pragma omp parallel for
        for (index_type k = m_upper_[2]; k < m_start_[2] + m_count_[2]; ++k)
            for (index_type i = m_lower_[0]; i < m_upper_[0]; ++i)
                for (index_type j = m_lower_[1]; j < m_upper_[1]; ++j)
                    for (index_type l = m_start_[3]; l < m_start_[3] + m_count_[3]; ++l)
                    {
                        get(i, j, k, l) = fun(i, j, k, l);
                    }
//        print(std::cout);

    };
};
}}//namespace simpla { namespace data_block
#endif //SIMPLA_ARRAYPATCH_H
