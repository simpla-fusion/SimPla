//
// Created by salmon on 16-10-31.
//

#ifndef SIMPLA_ARRAYPATCH_H
#define SIMPLA_ARRAYPATCH_H

#include <type_traits>
#include <simpla/toolbox/PrettyStream.h>
#include <simpla/toolbox/Memory.h>
#include "DataEntity.h"
#include "DataBase.h"

namespace simpla { namespace data
{
enum
{
    SLOW_FIRST = 0, //C_OLDER
    FAST_FIRST = 1//FORTRAN_ORDER
};

template<typename V>
class DataEntityNDArray : public DataEntityHeavy
{
public:
    typedef DataEntityNDArray<V> this_type;
    typedef V value_type;

    DataEntityNDArray() : m_data_(nullptr), m_holder_(nullptr), m_order_(SLOW_FIRST), m_ndims_(0), m_size_(0) {}


    DataEntityNDArray(value_type *p, int ndims, index_type const *lo, index_type const *hi,
                      int order = SLOW_FIRST)
            : m_data_(p), m_holder_(nullptr), m_order_(order), m_size_(0) { initialize(ndims, lo, hi, order); }

    DataEntityNDArray(std::shared_ptr<value_type> const &p, int ndims, index_type const *lo, index_type const *hi,
                      int order = SLOW_FIRST)
            : m_holder_(p), m_data_(p.get()), m_order_(order), m_size_(0) { initialize(ndims, lo, hi, order); };


    DataEntityNDArray(this_type const &other) = delete;

    virtual ~DataEntityNDArray() {};

private:
    void initialize(int ndims, index_type const *lo, index_type const *hi, int order = SLOW_FIRST)
    {
        m_order_ = order;
        m_ndims_ = ndims;
        m_size_ = 1;
        for (int i = 0; i < m_ndims_; ++i)
        {
            m_start_[i] = lo[i];
            if (hi[i] > lo[i])
            {
                m_count_[i] = static_cast<size_type>(hi[i] - lo[i]);
            } else
            {
                m_count_[i] = 1;
            }
            m_size_ *= m_count_[i];
        }
        for (int j = m_ndims_; j < MAX_NDIMS_OF_ARRAY; ++j)
        {
            m_count_[j] = 1;
            m_start_[j] = 0;
            m_strides_[j] = 0;
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
//        CHECK("strides") << m_strides_[0] << "," << m_strides_[1] << "," << m_strides_[2] << "," << m_strides_[3]   << std::endl;
    }

public:

    virtual bool is_valid() const { return !empty(); };

    virtual bool empty() const { return m_data_ == nullptr; }

    virtual std::string get_class_name() const { return class_name(); }

    static std::string class_name()
    {
        return std::string("DataEntityNDArray<") + traits::type_id<value_type>::name() + std::string(",4>");
    }

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

    virtual void load(DataBase const &, std::string const & = "") {};

    virtual void save(DataBase *, std::string const & = "") const {};

    virtual bool is_a(std::type_info const &t_info) const
    {
        return t_info == typeid(this_type) || DataEntityHeavy::is_a(t_info);
    };

    virtual bool is_null() const { return false; };

    virtual bool is_deployed() const { return m_data_ != nullptr; }

    virtual void deploy()
    {
        if (m_data_ == nullptr)
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

    virtual void clear()
    {
        deploy();

        toolbox::MemorySet(m_data_, 0, m_size_ * sizeof(value_type));
    }


    virtual void *data() { return m_data_; }

    virtual void const *data() const { return m_data_; }

    template<typename ...Args>
    value_type &get(Args &&...args) { return m_data_[hash(std::forward<Args>(args)...)]; }

    template<typename ...Args>
    value_type const &get(Args &&...args) const { return m_data_[hash(std::forward<Args>(args)...)]; }

private:

    int m_ndims_;

    index_type m_start_[MAX_NDIMS_OF_ARRAY];
    size_type m_count_[MAX_NDIMS_OF_ARRAY];
    size_type m_strides_[MAX_NDIMS_OF_ARRAY];
    size_type m_size_;

    std::shared_ptr<value_type> m_holder_;
    value_type *m_data_;
    int m_order_ = SLOW_FIRST;

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
//        return (x0 - m_start_[0] + m_count_[0] * 2) % m_count_[0] * m_strides_[0]
//               + (x1 - m_start_[1] + m_count_[1] * 2) % m_count_[1] * m_strides_[1]
//               + (x2 - m_start_[2] + m_count_[2] * 2) % m_count_[2] * m_strides_[2];

        return (x0 - m_start_[0]) * m_strides_[0] +
               (x1 - m_start_[1]) * m_strides_[1] +
               (x2 - m_start_[2]) * m_strides_[2];
    }

    inline constexpr size_type hash(index_type x0, index_type x1, index_type x2, index_type x3) const
    {
        return (x0 - m_start_[0]) * m_strides_[0] +
               (x1 - m_start_[1]) * m_strides_[1] +
               (x2 - m_start_[2]) * m_strides_[2] +
               (x3 - m_start_[3]) * m_strides_[3];
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

    inline constexpr size_type hash(nTuple<index_type, 4> const idx) const
    {
        return hash(idx[0], idx[1], idx[2], idx[3]);
    }

    inline constexpr size_type hash(nTuple<index_type, 3> const idx) const { return hash(idx[0], idx[1], idx[2]); }

public:
    template<typename TOP, typename TFUN>
    void for_each(TOP const &op, TFUN const &fun)
    {
        ASSERT(m_ndims_ <= 4);
//#pragma omp parallel for
        for (int i = 1; i < m_count_[0] - 1; ++i)
            for (int j = 1; j < m_count_[1] - 1; ++j)
                for (int k = 1; k < m_count_[2] - 1; ++k)
                    for (int l = 0; l < m_count_[3]; ++l)
                    {
                        op(get(i, j, k, l), fun(i, j, k, l));
                    }


    };
};
}}//namespace simpla { namespace data
#endif //SIMPLA_ARRAYPATCH_H
