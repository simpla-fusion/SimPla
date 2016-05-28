/**
 * @file array_view.h
 * @author salmon
 * @date 2016-01-13.
 */

#ifndef SIMPLA_ARRAY_VIEW_H
#define SIMPLA_ARRAY_VIEW_H

 #include "primitives.h"

namespace simpla { namespace tags
{
struct proportional_split;
struct split;
}}

namespace simpla { namespace gtl
{
struct ArrayViewBase
{


    typedef ArrayViewBase this_type;

    int m_ndims_;

    int m_ele_size_in_byte_;

    nTuple<size_t, MAX_NDIMS_OF_ARRAY> m_dims_;
    nTuple<size_t, MAX_NDIMS_OF_ARRAY> m_start_;
    nTuple<size_t, MAX_NDIMS_OF_ARRAY> m_strides_;
    nTuple<size_t, MAX_NDIMS_OF_ARRAY> m_count_;
    nTuple<size_t, MAX_NDIMS_OF_ARRAY> m_block_;
    nTuple<size_t, MAX_NDIMS_OF_ARRAY> m_grain_size_;

    std::shared_ptr<void> m_data_;


    typedef size_t size_type;

    ArrayViewBase(this_type const &other) :
            m_ndims_(other.m_ndims_),
            m_ele_size_in_byte_(other.m_ele_size_in_byte_),
            m_dims_(other.m_dims_),
            m_start_(other.m_start_),
            m_strides_(other.m_strides_),
            m_count_(other.m_count_),
            m_block_(other.m_block_),
            m_grain_size_(other.m_grain_size_),
            m_data_(other.m_data_),
    {


    }

    ~ArrayViewBase() { }


    template<typename T>
    T &get(size_t *const idx)
    {
        return *reinterpret_cast<T *>(reinterpret_cast<char *>(m_data_.get()) + hash_(idx) * m_ele_size_in_byte_);
    }

    template<typename T>
    T const &get(size_t *const idx) const
    {
        return *reinterpret_cast<T const *>(reinterpret_cast<char const *>(m_data_.get()) +
                                            hash_(idx) * m_ele_size_in_byte_);
    }

private:
    size_t hash_(size_t const *idx) const
    {
        size_t res = 0;
        size_t stride = 1;

        for (int i = 0, ie = m_ndims_; i < ie; ++i)
        {
            res += (idx[i] + m_start_[i]) * stride;
            stride *= m_dims_[i];
        }
    }
};

template<typename T>
class ArrayView : public ArrayViewBase
{

public:
    typedef T value_type;

    template<typename ...Args>
    value_type &get(Args &&...args)
    {
        return ArrayViewBase::get<value_type>(std::forward<Args>(args)...);
    }

    template<typename ...Args>
    value_type const &get(Args &&...args) const
    {
        return ArrayViewBase::get<value_type>(std::forward<Args>(args)...);
    }

    ArrayView(int ndims, size_t const *dims)
            : m_ndims_(ndims)
    {
        m_dims_ = (dims);
    }

    ArrayView(this_type &other, tags::split)
    {

    }


    ArrayView(this_type &r, tags::proportional_split &proportion)
    {
        m_begin_(r.m_begin_ + r.size() * proportion.left() / (proportion.left() + proportion.right())),
                m_end_(r.m_end_),
                m_grain_size_(r.grainsize())

    };


    // Proportional split is enabled
    static const bool is_splittable_in_proportion = true;

    // capacity
    size_type size() const;

    bool empty() const { return m_data_ == nullptr; };

    // access
    size_type grainsize() const { return m_grain_size_; }

    bool is_divisible() const { return m_count_ > m_grain_size_; }

    // iterators
    iterator begin() const
    {
        return iterator(m_data_, m_ele_size_in_byte_, m_ndims_, &m_dims_[0], &m_start_[0], &m_strides_[0], &m_count_[0],
                        &m_block_[0]);
    }

    const_iterator end() const { return m_end_; }

    struct iterator;
    struct const iterator;
};

template<typename T>
struct ArrayView<T>::iterator : public std::iterator<typename std::random_access_iterator_tag, value_type, ptrdiff_t>
{
    ArrayViewBase m_data_;

    nTuple<size_t, MAX_NDIMS_OF_ARRAY> m_idx_;


public:
    iterator(ArrayViewBase &d, size_t const *idx = nullptr) : m_data_(d)
    {
        if (idx == nullptr)
        {
            m_idx_ = m_data_.m_start_;
        }

    }

    iterator(iterator const &other) : m_data_(other.m_data_), m_idx_(other.m_idx_) { }

    iterator(iterator &&other) : m_data_(other.m_data_), m_idx_(other.m_idx_) { }


    ~iterator() { }


    iterator &operator=(iterator const &other)
    {
        iterator(other).swap(*this);
        return *this;
    }


    void swap(iterator &other)
    {
        std::swap(m_data_, other.m_data_);
        std::swap(m_idx_, other.m_idx_);
    }


    iterator operator+(difference_type const &s) const
    {
        iterator res(*this);
        res.advance(s);
        return std::move(res);
    }

    value_type const &operator*() const { return m_data_.template get<value_type>(&m_idx_[0]); }

    value_type &operator*() { return m_data_.template get<value_type>(&m_idx_[0]); }


private:
    void advance(ptrdiff_t n)
    {

    }
};


}}
#endif //SIMPLA_ARRAY_VIEW_H
