//
// Created by salmon on 16-5-24.
//

#ifndef SIMPLA_RANGE_H
#define SIMPLA_RANGE_H

#include <iterator>
#include "iterator_adapter.h"

namespace simpla
{
namespace tags
{
struct split { };

struct proportional_split
{
    int m_left_ = 1, m_right_ = 1;

    int left() const { return m_left_; }

    int right() const { return m_right_; }
};
}
template<typename _Category, typename _Tp,
        typename _Distance = ptrdiff_t, typename _Pointer = _Tp *, typename _Reference = _Tp &> class range;


template<typename _Category, typename _Tp, typename _Distance, typename _Pointer, typename _Reference>
class range
{
    typedef range<_Category, _Tp, _Distance, _Pointer, _Reference> this_type;
public:
    typedef iterator_adapter<_Category, _Tp, _Distance, _Pointer, _Reference> iterator;
    typedef iterator const_iterator;

    range() : m_grain_size_(1) { }

    template<typename TI>
    range(TI const &b, TI const &e, size_t grain_size = 1) :
            m_begin_(iterator(b)), m_end_(iterator(e)), m_grain_size_(grain_size) { }

    //****************************************************************************
    // TBB RangeHolder Concept Begin

    ~range() { }

    range(range const &other) :
            m_begin_(other.m_begin_),
            m_end_(other.m_end_),
            m_grain_size_(other.m_grain_size_) { }

    range(range &other, tags::split,
          ENABLE_IF((std::is_same<std::random_access_iterator_tag, _Category>::value))) :
            m_begin_(other.m_begin_),
            m_end_(m_begin_ + (other.m_end_ - other.m_begin_) / 2),
            m_grain_size_(other.m_grain_size_)
    {
        other.m_begin_ = m_end_;
    }

    range(range &other, tags::proportional_split proportion,
          ENABLE_IF((std::is_same<std::random_access_iterator_tag, _Category>::value))) :
            m_begin_(other.m_begin_),
            m_end_(m_begin_ + ((other.m_end_ - other.m_begin_) * proportion.left())
                              / (proportion.left() + proportion.right())),
            m_grain_size_(other.m_grain_size_)
    {
        other.m_begin_ = m_end_;
    }

    static const bool is_splittable = std::is_same<std::random_access_iterator_tag, _Category>::value;
    static const bool is_splittable_in_proportion = std::is_same<std::random_access_iterator_tag, _Category>::value;

    bool is_divisible(ENABLE_IF((std::is_same<std::random_access_iterator_tag, _Category>::value))) const
    {
        return m_end_ - m_begin_ > m_grain_size_;
    }

    bool is_divisible(ENABLE_IF((!std::is_same<std::random_access_iterator_tag, _Category>::value))) const
    {
        return false;
    }

    bool empty() const { return m_end_ == m_begin_; }

    void swap(this_type &other)
    {
        std::swap(m_begin_, other.m_begin_);
        std::swap(m_end_, other.m_end_);
        std::swap(m_grain_size_, other.m_grain_size_);
    }

    // TBB RangeHolder Concept End
    //****************************************************************************
    iterator begin() const { return m_begin_; }

    iterator end() const { return m_end_; }

    const_iterator const_begin() const { return m_begin_; }

    const_iterator const_end() const { return m_end_; }

private:
    iterator m_begin_, m_end_;
    size_t m_grain_size_;
};


}//namespace simpla { namespace get_mesh

#endif //SIMPLA_RANGE_H
