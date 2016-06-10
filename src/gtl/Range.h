/**
 * @file Range.h
 * @author salmon
 * @date 2016-05-18.
 */

#ifndef SIMPLA_GTL_RANGE_H
#define SIMPLA_GTL_RANGE_H

#include <cstddef>

namespace simpla { namespace gtl
{
namespace tags { struct split; }


template<typename ...> struct Range;

/**
 *
 * PlaceHolder on RangeHolder concept in TBB
 *
 */

template<typename Iterator>
class Range<Iterator>
{
    typedef Range<Iterator> this_type;

public:


    typedef Iterator const_iterator;

    typedef Iterator iterator;

    Range() : m_begin_(), m_end_(m_begin_), m_grain_size_(0) { }

    // constructors
    template<typename T0, typename T1>
    Range(T0 const &b, T1 const &e, ptrdiff_t grain_size = 1)
            : m_begin_(b), m_end_(e), m_grain_size_(grain_size)
    {
    }

    template<typename TSplit>
    Range(Range &other, TSplit const &)
            : m_begin_(other.m_begin_), m_end_(m_begin_ + (other.m_end_ - other.m_begin_) / 2),
              m_grain_size_(other.m_grain_size_)
    {
        other.m_begin_ = m_end_;
    }

    // constructors
    Range(const_iterator const &b, const_iterator const &e, ptrdiff_t grain_size = 1)
            : m_begin_(b), m_end_(e), m_grain_size_(grain_size)
    {

    }


    Range(this_type &r, tags::proportional_split &proportion) :
            m_begin_(r.m_begin_ + r.size() * proportion.left() /
                                  ((proportion.left() + proportion.right() > 0) ? (proportion.left() +
                                                                                   proportion.right()) : 1)),
            m_end_(r.m_end_),
            m_grain_size_(r.grainsize())
    {
        r.m_end_ = m_begin_;
    };

    ~Range() { }

    void swap(this_type &other)
    {
        std::swap(m_begin_, other.m_begin_);
        std::swap(m_end_, other.m_end_);
        std::swap(m_grain_size_, other.m_grain_size_);
    }

    // Proportional split is enabled
    static const bool is_splittable_in_proportion = true;

    // capacity
    size_t size() const { return (m_end_ - m_begin_); };

    bool empty() const { return m_begin_ == m_end_; };

    // access
    ptrdiff_t grainsize() const { return m_grain_size_; }

    bool is_divisible() const { return size() > grainsize(); }

    // iterators
    const_iterator const &begin() const { return m_begin_; }

    const_iterator const &end() const { return m_end_; }

private:

    iterator m_begin_, m_end_;

    ptrdiff_t m_grain_size_;

};


}}//namespace simpla { namespace gtl

#endif //SIMPLA_GTL_RANGE_H
