//
// Created by salmon on 16-5-24.
//

#ifndef SIMPLA_RANGE_H
#define SIMPLA_RANGE_H

#include <iterator>


namespace simpla
{
namespace tags
{
struct split { };
struct proportional_split { int left, right; };
}

namespace detail
{
template<typename ...> class Range_;

template<typename TIter>
class Range_<TIter, std::random_access_iterator_tag>
{
    typedef Range_<TIter, std::random_access_iterator_tag> this_type;
public:
    typedef TIter const_iterator;
    typedef TIter iterator;

    Range_() : m_grain_size_(1) { }

    template<typename TI>
    Range_(TI const &b, TI const &e, size_t grain_size = 1) :
            m_begin_(b), m_end_(e), m_grain_size_(grain_size) { }

    //****************************************************************************
    // TBB Range Concept Begin

    ~Range_() { }

    Range_(Range_ const &other) : m_begin_(other.m_begin_),
                                  m_end_(other.m_end_),
                                  m_grain_size_(other.m_grain_size_) { }

    Range_(Range_ &other, tags::split)
            : m_begin_(other.m_begin_),
              m_end_(m_begin_ + (other.m_end_ - other.m_begin_) / 2),
              m_grain_size_(other.m_grain_size_)
    {
        other.m_begin_ = m_end_;
    }

    Range_(Range_ &other, tags::proportional_split proportion) :
            m_begin_(other.m_begin_),
            m_end_(m_begin_ +
                   ((other.m_end_ - other.m_begin_) * proportion.left)
                   / (proportion.left + proportion.right)),
            m_grain_size_(other.m_grain_size_)
    {
        other.m_begin_ = m_end_;
    }

    static const bool is_splittable_in_proportion = true;

    bool is_divisible() const { return m_end_ - m_begin_ > m_grain_size_; }

    bool empty() const { return m_end_ == m_begin_; }

    void swap(this_type &other)
    {
        std::swap(m_begin_, other.m_begin_);
        std::swap(m_end_, other.m_end_);
        std::swap(m_grain_size_, other.m_grain_size_);
    }

    // TBB Range Concept End
    //****************************************************************************
    iterator begin() const { return m_begin_; }

    iterator end() const { return m_end_; }

    const_iterator const_begin() const { return m_begin_; }

    const_iterator const_end() const { return m_end_; }

private:
    iterator m_begin_, m_end_;
    size_t m_grain_size_;
};
}

template<typename TIterator> using Range=detail::Range_<TIterator, typename TIterator::iterator_category>;


}//namespace simpla { namespace mesh

#endif //SIMPLA_RANGE_H
