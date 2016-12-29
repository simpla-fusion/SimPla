//
// Created by salmon on 16-5-24.
//

#ifndef SIMPLA_RANGE_H
#define SIMPLA_RANGE_H

#include <iterator>
#include <simpla/concept/Splittable.h>
#include "iterator_adapter.h"

namespace simpla
{

template<typename _Category, typename _Tp,
        typename _Distance = ptrdiff_t, typename _Pointer = _Tp *, typename _Reference = _Tp &> class range;

/**
 * @brief Range concept
 *   @ref https://github.com/ericniebler/range-v3
 *   @ref TBB Range concept https://software.intel.com/en-us/node/506143
 *
 *   - lightweight
 *   - non-owning
 *   - range may/may not be iteraterable (begin,end may not be defined)
 *   - range may/may not be splittable
 *   - range may be iterated in an arbitrary order, or in parallel
 *   example:
 *     double d[10]
 *     range<double> r(d,d+10);

 *     r.apply([&](double & v){ r+=1.0;});
 *
 *     range<double> r1(r,tags::split());
 *
 *     auto r2=r1.split();
 *
 *     range<const double> cr(d,d+10);
 *     r.apply([&](double const & v){ r+=1.0;});
 *
 * @tparam _Category
 * @tparam _Tp  value_type
 * @tparam _Distance difference_type
 * @tparam _Pointer  value_type *
 * @tparam _Reference value_type &
 */
template<typename _Tp, typename _Category, typename _Distance, typename _Pointer, typename _Reference>
class range
{
    typedef range<_Tp, _Category, _Distance, _Pointer, _Reference> this_type;
public:
    typedef iterator_adapter<_Tp, _Category, _Distance, _Pointer, _Reference> iterator;
    typedef iterator const_iterator;

    typedef _Tp value_type;

    range() : m_grain_size_(1) {}

    template<typename TI>
    range(TI const &b, TI const &e, size_t grain_size = 1) :
            m_begin_(iterator(b)), m_end_(iterator(e)), m_grain_size_(grain_size) {}

    //****************************************************************************
    // TBB RangeHolder Concept Begin

    ~range() {}

    range(range const &other) :
            m_begin_(other.m_begin_),
            m_end_(other.m_end_),
            m_grain_size_(other.m_grain_size_) {}

    range(range &other, concept::tags::split,
          ENABLE_IF((std::is_same<std::random_access_iterator_tag, _Category>::value))) :
            m_begin_(other.m_begin_),
            m_end_(m_begin_ + (other.m_end_ - other.m_begin_) / 2),
            m_grain_size_(other.m_grain_size_)
    {
        other.m_begin_ = m_end_;
    }

    range(range &other, concept::tags::proportional_split proportion,
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

    template<typename TFun> void apply(TFun const &fun) const
    {
        static_assert(traits::is_callable<TFun(value_type &)>::value,
                      "Function is not  applicable! ");
        for (auto it = m_begin_; it != m_end_; ++it) { fun(*it); }
    }


private:
    iterator m_begin_, m_end_;
    size_t m_grain_size_;
};


}//namespace simpla { namespace get_mesh

#endif //SIMPLA_RANGE_H
