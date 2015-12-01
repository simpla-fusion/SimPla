/**
 * @file block_range.h
 * @author salmon
 * @date 2015-10-26.
 */

#ifndef SIMPLA_BLOCK_RANGE_H
#define SIMPLA_BLOCK_RANGE_H

#include <stddef.h>
#include <algorithm>
#include <cstdbool>
#include <functional>
#include <set>
#include <tuple>
#include <type_traits>

#include "../type_traits.h"

namespace simpla
{

//namespace tags
template<typename ...> struct BlockRange;


// base on Range concept in TBB

template<typename Iterator>
class BlockRange<Iterator>
{
public:

    // types

    typedef size_t size_type;

    typedef Iterator const_iterator;

    // constructors
    BlockRange(const_iterator b, const_iterator e, size_type grain_size = 1)
            : m_begin_(b), m_end_(e), m_grain_size_(grain_size)
    {

    }

    BlockRange(BlockRange &r, tags::split) :
            m_begin_(r.m_begin_ + r.size() / 2), m_end_(r.m_end_), m_grain_size_(r.grainsize())
    {
        r.m_end_ = m_begin_;
    };

    BlockRange(BlockRange &r, tags::proportional_split &proportion) :
            m_begin_(r.m_begin_ + r.size() * proportion.left() / (proportion.left() + proportion.right())),
            m_end_(r.m_end_),
            m_grain_size_(r.grainsize())
    {
        r.m_end_ = m_begin_;
    };

    // Proportional split is enabled
    static const bool is_splittable_in_proportion = true;

    // capacity
    size_type size() const { return traits::distance(m_begin_, m_end_); };

    bool empty() const { return m_begin_ == m_end_; };

    // access
    size_type grainsize() const { return m_grain_size_; }

    bool is_divisible() const { return size() > grainsize(); }

    // iterators
    const_iterator begin() const { return m_begin_; }

    const_iterator end() const { return m_end_; }

private:

    const_iterator m_begin_, m_end_;

    size_type m_grain_size_;




//        struct iterator : public std::iterator<
//                typename std::bidirectional_iterator_tag, id_type,
//                difference_type>
//        {
//        private:
//            id_type m_idx_min_, m_idx_max_, m_self_;
//        public:
//            iterator(id_type const &min, id_type const &max,
//                     id_type const &self) :
//                    m_idx_min_(min), m_idx_max_(max), m_self_(self)
//            {
//            }
//
//            iterator(id_type const &min, id_type const &max) :
//                    m_idx_min_(min), m_idx_max_(max), m_self_(min)
//            {
//            }
//
//            iterator(iterator const &other) :
//                    m_idx_min_(other.m_idx_min_), m_idx_max_(other.m_idx_max_), m_self_(
//                    other.m_self_)
//            {
//            }
//
//            ~iterator()
//            {
//
//            }
//
//            typedef iterator this_type;
//
//            bool operator==(this_type const &other) const
//            {
//                return m_self_ == other.m_self_;
//            }
//
//            bool operator!=(this_type const &other) const
//            {
//                return m_self_ != other.m_self_;
//            }
//
//            value_type const &operator*() const
//            {
//                return m_self_;
//            }
//
//        private:
//
//            index_type carray_(index_type *self, index_type min, index_type max,
//                               index_type flag = 0)
//            {
//
//                auto div = std::div(
//                        static_cast<long>(*self + flag * (_D << 1) + max
//                                          - min * 2), static_cast<long>(max - min));
//
//                *self = static_cast<id_type>(div.rem + min);
//
//                return div.quot - 1L;
//            }
//
//            index_type carray(id_type *self, id_type xmin, id_type xmax,
//                              index_type flag = 0)
//            {
//                index_tuple idx, min, max;
//
//                idx = unpack(*self);
//                min = unpack(xmin);
//                max = unpack(xmax);
//
//                flag = carray_(&idx[0], min[0], max[0], flag);
//                flag = carray_(&idx[1], min[1], max[1], flag);
//                flag = carray_(&idx[2], min[2], max[2], flag);
//
//                *self = pack(idx) | (std::abs(flag) << (FULL_DIGITS - 1));
//                return flag;
//            }
//
//        public:
//            void next()
//            {
//                m_self_ = rotate(m_self_);
//                if (sub_index(m_self_) == 0)
//                {
//                    carray(&m_self_, m_idx_min_, m_idx_max_, 1);
//                }
//
//            }
//
//            void prev()
//            {
//                m_self_ = inverse_rotate(m_self_);
//                if (sub_index(m_self_) == 0)
//                {
//                    carray(&m_self_, m_idx_min_, m_idx_max_, -1);
//                }
//            }
//
//            this_type &operator++()
//            {
//                next();
//                return *this;
//            }
//
//            this_type &operator--()
//            {
//                prev();
//
//                return *this;
//            }
//
//            this_type operator++(int)
//            {
//                this_type res(*this);
//                ++(*this);
//                return std::move(res);
//            }
//
//            this_type operator--(int)
//            {
//                this_type res(*this);
//                --(*this);
//                return std::move(res);
//            }
//
//        };
//
//    };

};

}//namespace simpla

#endif //SIMPLA_BLOCK_RANGE_H
