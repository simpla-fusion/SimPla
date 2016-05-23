/**
 * @file MeshEntitySet.h
 * @author salmon
 * @date 2016-05-18.
 */

#ifndef SIMPLA_MESH_MESHENTITYSET_H
#define SIMPLA_MESH_MESHENTITYSET_H

#include <iterator>
#include "MeshEntity.h"

namespace simpla { namespace tags
{
struct split;
struct proportional_split;
}}//namespace simpla { namespace tags


namespace simpla { namespace gtl
{

class MeshEntityIterator : public std::iterator<
        typename std::random_access_iterator_tag,
        mesh_entity_id_t, mesh_entity_id_diff_t>
{
};

template<typename TIter>
struct Range
{
    typedef TIter iterator;

    typedef TIter const_iterator;

    Range(Range &other, tags::split);

    ~Range();

    // constructors
    Range(const_iterator b, const_iterator e, size_t grain_size = 1)
            : m_begin_(b), m_end_(e), m_grain_size_(grain_size)
    {

    }

    Range(Range &r, tags::split) :
            m_begin_(r.m_begin_ + r.size() / 2), m_end_(r.m_end_), m_grain_size_(r.grainsize())
    {
        r.m_end_ = m_begin_;
    };

    Range(Range &r, tags::proportional_split &proportion) :
            m_begin_(r.m_begin_ + r.size() * proportion.left() / (proportion.left() + proportion.right())),
            m_end_(r.m_end_),
            m_grain_size_(r.grainsize())
    {
        r.m_end_ = m_begin_;
    };

    // Proportional split is enabled
    static const bool is_splittable_in_proportion = true;

    // capacity
    size_t size() const { return traits::distance(m_begin_, m_end_); };

    bool empty() const { return m_begin_ == m_end_; };

    // access
    size_t grainsize() const { return m_grain_size_; }

    bool is_divisible() const { return size() > grainsize(); }

    // iterators
    const_iterator begin() const { return m_begin_; }

    const_iterator end() const { return m_end_; }

    iterator begin() const { return m_begin_; };

    iterator end() const { return m_end_; };

private:
    iterator m_begin_, m_end_;
    size_t m_grain_size_;

};


}}

#endif //SIMPLA_MESHENTITYSET_H
