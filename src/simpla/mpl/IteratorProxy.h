//
// Created by salmon on 16-6-2.
//

#ifndef SIMPLA_ITERATORPROXY_H
#define SIMPLA_ITERATORPROXY_H

#include <set>
#include "IteratorAdapter.h"
#include "RangeAdapter.h"

namespace simpla
{
template<typename TIterator>
using IteratorHolder= IteratorAdapter<
        typename std::iterator_traits<TIterator>::iterator_category,
        typename std::iterator_traits<TIterator>::value_type,
        typename std::iterator_traits<TIterator>::difference_type,
        typename std::iterator_traits<TIterator>::pointer,
        typename std::iterator_traits<TIterator>::reference
>;

template<typename TIterator>
class RangeHolder
{
    typedef IteratorHolder<TIterator> iterator;

    iterator m_begin_, m_end_;

    iterator const &begin() const { return m_begin_; }

    iterator const &end() const { return m_end_; }
};

}//namespace simpla{
#endif //SIMPLA_ITERATORPROXY_H
