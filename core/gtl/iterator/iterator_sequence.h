/*
 * sp_iterator_sequence.h
 *
 *  Created on: 2014-11-27
 *      Author: salmon
 */

#ifndef CORE_CONTAINERS_SP_ITERATOR_SEQUENCE_H_
#define CORE_CONTAINERS_SP_ITERATOR_SEQUENCE_H_


#include "range.h"
#include "../type_traits.h"

namespace simpla {
namespace gtl {

/**
 *  @ingroup iterator
 *
 */
template<typename TI>
class IteratorSequence
{
    TI b_;
public:
    typedef IteratorSequence<TI> this_type;

    typedef TI value_type;

    typedef typename std::result_of<
            std::minus<value_type>(value_type, value_type)>::type diff_type;

    IteratorSequence()
    {
    }

    IteratorSequence(value_type const &ib) :
            b_(ib)
    {
    }

    ~IteratorSequence()
    {
    }

    bool operator==(this_type const &other) const
    {
        return b_ == other.b_;
    }

    bool operator!=(this_type const &other) const
    {
        return b_ != other.b_;
    }

    this_type &operator++()
    {
        ++b_;
        return *this;
    }

    this_type operator++(int)
    {
        this_type res(*this);
        ++res;
        return std::move(res);
    }

    this_type &operator--()
    {
        --b_;
        return *this;
    }

    this_type operator--(int)
    {
        this_type res(*this);
        --res;
        return std::move(res);
    }

    value_type operator*() const
    {
        return b_;
    }

    diff_type operator-(this_type const &r) const
    {
        return b_ - r.b_;
    }

    this_type operator+(diff_type const &r) const
    {
        return this_type(b_ + r);
    }

};

template<typename TI>
IteratorSequence<TI> make_seq_iterator(TI const &b)
{
    return IteratorSequence<TI>(b);
}

template<typename TI>
auto make_seq_range(TI const &b, TI const &e)
DECL_RET_TYPE((make_range(make_seq_iterator(b), make_seq_iterator(e))))

}
}//  namespace simpla::gtl

#endif /* CORE_CONTAINERS_SP_ITERATOR_SEQUENCE_H_ */
