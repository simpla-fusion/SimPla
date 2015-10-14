/*
 * sp_iterator_filter.h
 *
 *  created on: 2014-6-5
 *      Author: salmon
 */

#ifndef SP_ITERATOR_FILTER_H_
#define SP_ITERATOR_FILTER_H_

namespace simpla {
namespace gtl {

/**
 *  @ingroup iterator
 *
 *  @note  boost::filter_iterator
 *
 */
template<typename BaseIterator, typename TPred>
struct sp_fliter_iterator : public BaseIterator
{
    typedef BaseIterator base_iterator;

    typedef TPred predicate_fun;

    typedef sp_fliter_iterator<base_iterator, predicate_fun> this_type;

    base_iterator m_last_;
    predicate_fun m_pred_;

    sp_fliter_iterator()
    {
    }

    sp_fliter_iterator(this_type const &other)
            : base_iterator(other), m_last_(other.m_last_), m_pred_(
            other.m_pred_)
    {
    }

    sp_fliter_iterator(base_iterator first, base_iterator last,
                       predicate_fun const &p)
            : base_iterator(first), m_last_(last), m_pred_(p)
    {
        filter();
    }

    ~sp_fliter_iterator()
    {
    }

    using base_iterator::operator==;
    using base_iterator::operator!=;
    using base_iterator::operator*;
    using base_iterator::operator->;

    this_type &operator++()
    {
        base_iterator::operator++();
        filter();
        return *this;
    }

    this_type &operator--()
    {
        base_iterator::operator--();
        filter_reverse();
        return *this;
    }

    this_type operator++(int) const
    {
        this_type res(*this);
        ++res;
        return std::move(res);
    }

    this_type operator--(int) const
    {
        this_type res(*this);
        --res;
        return std::move(res);
    }

private:
    void filter()
    {
        while (!m_pred_(base_iterator::operator*())
               && base_iterator::operator!=(m_last_))
        {
            base_iterator::operator++();
        }
    }

    void filter_reverse()
    {
        while (!m_pred_(base_iterator::operator*())
               && base_iterator::operator!=(m_last_))
        {
            base_iterator::operator--();
        }
    }

};

template<typename BaseIterator, typename TPred, typename ...Args>
sp_fliter_iterator<BaseIterator, TPred> make_filter_iterator(BaseIterator first,
                                                             BaseIterator last, TPred pred)
{
    return sp_fliter_iterator<BaseIterator, TPred>(first, last, pred);
}

}
}//  namespace simpla::gtl

#endif /* SP_ITERATOR_FILTER_H_ */
