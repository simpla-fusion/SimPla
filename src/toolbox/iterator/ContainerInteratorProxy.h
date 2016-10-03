//
// Created by salmon on 16-5-24.
//

#ifndef SIMPLA_CONTAINERINTERATORPROXY_H
#define SIMPLA_CONTAINERINTERATORPROXY_H

#include <iterator>
#include "RandomAccessIterator.h"
#include "IteratorAdapter.h"

namespace simpla
{

namespace detail
{

template<typename ...> class ContainerIteratorProxy_;

template<typename TContainer, typename TIterator>
class ContainerIteratorProxy_<TContainer, TIterator, std::random_access_iterator_tag> :
        public RandomAccessIterator<typename TContainer::value_type, typename TIterator::value_type>
{
    typedef ContainerIteratorProxy_<TContainer, TIterator, std::random_access_iterator_tag> this_type;
public:

    ContainerIteratorProxy_(TContainer &container, TIterator it) : m_data_(&container), m_iter_(it) { }

    ~ContainerIteratorProxy_();

    virtual void swap(this_type &other)
    {
        std::swap(m_data_, other.m_data_);
        std::swap(m_iter_, other.m_iter_);
    };

    virtual bool equal(this_type const &other) const
    {
        return m_data_ == other.m_data_ && m_iter_ == other.m_iter_;
    };

    /** return  current value */
    virtual value_type const &get() const { return (*m_data_)[*m_iter_]; };

    virtual value_type &get() { return (*m_data_)[*m_iter_]; };

    /** advance iterator n steps, return number of actually advanced steps  */
    virtual difference_type advance(difference_type n = 1) { return m_iter_.advance(n); };

    /** return the distance between current position to  minimal position */
    virtual difference_type distance() const { return m_iter_.distance(); };

private:
    TContainer *m_data_;
    TIterator m_iter_;
};
}//namespace detail
template<typename TContainer, typename TIterator> using ContainerIteratorProxy=
ContainerIteratorProxy_<TContainer, TIterator, typename TIterator::iterator_category>;
}//namespace simpla

#endif //SIMPLA_CONTAINERINTERATORPROXY_H
