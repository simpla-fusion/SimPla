/**
 * @file block_iterator.h
 * @author salmon
 * @date 2015-10-26.
 */

#ifndef SIMPLA_BLOCK_ITERATOR_H
#define SIMPLA_BLOCK_ITERATOR_H

#include <cstdlib>
#include <cmath>
#include <iterator>
#include "simpla/algebra/nTuple.h"
#include "type_traits_ext.h"

namespace simpla { namespace toolbox
{

template<typename TV, int NDIMS>
struct IteratorBlock : public std::iterator<
        typename std::random_access_iterator_tag,
        nTuple<TV, NDIMS>, ptrdiff_t>
{
private:
    typedef std::iterator<typename std::random_access_iterator_tag, nTuple<TV, NDIMS>, ptrdiff_t>
            base_type;

    typedef IteratorBlock<TV, NDIMS> this_type;

    nTuple<TV, NDIMS> m_min_, m_max_, m_self_;


public:

    using typename base_type::value_type;
    using typename base_type::difference_type;

    IteratorBlock() : m_min_(), m_max_(m_min_), m_self_(m_min_) { }

    IteratorBlock(nTuple<TV, NDIMS> const &self, nTuple<TV, NDIMS> const &min, nTuple<TV, NDIMS> const &max) :
            m_min_(min), m_max_(max), m_self_(self)
    {
    }

    IteratorBlock(nTuple<TV, NDIMS> const &min, nTuple<TV, NDIMS> const &max) :
            m_min_(min), m_max_(max), m_self_(min)
    {
    }

    IteratorBlock(this_type const &other) :
            m_min_(other.m_min_), m_max_(other.m_max_), m_self_(other.m_self_)
    {
    }

    IteratorBlock(this_type &&other) :
            m_min_(other.m_min_), m_max_(other.m_max_), m_self_(other.m_self_)
    {
    }

    ~IteratorBlock() { }

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    void swap(this_type &other)
    {
        std::swap(m_max_, other.m_max_);
        std::swap(m_min_, other.m_min_);
        std::swap(m_self_, other.m_self_);
    }

    this_type end() const
    {
        this_type res(*this);

        res.m_self_ = m_max_ - 1;

        res.advance();

        return std::move(res);

    }

    bool operator==(this_type const &other) const { return m_self_ == other.m_self_; }

    bool operator!=(this_type const &other) const { return m_self_ != other.m_self_; }

    value_type const &operator*() const { return m_self_; }

    this_type &operator++()
    {
        advance(1);
        return *this;
    }

    this_type &operator--()
    {
        advance(-1);
        return *this;
    }

    this_type operator++(int) const
    {
        this_type res(*this);
        ++(*this);
        return std::move(res);
    }

    this_type operator--(int) const
    {
        this_type res(*this);
        --(*this);
        return std::move(res);
    }

    this_type &operator+=(ptrdiff_t const &n)
    {
        advance(n);
        return *this;
    }

    this_type &operator-=(ptrdiff_t const &n)
    {
        advance(-n);
        return *this;
    }


    this_type operator+(ptrdiff_t const &n) const
    {

        this_type res(*this);
        res += n;
        return std::move(res);
    }

    this_type operator-(ptrdiff_t const &n) const
    {

        this_type res(*this);
        res -= n;
        return std::move(res);
    }


    nTuple<TV, NDIMS> operator[](int const &n) const
    {
        this_type res(*this);
        res += n;
        return *res;
    }

    ptrdiff_t operator-(this_type const &other) const
    {
        return distance() - other.distance();
    }


    bool operator<(this_type const &other) const { return (*this - other) < 0; }

    bool operator>(this_type const &other) const { return (*this - other) > 0; }

    bool operator<=(this_type const &other) const { return (*this - other) <= 0; }

    bool operator>=(this_type const &other) const { return (*this - other) >= 0; }


    ptrdiff_t advance(ptrdiff_t n = 1)
    {
        for (int i = NDIMS - 1; i > 0; --i)
        {
            auto L = m_max_[i] - m_min_[i];
            if (L == 0)L = 1;
            auto d = div(m_self_[i] - m_min_[i] + n + m_max_[i] - m_min_[i], L);

            m_self_[i] = d.rem + m_min_[i];

            n = d.quot - 1;
        }
        m_self_[0] += n;
        return n;
    }

    ptrdiff_t distance() const
    {

        ptrdiff_t res = 0;


        for (int i = 0; i < NDIMS - 1; ++i)
        {
            res += m_self_[i] - m_min_[i];

            res *= m_max_[i + 1] - m_min_[i + 1];
        }

        res += m_self_[NDIMS - 1] - m_min_[NDIMS - 1];

        return res;
    }

};


}}// namespace simpla{namespace toolbox{

#endif //SIMPLA_BLOCK_ITERATOR_H
