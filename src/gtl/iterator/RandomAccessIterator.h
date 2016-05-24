//
// Created by salmon on 16-5-24.
//

#ifndef SIMPLA_RANDOMACCESSITERATOR_H
#define SIMPLA_RANDOMACCESSITERATOR_H

#include <iterator>

namespace simpla
{

template<typename TValue, typename TDiff>
class RandomAccessIterator : public std::iterator<
        typename std::random_access_iterator_tag,
        TValue, TDiff>
{
    typedef RandomAccessIterator<TValue, TDiff> this_type;
public:
    RandomAccessIterator() { }

    virtual   ~RandomAccessIterator() { }

    value_type  operator*() const { return get(); }

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


    this_type &operator+=(difference_type const &n)
    {
        advance(n);
        return *this;
    }

    this_type &operator-=(difference_type const &n)
    {
        advance(-n);
        return *this;
    }


    this_type operator+(difference_type const &n) const
    {
        this_type res(*this);
        res += n;
        return std::move(res);
    }

    this_type operator-(difference_type const &n) const
    {
        this_type res(*this);
        res -= n;
        return std::move(res);
    }


    difference_type operator-(this_type const &other) const { return distance() - other.distance(); }

    bool operator==(this_type const &other) const { return equal(other); }

    bool operator!=(this_type const &other) const { return !equal(other); }

    bool operator<(this_type const &other) const { return distance() < other.distance(); }

    bool operator>(this_type const &other) const { return distance() > other.distance(); }

    bool operator<=(this_type const &other) const { return distance() <= other.distance(); }

    bool operator>=(this_type const &other) const { return distance() >= other.distance(); }


    virtual void swap(this_type &other) = 0;

    virtual bool equal(this_type const &) const = 0;

    /** return  current value */
    virtual value_type get() const = 0;

    /** advance iterator n steps, return number of actually advanced steps  */
    virtual difference_type advance(difference_type n = 1) = 0;

    /** return the distance between current position to  minimal position */
    virtual difference_type distance() const = 0;
};
}//namespace simpla

#endif //SIMPLA_RANDOMACCESSITERATOR_H
