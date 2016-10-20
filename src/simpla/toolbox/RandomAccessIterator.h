//
// Created by salmon on 16-5-24.
//

#ifndef SIMPLA_RANDOMACCESSITERATOR_H
#define SIMPLA_RANDOMACCESSITERATOR_H

#include <iterator>

namespace simpla
{

template<typename _Tp, typename _Distance = ptrdiff_t,
        typename _Pointer = _Tp *, typename _Reference = _Tp &>
class RandomAccessIterator
        : public std::iterator<typename std::random_access_iterator_tag, _Tp, _Distance, _Pointer, _Reference>
{
    typedef std::iterator<typename std::random_access_iterator_tag, _Tp, _Distance, _Pointer, _Reference> category_type;
    typedef RandomAccessIterator<_Tp, _Distance, _Pointer, _Reference> this_type;


public:
    using typename category_type::reference;
    using typename category_type::value_type;
    using typename category_type::difference_type;

    RandomAccessIterator() { }

    virtual   ~RandomAccessIterator() { }

    const reference   operator*() const { return get(); }

    reference  operator*() { return get(); }

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


//    this_type  operator++(int)
//    {
//        this_type res(*this);
//        ++(*this);
//        return std::move(res);
//    }
//
//    this_type operator--(int)
//    {
//        this_type res(*this);
//        --(*this);
//        return std::move(res);
//    }


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


//    this_type operator+(difference_type const &n) const
//    {
//        this_type res(*this);
//        res += n;
//        return std::move(res);
//    }
//
//    this_type operator-(difference_type const &n) const
//    {
//        this_type res(*this);
//        res -= n;
//        return std::move(res);
//    }


    difference_type operator-(this_type const &other) const { return std::distance(*this, other); }

    bool operator==(this_type const &other) const { return equal(other); }

    bool operator!=(this_type const &other) const { return !equal(other); }

//    bool operator<(this_type const &other) const { return distance() < other.distance(); }
//
//    bool operator>(this_type const &other) const { return distance() > other.distance(); }
//
//    bool operator<=(this_type const &other) const { return distance() <= other.distance(); }
//
//    bool operator>=(this_type const &other) const { return distance() >= other.distance(); }


//    virtual void swap(this_type &other) = 0;

    virtual bool equal(this_type const &) const = 0;

    /** return  current value */
    virtual const reference get() const = 0;

    virtual reference get() = 0;

    /** advance iterator n steps, return number of actually advanced steps  */
    virtual difference_type advance(difference_type n = 1) = 0;

//    /** return the distance between current position to  minimal position */
//    virtual difference_type distance() const = 0;
};
}//namespace simpla

#endif //SIMPLA_RANDOMACCESSITERATOR_H
