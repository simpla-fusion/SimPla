//
// Created by salmon on 16-5-24.
//

#ifndef SIMPLA_ITERATORADAPTER_H
#define SIMPLA_ITERATORADAPTER_H

#include <memory>

namespace simpla { namespace mesh
{

/**
 * adapt other iterator to TIter iterator;
 */
template<typename _Category, typename _Tp, typename _Distance = ptrdiff_t, typename _Pointer = _Tp *, typename _Reference = _Tp &>
class IteratorAdapter;

template<typename _Tp, typename _Distance, typename _Pointer, typename _Reference>
class IteratorAdapter<std::random_access_iterator_tag, _Tp, _Distance, _Pointer, _Reference>
        : public std::iterator<typename std::random_access_iterator_tag, _Tp, _Distance, _Pointer, _Reference>
{
    typedef IteratorAdapter<std::random_access_iterator_tag, _Tp, _Distance, _Pointer, _Reference> this_type;
    typedef std::iterator<std::random_access_iterator_tag, _Tp, _Distance, _Pointer, _Reference> category_type;

    struct _Base;

    template<typename TOther> struct Holder;

    std::shared_ptr<_Base> m_iter_;

public:
    using typename category_type::reference;
    using typename category_type::value_type;
    using typename category_type::difference_type;


    IteratorAdapter() : m_iter_(nullptr) { }

    template<typename TOther>
    IteratorAdapter(TOther const &it) : m_iter_(
            std::dynamic_pointer_cast<_Base>(std::make_shared<Holder<TOther> >(it))) { }

    IteratorAdapter(this_type const &other) : m_iter_(other.m_iter_->clone()) { }

    IteratorAdapter(this_type &&other) : m_iter_(other.m_iter_) { }

    virtual  ~IteratorAdapter() { }

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    void swap(this_type &other) { std::swap(m_iter_, other.m_iter_); }


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


    this_type  operator++(int)
    {
        this_type res(*this);
        ++(*this);
        return std::move(res);
    }

    this_type operator--(int)
    {
        this_type res(*this);
        --(*this);
        return std::move(res);
    }

    reference operator[](difference_type n) const
    {
        this_type res(*this);
        res += n;
        return *res;
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


    difference_type operator-(this_type const &other) const { return distance(other); }

    bool operator==(this_type const &other) const { return equal(other); }

    bool operator!=(this_type const &other) const { return !equal(other); }

    bool operator<(this_type const &other) const { return distance(other) < 0; }

    bool operator>(this_type const &other) const { return distance(other) > 0; }

    bool operator<=(this_type const &other) const { return distance(other) <= 0; }

    bool operator>=(this_type const &other) const { return distance(other) >= 0; }


    bool equal(this_type const &other) const
    {
        return (m_iter_ == nullptr && other.m_iter_ == nullptr) || m_iter_->equal(*other.m_iter_);
    }

    /** return  current value */
    const reference get() const { return m_iter_->get(); }

    reference get() { return m_iter_->get(); }

    /** advance iterator n steps, return number of actually advanced steps  */
    void advance(difference_type n = 1) { m_iter_->advance(n); }

    difference_type distance(this_type const &other) const { m_iter_->distance(*other.m_iter_); };

private:

    struct _Base
    {

        virtual std::shared_ptr<_Base> clone() const = 0;

        virtual bool is_a(std::type_info const &) const = 0;

        template<typename T> bool is_a() const { return is_a(typeid(T)); }

        virtual bool equal(_Base const &) const = 0;

        /** return  current value */
        virtual const reference get() const = 0;

        virtual reference get() = 0;

        /** advance iterator n steps, return number of actually advanced steps  */
        virtual void advance(difference_type n = 1) = 0;

        virtual difference_type distance(_Base const &other) const = 0;
    };

    template<typename Tit>
    struct Holder : public _Base
    {
        typedef Holder<Tit> this_type;
        Tit m_iter_;
    public:

        Holder(Tit const &o_it) : m_iter_(o_it) { }

        virtual  ~Holder() final { }

        virtual std::shared_ptr<_Base> clone() const
        {
            return std::dynamic_pointer_cast<_Base>(std::make_shared<this_type>(m_iter_));
        }

        virtual bool is_a(std::type_info const &t_info) const final { return typeid(Tit) == t_info; }

        virtual bool equal(_Base const &other) const final
        {
            assert(other.template is_a<Tit>());
            return static_cast<Holder<Tit> const & >(other).m_iter_ == m_iter_;
        };

        virtual difference_type distance(_Base const &other) const final
        {
            assert(other.template is_a<Tit>());
            return std::distance(static_cast<Holder<Tit> const & >(other).m_iter_, m_iter_);
        }


        /** return  current value */
        virtual const reference get() const final { return *m_iter_; };

        virtual reference get() final { return *m_iter_; };

        /** advance iterator n steps, return number of actually advanced steps  */
        virtual void advance(difference_type n = 1) final { return std::advance(m_iter_, n); }


    };

};


}}//namespace simpla { namespace mesh

#endif //SIMPLA_ITERATORADAPTER_H
