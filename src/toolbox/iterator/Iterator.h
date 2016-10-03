/**
 * @file iterator.h
 *
 * @date 2014-7-4
 * @author salmon
 */

#ifndef SP_ITERATOR_H_
#define SP_ITERATOR_H_

#include "../type_traits.h"
#include <iterator>

namespace simpla
{


template<typename _Category, typename _Tp, typename _Distance = ptrdiff_t, typename _Pointer = _Tp *, typename _Reference = _Tp &>
class Iterator : public std::iterator<_Category, _Tp, _Distance, _Pointer, _Reference>
{
    typedef Iterator<std::random_access_iterator_tag, _Tp, _Distance, _Pointer, _Reference> this_type;
    typedef std::iterator<std::random_access_iterator_tag, _Tp, _Distance, _Pointer, _Reference> category_type;

    struct HolderBase;

    template<typename TOther> struct Holder;

    std::shared_ptr<HolderBase> m_holder_;

public:
    using typename category_type::reference;
    using typename category_type::value_type;
    using typename category_type::difference_type;


    Iterator() : m_holder_(nullptr) { }

    template<typename TOther>
    Iterator(TOther const &it) : m_holder_(
            std::dynamic_pointer_cast<HolderBase>(std::make_shared<Holder<TOther> >(it))) { }

    Iterator(this_type const &other) : m_holder_(other.m_holder_->clone()) { }

    Iterator(this_type &&other) : m_holder_(other.m_holder_) { }

    virtual  ~Iterator() { }

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    void swap(this_type &other) { std::swap(m_holder_, other.m_holder_); }


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
        return (m_holder_ == nullptr && other.m_holder_ == nullptr) || m_holder_->equal(*other.m_holder_);
    }

    /** return  current value */
    const reference get() const { return m_holder_->get(); }

    reference get() { return m_holder_->get(); }

    /** advance iterator n steps, return number of actually advanced steps  */
    void advance(difference_type n = 1) { m_holder_->advance(n); }

    difference_type distance(this_type const &other) const
    {
        return m_holder_->distance(*other.m_holder_);
    };

private:

    struct HolderBase
    {

        virtual std::shared_ptr<HolderBase> clone() const = 0;

        virtual bool is_a(std::type_info const &) const = 0;

        template<typename T> bool is_a() const { return is_a(typeid(T)); }

        virtual bool equal(HolderBase const &) const = 0;

        /** return  current value */
        virtual const reference get() const = 0;

        virtual reference get() = 0;

        /** advance iterator n steps, return number of actually advanced steps  */
        virtual void advance(difference_type n = 1) = 0;

        virtual difference_type distance(HolderBase const &other) const = 0;
    };

    template<typename Tit>
    struct Holder : public HolderBase
    {
        typedef Holder<Tit> this_type;
        Tit m_iter_;
    public:

        Holder(Tit const &o_it) : m_iter_(o_it) { }

        virtual  ~Holder() final { }

        virtual std::shared_ptr<HolderBase> clone() const
        {
            return std::dynamic_pointer_cast<HolderBase>(std::make_shared<this_type>(m_iter_));
        }

        virtual bool is_a(std::type_info const &t_info) const final { return typeid(Tit) == t_info; }

        virtual bool equal(HolderBase const &other) const final
        {
            assert(other.template is_a<Tit>());
            return static_cast<Holder<Tit> const & >(other).m_iter_ == m_iter_;
        };

        virtual difference_type distance(HolderBase const &other) const final
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


} // namespace simpla



#endif /* SP_ITERATOR_H_ */
