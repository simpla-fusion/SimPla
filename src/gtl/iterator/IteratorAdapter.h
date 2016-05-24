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
template<typename TIter>
class IteratorAdapter : public TIter
{
    typedef IteratorAdapter<TIter> this_type;
    typedef TIter base_type;

    std::shared_ptr<base_type> m_iter_;
public:

    template<typename TOther>
    IteratorAdapter(TOther const &it)
    {
        if (std::is_base_of<base_type, TOther>::value)
        {
            m_iter_ = std::dynamic_pointer_cast<base_type>(std::make_shared<T>(it));
        }
        else
        {
            m_iter_ = std::shared_ptr<base_type>(new Holder<TOther>(it));
        }
    }

    IteratorAdapter(this_type const &other) : m_iter_(other.m_iter_) { }

    IteratorAdapter(this_type &&other) : m_iter_(other.m_iter_) { }

    virtual  ~IteratorAdapter() { }

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    virtual void swap(this_type &other) { std::swap(m_iter_, other.m_iter_); };

    virtual bool equal(this_type const &other) { *m_iter_ == *other.m_iter_; };

    /** return  current value */
    virtual value_type get() const { return m_iter_->get(); };

    /** advance iterator n steps, return number of actually advanced steps  */
    virtual difference_type advance(difference_type n = 1) { return m_iter_->advance(n); }

    /** return the distance between current position to  minimal position */
    virtual difference_type distance() const { return m_iter_->distance(); };

private:
    template<typename TOther>
    struct Holder : public base_type
    {
        typedef Holder<TOther> this_type;
        TOther m_iter_;
    public:
        Holder(TOther const &o_it) : m_iter_(o_it) { }

        virtual ~Holder() { }

        virtual void swap(this_type &other) { m_iter_.swap(other.m_iter_); }

        virtual bool equal(base_type const &) const { return static_cast<T const & >(other) == m_iter_; };

        /** return  current value */
        virtual value_type get() const { return *m_iter_; };

        /** advance iterator n steps, return number of actually advanced steps  */
        virtual difference_type advance(difference_type n = 1)
        {
            return m_iter_.advance(n);
        }

        /** return the distance between current position to  minimal position */
        virtual difference_type distance() const
        {
            return m_iter_.distance();
        };


    };

};
}}//namespace simpla { namespace mesh

#endif //SIMPLA_ITERATORADAPTER_H
