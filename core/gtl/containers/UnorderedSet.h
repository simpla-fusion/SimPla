/**
 * @file unordered_set.h
 * @author salmon
 * @date 2016-01-14.
 */

#ifndef SIMPLA_UNORDERED_SET_H
#define SIMPLA_UNORDERED_SET_H

#include <list>
#include <algorithm>
#include "../../parallel/Parallel.h"

namespace simpla { namespace gtl
{
template<typename ValueType, typename Key=size_t>
class UnorderedSet : public parallel::concurrent_hash_map<Key, std::list<ValueType>>
{
    typedef UnorderedSet<ValueType, Key> this_type;

    typedef std::list<ValueType> bucket_type;

    typedef typename parallel::concurrent_hash_map<Key, bucket_type> base_type;

    typedef Key key_type;

    typedef ValueType value_type;


public:

    typedef base_type buffer_type;


    UnorderedSet();

    virtual ~UnorderedSet();


    UnorderedSet(this_type const &) = delete;

    UnorderedSet(this_type &&) = delete;

    this_type &operator=(this_type const &other) = delete;

    void swap(this_type const &other) = delete;

    using base_type::range;


    //! @name as container
    //! @{

    /**
     * @require TConstraint = map<key_type,TARGS>
     *          Fun         = function<void(TARGS const &,sample_type*)>
     */

    template<typename TFun> void filter(TFun const &fun, key_type const &);

    template<typename TFun> void filter(TFun const &fun, key_type const &) const;

    template<typename TFun> void filter(TFun const &fun, typename base_type::value_type const &);

    template<typename TFun> void filter(TFun const &fun, typename base_type::value_type const &) const;

    template<typename TRange, typename TFun> void filter(TFun const &fun, TRange const &);

    template<typename TRange, typename TFun> void filter(TFun const &fun, TRange const &) const;

    template<typename TRange, typename TFun>
    void filter(std::tuple<TFun, TRange> const &f) { filter(std::get<0>(f), std::get<1>(f)); }

    template<typename TRange, typename TFun>
    void filter(std::tuple<TFun, TRange> const &f) const { filter(std::get<0>(f), std::get<1>(f)); }

    template<typename TFun> void filter(TFun const &fun);

    template<typename TFun> void filter(TFun const &fun) const;


    template<typename Predicate> void remove_if(Predicate const &pred, key_type const &r);

    template<typename Predicate> void remove_if(Predicate const &pred, typename base_type::value_type const &r);

    template<typename TRange, typename Predicate> void remove_if(Predicate const &pred, key_type const &r);

    template<typename Predicate> void remove_if(Predicate const &pred);

    void insert(value_type const &p, key_type const &s = 0);

    template<typename Hash>
    void insert(value_type const &p, Hash const &hash);

    template<typename InputIterator>
    void insert(InputIterator const &b, InputIterator const &e, key_type const &s);

    template<typename InputIterator, typename THash>
    void insert(InputIterator const &b, InputIterator const &e, THash const &);


    using base_type::erase;

    void erase(typename base_type::range_type const &r);


    template<typename TRange> void erase(TRange const &r);


    void erase_all() { base_type::clear(); }


    size_t size(key_type const &s) const;

    size_t size(typename base_type::value_type const &v) const;

    template<typename TRange> size_t size(TRange const &r) const;

    size_t size() const;


    template<typename OutputIT> OutputIT copy(OutputIT out_it, key_type const &s) const;

    template<typename OutputIT, typename TRange> OutputIT copy(OutputIT out_it, TRange const &r) const;

    template<typename OutputIT> OutputIT copy(OutputIT out_it) const;


    void merge(key_type const &s, base_type *other);

    template<typename TRange> void merge(TRange const &r, base_type *other);

    void merge(base_type *other);


    template<typename Hash> void rehash(key_type const &key, Hash const &hash, buffer_type *out_buffer);

    template<typename TRange, typename Hash> void rehash(TRange const &r, Hash const &hash, buffer_type *out_buffer);

    template<typename Hash> void rehash(Hash const &hash, buffer_type *out_buffer = nullptr);

    //! @}




};//class UnorderedSet

template<typename P, typename M>
UnorderedSet<P, M>::UnorderedSet()
{

}


template<typename P, typename M>
UnorderedSet<P, M>::~UnorderedSet() { }


//**************************************************************************************************
template<typename P, typename M> void
UnorderedSet<P, M>::insert(value_type const &v, key_type const &s)
{
    typename base_type::accessor acc;

    base_type::insert(acc, s);

    acc->second.push_back(v);
}

template<typename P, typename M>
template<typename InputIterator>
void UnorderedSet<P, M>::insert(InputIterator const &b, InputIterator const &e, key_type const &s)
{
    for (auto it = b; it != e; ++it) { insert(*it, s); }
}


template<typename P, typename M>
template<typename InputIterator, typename THash>
void UnorderedSet<P, M>::insert(InputIterator const &b, InputIterator const &e, THash const &hash)
{
    for (auto it = b; it != e; ++it) { insert(*it, hash); }
}
//**************************************************************************************************

template<typename P, typename M> size_t
UnorderedSet<P, M>::size(key_type const &s) const
{
    typename base_type::const_accessor acc;

    size_t res = 0;

    if (this->find(acc, s)) { res = acc->second.size(); }

    return res;
}

template<typename P, typename M> size_t
UnorderedSet<P, M>::size(typename base_type::value_type const &item) const { return item.second.size(); }


template<typename P, typename M>
template<typename TRange> size_t
UnorderedSet<P, M>::size(TRange const &r) const
{
    return parallel::parallel_reduce(
            r, 0U,
            [&](TRange const &r, size_t init) -> size_t
            {
                for (auto const &s:r) { init += size(s); }

                return init;
            },
            [](size_t x, size_t y) -> size_t
            {
                return x + y;
            }
    );
}

template<typename P, typename M> size_t
UnorderedSet<P, M>::size() const
{
    return size(this->range());
}


//**************************************************************************************************


template<typename P, typename M> void
UnorderedSet<P, M>::erase(typename base_type::range_type const &r)
{
    UNIMPLEMENTED;

};


template<typename P, typename M> template<typename TRange> void
UnorderedSet<P, M>::erase(TRange const &r)
{
    parallel::parallel_for(r, [&](TRange const &r) { for (auto const &s:r) { base_type::erase(s); }});
}
//**************************************************************************************************



template<typename P, typename M>
template<typename OutputIterator> OutputIterator
UnorderedSet<P, M>::copy(OutputIterator out_it, key_type const &s) const
{
    typename base_type::const_accessor c_accessor;
    if (base_type::find(c_accessor, s))
    {
        out_it = std::copy(c_accessor->second.begin(), c_accessor->second.end(), out_it);
    }
    return out_it;
}

template<typename P, typename M>
template<typename OutputIT, typename TRange> OutputIT
UnorderedSet<P, M>::copy(OutputIT out_it, TRange const &r) const
{
    //TODO need optimize
    for (auto const &s:r) { out_it = copy(out_it, s); }
    return out_it;
}

template<typename P, typename M>
template<typename OutputIterator> OutputIterator
UnorderedSet<P, M>::copy(OutputIterator out_it) const
{
    return copy(out_it, this->range());
}
//*******************************************************************************


template<typename P, typename M> void
UnorderedSet<P, M>::merge(buffer_type *buffer) { merge(buffer->range(), buffer); }

template<typename P, typename M> template<typename TRange> void
UnorderedSet<P, M>::merge(TRange const &r0, buffer_type *other)
{
    parallel::parallel_for(
            r0,
            [&](TRange const &r)
            {
                for (auto const &s:r)
                {
                    typename base_type::accessor acc0;


                    if (other->find(acc0, s))
                    {
                        typename base_type::accessor acc1;
                        base_type::insert(acc1, s);
                        acc1->second.splice(acc1->second.end(), acc0->second);
                    }
                }
            }
    );
}

//*******************************************************************************


template<typename P, typename M> template<typename Hash> void
UnorderedSet<P, M>::rehash(key_type const &key, Hash const &hash, buffer_type *out_buffer)
{
    ASSERT(out_buffer != nullptr);

    typename buffer_type::accessor acc0;

    if (this->find(acc0, key))
    {

        auto &src = acc0->second;

        auto it = src.begin(), ie = src.end();

        while (it != ie)
        {
            auto p = it;

            ++it;
            auto s = hash(*p);
            if (s != key)
            {

                typename buffer_type::accessor acc1;

                out_buffer->insert(acc1, s);

                acc1->second.splice(acc1->second.end(), src, p);
            }
        }


    }
    acc0.release();


}

template<typename P, typename M> template<typename TRange, typename Hash> void
UnorderedSet<P, M>::rehash(TRange const &r0, Hash const &hash, buffer_type *out_buffer)
{
    ASSERT(out_buffer != nullptr);

    parallel::parallel_for(
            r0,
            [&](TRange const &r) { for (auto const &s:r) { rehash(s, hash, out_buffer); }}
    );

}

template<typename P, typename M> template<typename Hash> void
UnorderedSet<P, M>::rehash(Hash const &hash, buffer_type *out_buffer)
{
    if (out_buffer == nullptr)
    {
        buffer_type tmp;

        rehash(this->range(), hash, &tmp);

        this->merge(&tmp);
    }
    else
    {
        rehash(this->range(), hash, out_buffer);
    }


}


//*******************************************************************************

template<typename P, typename M> template<typename Predicate> void
UnorderedSet<P, M>::remove_if(Predicate const &pred, key_type const &s)
{
    typename base_type::accessor acc;

    if (base_type::find(acc, std::get<0>(s)))
    {
        acc->second.remove_if([&](value_type const &p) { return pred(p, s); });
    }
}


template<typename P, typename M> template<typename Predicate> void
UnorderedSet<P, M>::remove_if(Predicate const &pred, typename base_type::value_type const &item)
{
    item.second.remove_if([&](value_type const &p) { return pred(p, item.first); });
}


template<typename P, typename M> template<typename TRange, typename Predicate> void
UnorderedSet<P, M>::remove_if(Predicate const &pred, key_type const &r0)
{
    parallel::parallel_for(r0, [&](TRange const &r) { for (auto const &s:r) { remove_if(pred, s); }});
}

template<typename P, typename M> template<typename Predicate> void
UnorderedSet<P, M>::remove_if(Predicate const &pred)
{
    remove_if(base_type::range(), pred);
}


template<typename P, typename M> template<typename TFun> void
UnorderedSet<P, M>::filter(TFun const &fun, key_type const &s)
{
    typename base_type::accessor acc;

    if (base_type::find(acc, s)) { for (auto &p:acc->second) { fun(&p); }}

};

template<typename P, typename M> template<typename TFun> void
UnorderedSet<P, M>::filter(TFun const &fun, key_type const &s) const
{
    typename base_type::const_accessor acc;

    if (base_type::find(acc, s)) { for (auto const &p:acc->second) { fun(p); }}
};

template<typename P, typename M> template<typename TFun> void
UnorderedSet<P, M>::filter(TFun const &fun, typename base_type::value_type const &item)
{
    for (auto &p:item.second) { fun(&p); }
}

template<typename P, typename M> template<typename TFun> void
UnorderedSet<P, M>::filter(TFun const &fun, typename base_type::value_type const &item) const
{
    for (auto const &p:item.second) { fun(p); }
}


template<typename P, typename M>
template<typename TRange, typename TFun> void
UnorderedSet<P, M>::filter(TFun const &fun, TRange const &r0)
{
    parallel::parallel_for(r0, [&](TRange const &r) { for (auto const &s:r) { filter(fun, s); }});
}

template<typename P, typename M>
template<typename TRange, typename TFun> void
UnorderedSet<P, M>::filter(TFun const &fun, TRange const &r0) const
{
    parallel::parallel_for(r0, [&](TRange const &r) { for (auto const &s:r) { filter(fun, s); }});
}

template<typename P, typename M> template<typename TFun> void
UnorderedSet<P, M>::filter(TFun const &fun)
{
    filter(base_type::range(), fun);
};

template<typename P, typename M> template<typename TFun> void
UnorderedSet<P, M>::filter(TFun const &fun) const
{
    filter(base_type::range(), fun);
};


}}//namespace simpla { namespace gtl
#endif //SIMPLA_UNORDERED_SET_H
