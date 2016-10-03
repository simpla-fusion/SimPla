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

namespace simpla
{
namespace toolbox
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

    template<typename Hash> void insert(value_type const &p, Hash const &hash);

    template<typename InputIterator, typename ...Others>
    void insert(InputIterator const &b, InputIterator const &e, Others &&...others)
    {
        for (auto it = b; it != e; ++it) { insert(*it, std::forward<Others>(others)...); }
    }


    template<typename InputIterator, typename ... Others>
    void insert(std::tuple<InputIterator, InputIterator> const &r, Others &&... others)
    {
        insert(std::get<0>(r), std::get<1>(r), std::forward<Others>(others)...);
    }

    using base_type::erase;

    void erase(typename base_type::range_type const &r);


    template<typename TRange> void erase(TRange const &r);


    void erase_all() { base_type::clear(); }


    size_t count(key_type const &s) const;

    size_t count(typename base_type::value_type const &v) const;

    template<typename TRange> size_t count(TRange const &r) const;

    size_t count() const;


    template<typename OutputIT> OutputIT copy_out(OutputIT out_it, key_type const &s) const;

    template<typename OutputIT, typename TRange> OutputIT copy_out(OutputIT out_it, TRange const &r) const;

    template<typename OutputIT> OutputIT copy_out(OutputIT out_it) const;


    void merge(buffer_type *other, key_type const &s);

    template<typename TRange> void merge(buffer_type *other, TRange const &r);

    void merge(base_type *other);


    template<typename Hash> void rehash(Hash const &hash, key_type const &key, buffer_type *out_buffer);

    template<typename TRange, typename Hash> void rehash(Hash const &hash, TRange const &r, buffer_type *out_buffer);

    template<typename Hash> void rehash(Hash const &hash, buffer_type *out_buffer = nullptr);

    //! @}




};//class UnorderedSet

template<typename ValueType, typename Key>
UnorderedSet<ValueType, Key>::UnorderedSet()
{

}


template<typename ValueType, typename Key>
UnorderedSet<ValueType, Key>::~UnorderedSet() { }


//**************************************************************************************************
template<typename ValueType, typename Key> void
UnorderedSet<ValueType, Key>::insert(value_type const &v, key_type const &s)
{
    typename base_type::accessor acc;

    base_type::insert(acc, s);

    acc->second.push_back(v);
}

//**************************************************************************************************

template<typename ValueType, typename Key> size_t
UnorderedSet<ValueType, Key>::count(key_type const &s) const
{
    typename base_type::const_accessor acc;

    size_t res = 0;

    if (this->find(acc, s)) { res = acc->second.size(); }

    return res;
}

template<typename ValueType, typename Key> size_t
UnorderedSet<ValueType, Key>::count(typename base_type::value_type const &item) const { return item.second.size(); }


template<typename ValueType, typename Key>
template<typename TRange> size_t
UnorderedSet<ValueType, Key>::count(TRange const &r) const
{
    return parallel::parallel_reduce(
            r, 0U,
            [&](TRange const &r, size_t init) -> size_t
            {
                for (auto const &s:r) { init += count(s); }

                return init;
            },
            [](size_t x, size_t y) -> size_t
            {
                return x + y;
            }
    );
}

template<typename ValueType, typename Key> size_t
UnorderedSet<ValueType, Key>::count() const
{
    return count(this->range());
}


//**************************************************************************************************


template<typename ValueType, typename Key> void
UnorderedSet<ValueType, Key>::erase(typename base_type::range_type const &r)
{
    UNIMPLEMENTED;

};


template<typename ValueType, typename Key> template<typename TRange> void
UnorderedSet<ValueType, Key>::erase(TRange const &r)
{
    parallel::parallel_for(r, [&](TRange const &r) { for (auto const &s:r) { base_type::erase(s); }});
}
//**************************************************************************************************



template<typename ValueType, typename Key>
template<typename OutputIterator> OutputIterator
UnorderedSet<ValueType, Key>::copy_out(OutputIterator out_it, key_type const &s) const
{
    typename base_type::const_accessor c_accessor;
    if (base_type::find(c_accessor, s))
    {
        out_it = std::copy(c_accessor->second.begin(), c_accessor->second.end(), out_it);
    }
    return out_it;
}

template<typename ValueType, typename Key>
template<typename OutputIT, typename TRange> OutputIT
UnorderedSet<ValueType, Key>::copy_out(OutputIT out_it, TRange const &r) const
{
    //TODO need optimize
    for (auto const &s:r) { out_it = copy_out(out_it, s); }
    return out_it;
}

template<typename ValueType, typename Key>
template<typename OutputIterator> OutputIterator
UnorderedSet<ValueType, Key>::copy_out(OutputIterator out_it) const
{
    return copy_out(out_it, this->range());
}
//*******************************************************************************


template<typename ValueType, typename Key> void
UnorderedSet<ValueType, Key>::merge(buffer_type *buffer) { merge(buffer, buffer->range()); }

template<typename ValueType, typename Key> template<typename TRange> void
UnorderedSet<ValueType, Key>::merge(buffer_type *other, TRange const &r0)
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


template<typename ValueType, typename Key> template<typename Hash> void
UnorderedSet<ValueType, Key>::rehash(Hash const &hash, key_type const &key, buffer_type *out_buffer)
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

template<typename ValueType, typename Key> template<typename TRange, typename Hash> void
UnorderedSet<ValueType, Key>::rehash(Hash const &hash, TRange const &r0, buffer_type *out_buffer)
{
    ASSERT(out_buffer != nullptr);

    parallel::parallel_for(
            r0,
            [&](TRange const &r) { for (auto const &s:r) { rehash(hash, s, out_buffer); }}
    );

}

template<typename ValueType, typename Key> template<typename Hash> void
UnorderedSet<ValueType, Key>::rehash(Hash const &hash, buffer_type *out_buffer)
{
    if (out_buffer == nullptr)
    {
        buffer_type tmp;

        rehash(hash, this->range(), &tmp);

        this->merge(&tmp);
    }
    else
    {
        rehash(hash, this->range(), out_buffer);
    }


}


//*******************************************************************************

template<typename ValueType, typename Key> template<typename Predicate> void
UnorderedSet<ValueType, Key>::remove_if(Predicate const &pred, key_type const &s)
{
    typename base_type::accessor acc;

    if (base_type::find(acc, std::get<0>(s)))
    {
        acc->second.remove_if([&](value_type const &p) { return pred(p, s); });
    }
}


template<typename ValueType, typename Key> template<typename Predicate> void
UnorderedSet<ValueType, Key>::remove_if(Predicate const &pred, typename base_type::value_type const &item)
{
    item.second.remove_if([&](value_type const &p) { return pred(p, item.first); });
}


template<typename ValueType, typename Key> template<typename TRange, typename Predicate> void
UnorderedSet<ValueType, Key>::remove_if(Predicate const &pred, key_type const &r0)
{
    parallel::parallel_for(r0, [&](TRange const &r) { for (auto const &s:r) { remove_if(pred, s); }});
}

template<typename ValueType, typename Key> template<typename Predicate> void
UnorderedSet<ValueType, Key>::remove_if(Predicate const &pred)
{
    remove_if(base_type::range(), pred);
}


template<typename ValueType, typename Key> template<typename TFun> void
UnorderedSet<ValueType, Key>::filter(TFun const &fun, key_type const &s)
{
    typename base_type::accessor acc;

    if (base_type::find(acc, s)) { for (auto &p:acc->second) { fun(&p); }}

};

template<typename ValueType, typename Key> template<typename TFun> void
UnorderedSet<ValueType, Key>::filter(TFun const &fun, key_type const &s) const
{
    typename base_type::const_accessor acc;

    if (base_type::find(acc, s)) { for (auto const &p:acc->second) { fun(p); }}
};

template<typename ValueType, typename Key> template<typename TFun> void
UnorderedSet<ValueType, Key>::filter(TFun const &fun, typename base_type::value_type const &item)
{
    for (auto &p:item.second) { fun(&p); }
}

template<typename ValueType, typename Key> template<typename TFun> void
UnorderedSet<ValueType, Key>::filter(TFun const &fun, typename base_type::value_type const &item) const
{
    for (auto const &p:item.second) { fun(p); }
}


template<typename ValueType, typename Key>
template<typename TRange, typename TFun> void
UnorderedSet<ValueType, Key>::filter(TFun const &fun, TRange const &r0)
{
    parallel::parallel_for(r0, [&](TRange const &r) { for (auto const &s:r) { filter(fun, s); }});
}

template<typename ValueType, typename Key>
template<typename TRange, typename TFun> void
UnorderedSet<ValueType, Key>::filter(TFun const &fun, TRange const &r0) const
{
    parallel::parallel_for(r0, [&](TRange const &r) { for (auto const &s:r) { filter(fun, s); }});
}

template<typename ValueType, typename Key> template<typename TFun> void
UnorderedSet<ValueType, Key>::filter(TFun const &fun)
{
    filter(base_type::range(), fun);
};

template<typename ValueType, typename Key> template<typename TFun> void
UnorderedSet<ValueType, Key>::filter(TFun const &fun) const
{
    filter(base_type::range(), fun);
};


}
}//namespace simpla { namespace toolbox
#endif //SIMPLA_UNORDERED_SET_H
