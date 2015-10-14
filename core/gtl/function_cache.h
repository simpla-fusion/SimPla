/**
 * @file function_cache.h
 *
 *  Created on: 2015-6-8
 *      Author: salmon
 */

#ifndef CORE_GTL_FUNCTION_CACHE_H_
#define CORE_GTL_FUNCTION_CACHE_H_

#include <functional>

namespace simpla {
namespace gtl {
template<typename RectValue, typename TFun, typename Hash>
struct FunctionCache
{
    typedef TFun function_type;

    typedef Hash hash_fun;

    typedef size_t key_type;

    typedef RectValue value_type;

    function_type m_fun_;

    hash_fun m_hash_;

    std::map <key_type, value_type> m_cache_;

    FunctionCache(function_type const &f) :
            m_fun_(f), m_hash_(hash_fun())
    {

    }

    ~FunctionCache()
    {
    }

    template<typename ...T>
    value_type operator()(T &&...args) const
    {
        auto key = m_hash_(std::forward<T>(args)...);

        auto it = m_cache_.find(key);

        if (it != m_cache_.end())
        {
            return it->second;
        }
        else
        {
            auto res = static_cast<value_type>(m_fun_(std::forward<T>(args)...));
            m_cache_[key] = res;
            return std::move(res);
        }

    }
};

template<typename RectValue, typename TFun>
struct FunctionCache<RectValue, TFun, void>
{
    typedef FunctionCache<RectValue, TFun, void> this_type;

    typedef TFun function_type;

    typedef size_t key_type;

    typedef RectValue value_type;

    function_type m_fun_;

    std::map <key_type, value_type> m_cache_;

    FunctionCache(function_type const &f) :
            m_fun_(f)
    {

    }

    ~FunctionCache()
    {
    }

    template<typename T>
    value_type operator()(T key) const
    {

        auto it = m_cache_.find(key);

        if (it != m_cache_.end())
        {
            return it->second;
        }
        else
        {
            auto res = static_cast<value_type>(m_fun_(key));
            const_cast<this_type *>(this)->m_cache_[key] = res;
            return std::move(res);
        }

    }
};

template<typename TRect, typename TFun>
FunctionCache<TRect, TFun, void> make_function_cache(TFun const &fun)
{
    return FunctionCache<TRect, TFun, void>(fun);
}

template<typename TRect, typename TFun, typename Hash>
FunctionCache<TRect, TFun, Hash> make_function_cache(TFun const &fun,
                                                     Hash const &hash)
{
    return FunctionCache<TRect, TFun, Hash>(fun, hash);
}
}
}//  namespace simpla::gtl

#endif /* CORE_GTL_FUNCTION_CACHE_H_ */
