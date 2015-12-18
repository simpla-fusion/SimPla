/**
 * @file LifeClick.h
 * @author salmon
 * @date 2015-12-18.
 */

#ifndef SIMPLA_LIFECLICK_H
#define SIMPLA_LIFECLICK_H

#include <atomic>
#include "../gtl/design_pattern/singleton_holder.h"

namespace simpla { namespace base
{

struct LifeClick
{
    size_t touch(size_t *r)
    {
        m_click_.fetch_add(1);

        auto current = m_click_.load();

        if (r != nullptr) { *r = current; }

        return current;
    }

    std::atomic<size_t> m_click_;

};

#define GLOBAL_CLICK_TOUCH SingletonHolder<::simpla::base::LifeClick>::instance().touch
}}//namespace simpla{namespace base{

#endif //SIMPLA_LIFECLICK_H
