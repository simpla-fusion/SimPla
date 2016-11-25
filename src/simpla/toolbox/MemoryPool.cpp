/**
 * @file memory_pool.cpp
 *
 *  Created on: 2014-11-14
 *      Author: salmon
 */

#include "MemoryPool.h"

#include <stddef.h>
#include <map>
#include <mutex>
#include <new>
#include <tuple>

#include "design_pattern/SingletonHolder.h"
#include "Log.h"

namespace simpla
{

struct MemoryPool::pimpl_s
{

    std::mutex locker_;

    std::multimap<size_t, void *> pool_;

    static constexpr size_t ONE_GIGA = 1024l * 1024l * 1024l;
    static constexpr size_t MAX_BLOCK_SIZE = 4 * ONE_GIGA; //std::numeric_limits<size_t>::max();
    static constexpr size_t MIN_BLOCK_SIZE = 256;

    size_t max_pool_depth_ = 16 * ONE_GIGA;
    size_t pool_depth_ = 0;

    void push(void *d, size_t s);

    void *pop(size_t s);

    void clear();
};

MemoryPool::MemoryPool() :
        pimpl_(new pimpl_s) //2G
{
    pimpl_->pool_depth_ = 0;
}

MemoryPool::~MemoryPool()
{
    clear();
}

//!  unused memory will be freed when total memory size >= pool size
void MemoryPool::max_size(size_t s)
{
    pimpl_->max_pool_depth_ = s;
}

/**
 *  return the total size of memory in pool
 * @return
 */
double MemoryPool::size() const
{
    return static_cast<double>(pimpl_->pool_depth_);
}

void MemoryPool::clear()
{
    pimpl_->clear();
}

void MemoryPool::pimpl_s::clear()
{
    locker_.lock();
    for (auto &item : pool_)
    {
        delete[] reinterpret_cast<byte_type *>(item.second);
    }
    locker_.unlock();
}

void MemoryPool::push(void *p, size_t s)
{
    pimpl_->push(p, s);
}

void MemoryPool::pimpl_s::push(void *p, size_t s)
{
    if ((s > MIN_BLOCK_SIZE) && (s < MAX_BLOCK_SIZE))
    {
        locker_.lock();

        if ((pool_depth_ + s < max_pool_depth_))
        {
            pool_.emplace(s, p);
            pool_depth_ += s;
            p = nullptr;
        }

        locker_.unlock();

    }
    if (p != nullptr)
    {
        delete[] reinterpret_cast<byte_type *>(p);
    }

//    VERBOSE << SHORT_FILE_LINE_STAMP << "Free memory [" << s << " ]" << std::endl;


}

void *MemoryPool::pop(size_t s)
{
//    VERBOSE << SHORT_FILE_LINE_STAMP << "Allocate memory [" << s << " ]" << std::endl;

    return pimpl_->pop(s);
}

void *MemoryPool::pimpl_s::pop(size_t s)
{
    void *addr = nullptr;

    if ((s > MIN_BLOCK_SIZE) && (s < MAX_BLOCK_SIZE))
    {
        locker_.lock();

        // find memory block which is not smaller than demand size
        auto pt = pool_.lower_bound(s);

        if (pt != pool_.end())
        {
            size_t ts = 0;

            std::tie(ts, addr) = *pt;

            if (ts < s * 2)
            {
                s = ts;

                pool_.erase(pt);

                pool_depth_ -= s;
            } else
            {
                addr = nullptr;
            }
        }

        locker_.unlock();

    }

    if (addr == nullptr)
    {

        try
        {
            addr = reinterpret_cast<void *>(new byte_type[s]);

        } catch (std::bad_alloc const &error)
        {
            THROW_EXCEPTION_BAD_ALLOC(s);

        }

    }
    return addr;

}


std::shared_ptr<void> sp_alloc_memory(size_t s)
{
    void *addr = SingletonHolder<MemoryPool>::instance().pop(s);

    return std::shared_ptr<void>(addr, MemoryPool::deleter_s(addr, s));
}


}  // namespace simpla
