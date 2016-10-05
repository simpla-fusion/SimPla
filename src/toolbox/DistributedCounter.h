/**
 * @file DistributedCounter.h
 * @author salmon
 * @date 2016-01-17.
 */

#ifndef SIMPLA_DISTRIBUTEDCOUNTER_H
#define SIMPLA_DISTRIBUTEDCOUNTER_H

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include "type_traits.h"
#include "../sp_def.h"
#include "MPIComm.h"
#include "MPIUpdate.h"

namespace simpla { namespace parallel
{

struct DistributedCounter
{

    std::atomic<size_t> m_start_, m_end_;

public:

    DistributedCounter() : m_start_(0), m_end_(0) { }

    virtual ~DistributedCounter() { }

    DistributedCounter(DistributedCounter const &other) = delete;

    /**
     * // thread safe
     *  @param pic      number of sample point
     *  @param volume   volume of sample region
     *  @param box      shape of spatial sample region (e.g. box(x_min,x_max))
     *  @param args...  other of args for v_dist
     */
    size_t get(size_t num) { return (m_start_ += num) - num; };


    void reserve(size_t num)
    {
        ASSERT(m_start_.load() == 0);

        int offset = 0;
        int total = 0;
        std::tie(m_start_, std::ignore) =
                parallel::sync_global_location(GLOBAL_COMM, static_cast<int>(num ));
        m_start_ = offset;
        m_end_ = offset + num;

    }

};

}}
#endif //SIMPLA_DISTRIBUTEDCOUNTER_H
