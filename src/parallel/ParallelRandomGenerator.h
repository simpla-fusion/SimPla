/**
 * @file ParallelRandomGenerator.h
 * @author salmon
 * @date 2016-01-17.
 */

#ifndef SIMPLA_PARALLELRANDOMGENERATOR_H
#define SIMPLA_PARALLELRANDOMGENERATOR_H

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include "../toolbox/type_traits.h"
#include "../sp_def.h"
#include "../parallel/MPIComm.h"
#include "../parallel/MPIUpdate.h"

namespace simpla { namespace parallel
{

struct DistributedCounter
{


    std::atomic<size_t> m_start_, m_end_;

public:

    DistributedCounter() : m_start_(0), m_end_(0) { }

    ~RandomGenerator() { }

    DistributedCounter(DistributedCounter const &other) = delete;

    struct input_iterator;

    /**
     *
     *  @param pic      number of sample point
     *  @param volume   volume of sample region
     *  @param box      shape of spatial sample region (e.g. box(x_min,x_max))
     *  @param args...  other of args for v_dist
     */
    template<typename TSeed>
    std::tuple<input_iterator, input_iterator>
    generator(size_t num) // thread safe
    {
        size_t end = (m_start_ += num);
        size_t start = end - num;

        return std::make_tuple(input_iterator(start), input_iterator(end));

    };


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

struct DistributedCounter::input_iterator : public std::iterator<std::input_iterator_tag, size_t>
{
private:
    size_t m_count_;
public:


    input_iterator(size_t start) : m_count_(0)
    {
        advance(start);
    }


    input_iterator(input_iterator const &other)
            : m_count_(0)
    {
        advance(other.m_count_);
    }

    ~input_iterator() { }

    value_type const &operator*() const { return m_count_; }

    value_type const *operator->() const { return &m_count_; }

    input_iterator &operator++()
    {
        ++m_count_;
        return *this;
    }

    void advance(size_t n) { m_count_ += n; }

    size_t count() const { return m_count_; }

    bool operator==(input_iterator const &other) const { return m_count_ == other.m_count_; }

    bool operator!=(input_iterator const &other) const { return (m_count_ != other.m_count_); }


}; //struct input_iterator
}}
#endif //SIMPLA_PARALLELRANDOMGENERATOR_H
