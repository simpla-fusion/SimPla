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
#include "../gtl/type_traits.h"
#include "../gtl/primitives.h"
#include "../parallel/MPIComm.h"
#include "../parallel/MPIUpdate.h"

namespace simpla { namespace parallel
{

template<typename TV, typename TSeed, int NUM_OF_SEED_PER_SAMPLE = 6>
struct RandomGenerator
{

private:
    typedef RandomGenerator<TV, TSeed, NUM_OF_SEED_PER_SAMPLE> this_type;

    typedef TV value_type;
    typedef TSeed seed_type;
    typedef std::function<value_type(seed_type &)> function_type;

    std::atomic<size_t> m_start_, m_end_;

    function_type m_func_;


public:

    RandomGenerator(function_type const &fun) : m_func_(fun)
    {
    }

    RandomGenerator(RandomGenerator const &other) : m_func_(other.m_func_)
    {
    }

    ~RandomGenerator()
    {
    }


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
    generator(size_t num)
    {
        size_t end = (m_start_ += num);
        size_t start = end - num;

        return std::make_tuple(input_iterator(start, m_func_), input_iterator(end, m_func_));

    };


    void reserve(size_t num)
    {
        ASSERT(m_start_.load() == 0);

        int offset = 0;
        int total = 0;
        std::tie(m_start_, std::ignore) =
                parallel::sync_global_location(GLOBAL_COMM,
                                               static_cast<int>(num *
                                                                NUM_OF_SEED_PER_SAMPLE));
        m_start_ = offset;
        m_end_ = offset + num;

    }

};

template<typename TV, typename DistFunc, typename TSeedGen, int NUM_OF_SEED_PER_SAMPLE>
struct RandomGenerator<TV, DistFunc, seed_type>::input_iterator : public std::iterator<std::input_iterator_tag, TV>
{
private:
    typedef TV value_type;
    size_t m_count_;
    DistFunc m_update_func_;
    value_type m_value_;
    seed_type m_seed_;

public:


    input_iterator(size_t start, DistFunc const &fun)
            : m_count_(0), m_update_func_(fun)
    {
        advance(start);
        generate_();
    }

    // construct end tag
    input_iterator(size_t start_count)
            : m_count_(start_count)
    {

    }

    input_iterator(input_iterator const &other)
            : m_count_(0), m_update_func_(other.m_update_func_)
    {
        advance(other.m_count_);
        generate_();
    }

    ~input_iterator() { }

    value_type const &operator*() const { return m_value_; }

    value_type const *operator->() const { return &m_value_; }

    input_iterator &operator++()
    {

        ++m_count_;
        generate_();
        return *this;
    }

    void advance(size_t n)
    {
        m_seed_.discard(n * NUM_OF_SEED_PER_SAMPLE);
        m_count_ += n;
    }

    bool operator==(input_iterator const &other) const { return m_count_ == other.m_count_; }

    bool operator!=(input_iterator const &other) const { return (m_count_ != other.m_count_); }

private:
    void generate_()
    {
        m_update_func_(m_seed_, m_count_, &m_value_);
    }

}; //struct input_iterator
}}
#endif //SIMPLA_PARALLELRANDOMGENERATOR_H
