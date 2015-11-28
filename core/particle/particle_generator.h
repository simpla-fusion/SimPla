/**
 * @file particle_generator.h
 *
 * @date 2015-2-12
 * @author salmon
 */

#ifndef CORE_PARTICLE_PARTICLE_GENERATOR_H_
#define CORE_PARTICLE_PARTICLE_GENERATOR_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include "../gtl/type_traits.h"
#include "../gtl/primitives.h"
#include "../numeric/rectangle_distribution.h"
#include "../numeric/multi_normal_distribution.h"

namespace simpla
{

template<typename Engine, typename TSeedGen=std::mt19937>
struct ParticleGenerator
{

private:
    typedef Engine particle_type;

    typedef TSeedGen seed_type;

    typedef ParticleGenerator<particle_type, seed_type> this_type;

    typedef rectangle_distribution<3> x_dist_engine;

    typedef multi_normal_distribution<3> v_dist_engine;


    typedef typename particle_type::sample_type value_type;

    particle_type const &m_p_engine_;

    std::mutex m_seed_mutex;

    seed_type m_seed_;

    x_dist_engine m_x_dist_;

    v_dist_engine m_v_dist_;

public:
    template<typename Args1, typename Args2>
    ParticleGenerator(particle_type const &p, Args1 const &args1,
                      Args2 const &args2)
            : m_p_engine_(p), m_x_dist_(args1), m_v_dist_(args2)
    {
    }

    ~ParticleGenerator()
    {
    }


    struct input_iterator : public std::iterator<std::input_iterator_tag, value_type>
    {
        particle_type const &m_p_engine_;
        x_dist_engine x_dist_;
        v_dist_engine v_dist_;
        std::shared_ptr<seed_type> m_seed_;

        size_t m_count_;
        value_type m_value_;

        template<typename TArgs1, typename TArgs2>
        input_iterator(particle_type const &p_engine, seed_type seed,
                       TArgs1 const &args1, TArgs2 args2)
                : m_p_engine_(p_engine), m_seed_(new seed_type(seed)),
                  x_dist_(args1), v_dist_(args2), m_count_(0)
        {
        }

        input_iterator(input_iterator const &other)
                : m_p_engine_(other.m_p_engine_), m_seed_(other.m_seed_), x_dist_(other.x_dist_),
                  v_dist_(other.v_dist_), m_count_(other.m_count_)
        {
        }

        ~input_iterator() { }

        value_type const &operator*() const { return m_value_; }

        value_type const *operator->() const { return &m_value_; }

        input_iterator &operator++()
        {
            // @note this_is not thread_safe
            auto z = m_p_engine_.lift(x_dist_(*m_seed_), v_dist_(*m_seed_));
            m_value_ = m_p_engine_.sample(x_dist_(*m_seed_), v_dist_(*m_seed_), 1.0);
            ++m_count_;
            return *this;
        }

        void advance(size_t n)
        {
            m_seed_->discard(n * 6);
            m_count_ += n;
        }

        bool operator==(input_iterator const &other) const { return m_count_ == other.m_count_; }

        bool operator!=(input_iterator const &other) const { return (m_count_ != other.m_count_); }
    }; //struct input_iterator

    template<typename ...Args>
    std::tuple<input_iterator, input_iterator>
    generator(size_t pic, Args &&...args)
    {
        // TODO lock seed
        std::lock_guard<std::mutex> guard(m_seed_mutex);

        input_iterator ib(m_p_engine_, m_seed_, std::forward<Args>(args)...);

        input_iterator ie(ib);

        ie.advance(pic);

        discard(pic);

        return std::make_tuple(ib, ie);

    };

    void discard(size_t num)
    {
        m_seed_.discard(num * 6);
    }


};


}  // namespace simpla

#endif /* CORE_PARTICLE_PARTICLE_GENERATOR_H_ */
