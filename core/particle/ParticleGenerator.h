/**
 * @file ParticleGenerator.h
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
#include "../parallel/MPIComm.h"
#include "../parallel/MPIUpdate.h"

namespace simpla { namespace particle
{
namespace traits
{
template<typename T>
struct fun_constant
{
    T m_v_;

    fun_constant(T const &v) : m_v_(v) { }

    fun_constant(fun_constant const &other) : m_v_(other.m_v_) { }

    ~fun_constant() { }

    template<typename ...Args>
    T const &operator()(Args &&...args) const { return m_v_; }
};
}
// namespace traits;
template<typename ...> class Particle;

template<typename Engine, typename Func=Real,
        typename XDist=rectangle_distribution<3>,
        typename VDist=multi_normal_distribution<3>,
        typename TSeedGen=std::mt19937>
struct ParticleGenerator;

template<typename ...> struct ParticleEngine;

template<typename TAGS>
using generator_t=ParticleGenerator<ParticleEngine<TAGS> >;

template<typename TEngine, typename Func>
ParticleGenerator<TEngine, Func>
make_generator(TEngine const &p, Func const &func)
{
    return ParticleGenerator<TEngine, Func>(p);
};

template<typename TEngine>
ParticleGenerator<TEngine, traits::fun_constant<Real> >
make_generator(TEngine const &p, Real f)
{
    return ParticleGenerator<TEngine, traits::fun_constant<Real>>(p, traits::fun_constant<Real>(f));
};

template<typename Engine,
        typename Func,
        typename XDist,
        typename VDist,
        typename TSeedGen>
struct ParticleGenerator
{

private:

    static constexpr int NUM_OF_SEED_PER_SAMPLE = 6;

    typedef Engine particle_type;

    typedef TSeedGen seed_type;

    typedef ParticleGenerator<particle_type, seed_type> this_type;

    typedef XDist x_dist_engine;

    typedef VDist v_dist_engine;

    typedef Func function_type;

    function_type m_func_;


    typedef typename particle_type::sample_type value_type;

    particle_type const &m_p_engine_;

    std::mutex m_seed_mutex;

    seed_type m_seed_;


public:

    template<typename ...Args>
    ParticleGenerator(particle_type const &p, Func const &fun)
            : m_p_engine_(p), m_func_(fun)
    {
    }

    ParticleGenerator(ParticleGenerator const &other)
            : m_p_engine_(other.m_p_engine_), m_seed_(other.m_seed_), m_func_(other.m_func_)
    {
    }

    ~ParticleGenerator()
    {
    }


    struct input_iterator : public std::iterator<std::input_iterator_tag, value_type>
    {
    private:
        particle_type const &m_p_engine_;

        seed_type m_seed_;

        x_dist_engine m_x_dist_;
        v_dist_engine m_v_dist_;
        function_type const &m_func_;
        Real m_inv_sample_density_;

        size_t m_count_;
        value_type m_value_;
    public:

        template<typename TArgs1, typename TArgs2, typename TArgs3>
        input_iterator(particle_type const &p_engine, seed_type seed, Real sample_density,
                       TArgs1 const &args1, TArgs2 const &args2, TArgs3 const &args3)
                : m_p_engine_(p_engine),
                  m_seed_(seed), m_x_dist_(args1), m_v_dist_(args2), m_func_(args3),
                  m_count_(0),
                  m_inv_sample_density_(1.0 / sample_density)
        {
            generate_();
        }

        input_iterator(input_iterator const &other)
                : m_p_engine_(other.m_p_engine_),
                  m_seed_(other.m_seed_),
                  m_x_dist_(other.m_x_dist_),
                  m_v_dist_(other.m_v_dist_),
                  m_func_(other.m_func_),
                  m_count_(other.m_count_),
                  m_value_(other.m_value_),
                  m_inv_sample_density_(other.m_inv_sample_density_)
        {
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
            auto x = m_x_dist_(m_seed_);
            auto v = m_v_dist_(m_seed_);
            m_value_ = m_p_engine_.sample(x, v,
                                          m_func_(x, v) * m_inv_sample_density_);
        }

    }; //struct input_iterator

    /**
     *
     *  @param pic      number of sample point
     *  @param volume   volume of sample region
     *  @param box      shape of spatial sample region (e.g. box(x_min,x_max))
     *  @param args...  other of args for v_dist
     */
    template<typename TBox, typename ...Args>
    std::tuple<input_iterator, input_iterator>
    generator(size_t pic, Real volume, TBox const &box, Args &&...args)
    {
        Real sample_density = static_cast<Real>(pic) / volume;

        std::lock_guard<std::mutex> guard(m_seed_mutex);

        input_iterator ib(m_p_engine_, m_seed_, sample_density,
                          box, std::forward<Args>(args)..., m_func_);

        input_iterator ie(ib);

        ie.advance(pic);

        m_seed_.discard(pic * NUM_OF_SEED_PER_SAMPLE);

        return std::make_tuple(ib, ie);

    };

    void discard(size_t num)
    {
        std::lock_guard<std::mutex> guard(m_seed_mutex);

        m_seed_.discard(num * NUM_OF_SEED_PER_SAMPLE);
    }

    void reserve(size_t num)
    {
        std::lock_guard<std::mutex> guard(m_seed_mutex);
        int offset = 0;
        int total = 0;
        std::tie(offset, total) =
                parallel::sync_global_location(GLOBAL_COMM,
                                               static_cast<int>(num * NUM_OF_SEED_PER_SAMPLE));

        m_seed_.discard(offset);

    }

};


}}//namespace simpla{namespace particle


#endif /* CORE_PARTICLE_PARTICLE_GENERATOR_H_ */
