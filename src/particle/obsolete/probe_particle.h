/**
 * @file probe_Particle.h
 *
 *  Created on: 2014-11-18
 *      Author: salmon
 */

#ifndef CORE_PARTICLE_PROBE_PARTICLE_H_
#define CORE_PARTICLE_PROBE_PARTICLE_H_

#include <iostream>
#include <memory>
#include <string>

#include "../gtl/dataset/dataset.h"
#include "../gtl/enable_create_from_this.h"
#include "../physics/physical_object.h"
#include "../gtl/utilities/utilities.h"
#include "../gtl/primitives.h"

namespace simpla {
/**
 * @ingroup particle
 *
 * @brief  ProbeParticle is a container of particle trajectory
 *
 *
 * @TODO
 * @{   It can cache the history of particle position.
 *
 *  function next_time_step(Point * p, Real dt, Args && ...)
 *  * p - m0  particle position  at m0 steps before
 *  * p - 1   particle position  at last time step
 *  * p       particle position  at current time step
 *  * p + 1   particle position  at next time step
 *  * p + 2   predicate particle position  after next two time steps
 *  * p + m1  predicate particle position  after next m1 time steps
 *  * default: m0=m1=0
 *
 *
 *  ## Requirement:
 *    engine_type::next_time_step(Point_s * p, others...);
 *
 *  - if  engine_type::memory_length  is not defined
 *     p point the "particle" at current time step
 *
 *  - if engine_type::memory_length = m
 *    p-1,p-2,... , p-m are valid and point to "Particles" at previous m steps
 *
 *
 * @}
 *
 *  ## Implement
 *  -
 */
template<typename Engine>
struct ProbeParticle : public SpObject,
                       public Engine,
                       public enable_create_from_this<ProbeParticle<Engine>>
{

public:
    typedef Engine engine_type;

    typedef ProbeParticle<engine_type> this_type;

    typedef typename engine_type::Point_s Point_s;

    typedef enable_split_from_this <this_type> base_type;

    ProbeParticle();

    ProbeParticle(ProbeParticle &, split);

    ProbeParticle(ProbeParticle const &);

    ~ProbeParticle();

    using engine_type::properties;

    std::ostream &print(std::ostream &os) const
    {
        engine_type::print(os);
        return os;
    }

    bool deploy();

    void sync();

    DataSet dataset() const;

    template<typename TDict, typename ...Others>
    void load(TDict const &dict, Others &&...others);

    static std::string get_type_as_string_staic()
    {
        return engine_type::get_type_as_string();
    }

    std::string get_type_as_string() const
    {
        return get_type_as_string_staic();
    }

    /**
     *
     * @param args arguments
     *
     * - Semantics
     @code
     for( Point_s & point: all particle)
     {
     engine_type::next_time_step(& point,std::forward<Args>(args)... );
     }
     @endcode
     *
     */
    template<typename ...Args>
    void next_timestep(Args &&...args);

    /**
     *
     * @param num_of_steps number of time steps
     * @param t0 start time point
     * @param dt delta time step
     * @param args other arguments
     * @return t0+num_of_steps*dt
     *
     *-Semantics
     @code
     for(s=0;s<num_of_steps;++s)
     {
     for( Point_s & point: all particle)
     {
     engine_type::next_time_step(& point,t0+s*dt,dt,std::forward<Args>(args)... );
     }
     }
     return t0+num_of_steps*dt;
     @endcode
     */
    template<typename ...Args>
    Real next_n_timesteps(size_t num_of_steps, Real t0, Real dt,
                          Args &&...args);

    /**
     *  push_back and emplace will invalid data in the cache
     * @param args
     */
    template<typename ...Args>
    void push_back(Args &&...args)
    {
        if (cache_is_valid())
        {
            download_cache();
        }

        data_.push_back(std::forward<Args>(args)...);
        cache_is_valid_ = false;
    }

    template<typename ...Args>
    void emplac_back(Args &&...args)
    {
        if (cache_is_valid())
        {
            download_cache();
        }

        data_.emplac_back(std::forward<Args>(args)...);

        cache_is_valid_ = false;
    }

    void increase_step_counter(size_t num_of_steps = 1);

    //! @name   @ref entity_id_range
    //! @{
    bool empty() const
    {
        return end_ <= begin_;
    }

    bool is_divisible() const
    {
        return (end_ - begin_) > 1;
    }

    template<typename TFun, typename ...Args>
    void foreach(TFun const &fun, Args &&...);
    //! @}

    void upload_cache();

    void download_cache();

    void resize_cache(size_t depth);

    bool cache_is_valid() const
    {
        return engine_type::properties.is_changed() || cache_is_valid_;
    }

    void cache_depth(size_t d)
    {
        resize_cache(d);
    }

    size_t cache_depth() const
    {
        return cache_depth_;
    }

    DataSet cache();

    DataSet cache() const;

    std::function<void(this_type const &)> on_cache_full;

    Point_s &operator[](size_t s)
    {
        return *(cache_.get() + (s * cache_depth_ + step_counter_));
    }

    Point_s const &operator[](size_t s) const
    {
        return *(cache_.get() + (s * cache_depth_ + step_counter_));
    }

private:

    std::vector<Point_s> data_;
    std::shared_ptr<Point_s> cache_;

    bool cache_is_valid_ = false;

    size_t step_counter_ = 0;
    size_t cache_depth_ = 0;
    size_t cache_width_ = 0;

    size_t begin_ = 0, end_ = 0;

    CHECK_MEMBER_VALUE(memory_length, 0);

    static constexpr size_t memory_length = check_member_value_memory_length<engine_type>::value;

    static constexpr bool is_markov_chain = (check_member_value_memory_length<engine_type>::value == 0);

    HAS_MEMBER_FUNCTION(next_timestep);

    template<typename TPIterator, typename ...Args>
    inline auto next_timestep_selector_(TPIterator p, Real time, Args &&... args) const
    -> typename std::enable_if<
            has_member_function_next_timestep<Engine, TPIterator, Real, Args...>::value, void>::type
    {
        engine_type::next_timestep(p, time, std::forward<Args>(args)...);
    }

    template<typename TPIterator, typename ...Args>
    inline auto next_timestep_selector_(TPIterator p, Real time, Args &&...args) const
    -> typename std::enable_if<
            (!has_member_function_next_timestep<Engine, TPIterator, Real, Args...>::value) &&
            (has_member_function_next_timestep<Engine, TPIterator, Args...>::value), void>::type
    {
        engine_type::next_timestep(p, std::forward<Args>(args)...);
    }

    template<typename TPIterator, typename ...Args>
    inline auto next_timestep_selector_(TPIterator p, Real time, Real dt, Args &&...args) const
    -> typename std::enable_if<
            (!has_member_function_next_timestep<Engine, TPIterator, Real, Real, Args...>::value) &&
            (!has_member_function_next_timestep<Engine, TPIterator, Real, Args...>::value), void>::type
    {
        RUNTIME_ERROR("Wrong Way");
    }

    this_type &self()
    {
        return *this;
    }

    this_type const &self() const
    {
        return *this;
    }
};

template<typename Engine>
ProbeParticle<Engine>::ProbeParticle()
{
}

template<typename Engine>
ProbeParticle<Engine>::ProbeParticle(ProbeParticle const &)
{

}

template<typename Engine>
ProbeParticle<Engine>::ProbeParticle(ProbeParticle<Engine> &other, split) :
        base_type(other)
{
    begin_ = other.begin_;
    end_ = (other.begin_ + other.end_) / 2;
    other.begin_ = end_;
}

template<typename Engine>
ProbeParticle<Engine>::~ProbeParticle()
{
}

template<typename Engine>
template<typename TDict, typename ...Others>
void ProbeParticle<Engine>::load(TDict const &dict, Others &&...others)
{

    if (dict["URL"])
    {
        UNIMPLEMENTED2(" read particle from file");
    }
    else
    {
        engine_type::load(dict, std::forward<Others>(others)...);
    }

}

template<typename Engine>
void ProbeParticle<Engine>::upload_cache()
{
    engine_type::properties("CacheLength").as(&cache_depth_);

    cache_width_ = data_.size() / (memory_length + 1);

    cache_ = sp_make_shared_array<Point_s>(cache_width_ * cache_depth_);

    //  move data from buffer_ to data_
//	for (size_t s = begin_; s < end_; ++s)
//	{
//		Point_s * p = cache_.get();
//
//		for (int i = 0; i <= memory_length; ++i)
//		{
//			p[s * (cache_depth_ + 1) + i] = ext_buffer[s * memory_length + i];
//		}
//
//	}

    cache_is_valid_ = true;

}

template<typename Engine>
void ProbeParticle<Engine>::download_cache()
{

}

template<typename Engine>
void ProbeParticle<Engine>::increase_step_counter(size_t num_of_steps)
{
    step_counter_ += num_of_steps;

    if (step_counter_ >= cache_depth_)
    {
        if (on_cache_full)
        {
            on_cache_full(*this);
        }

        for (size_t s = begin_; s < end_; ++s)
        {
            auto *p = cache_.get() + (s * cache_depth_);

            for (int i = 0; i <= memory_length; ++i)
            {
                p[i] = p[step_counter_ - memory_length + i];
            }
        }
    }
    step_counter_ = memory_length;
}

template<typename Engine>
bool ProbeParticle<Engine>::deploy()
{
    engine_type::update_properties();
    engine_type::update();

    return true;
}

template<typename Engine>
void ProbeParticle<Engine>::sync()
{
    VERBOSE << "Nothing to do." << endl;
}

template<typename Engine>
DataSet ProbeParticle<Engine>::dataset() const
{
    size_t dims[2] = {cache_width_, memory_length + 1};

    return std::move(make_dataset(cache_.get() + (

            cache_width_ * (step_counter_ - memory_length)

    ), 1, dims, properties()));
}

template<typename Engine>
DataSet ProbeParticle<Engine>::cache() const
{
    size_t dims[2] = {cache_width_, step_counter_};

    return std::move(make_dataset(cache_, 1, dims, properties()));
}

template<typename Engine>
template<typename TFun, typename ...Args>
void ProbeParticle<Engine>::foreach(TFun const &fun, Args &&... args)
{
    for (size_t s = begin_; s < end_; ++s)
    {
        fun(&(*this)[s], std::forward<Args>(args)...);
    }
}

template<typename Engine>
template<typename ... Args>
void ProbeParticle<Engine>::next_timestep(Args &&...args)
{
    if (cache_is_valid())
    {
        upload_cache();
    }

    foreach(

            [&](Point_s *p)
                {
                engine_type::next_timestep(&((*this)[s]), std::forward<Args>(args)...);
                }

    );

    increase_step_counter();
}

template<typename Engine>
template<typename ... Args>
Real ProbeParticle<Engine>::next_n_timesteps(size_t num_of_steps, Real t0,
                                             Real dt, Args &&...args)
{

    if (!cache_is_valid())
    {
        upload_cache();
    }

    if ((num_of_steps + step_counter_) > cache_depth_)
    {
        size_t n0 = cache_depth_ - step_counter_;
        size_t n1 = num_of_steps + step_counter_ - cache_depth_;

        t0 = next_n_timesteps(n0, t0, dt, std::forward<Args>(args)...);

        t0 = next_n_timesteps(n1, t0, dt, std::forward<Args>(args)...);

    }
    else
    {

        foreach(

                [&](Point_s *p)
                    {
                    Real t = t0;
                    for (int i = 0; i < num_of_steps; ++i)
                    {
                        next_timestep_selector_(p, t, dt, std::forward<Args>(args)...);
                        ++p;
                        t += dt;
                    }
                    }

        );

        t0 += num_of_steps * dt;

        increase_step_counter(num_of_steps);
    }

    return t0;

}

template<typename Engine, typename ...Others>
auto make_probe_particle(Others &&... others)
DECL_RET_TYPE((ProbeParticle<Engine>::create(
        std::forward<Others>(others)...)))

}  // namespace simpla

#endif /* CORE_PARTICLE_PROBE_PARTICLE_H_ */
