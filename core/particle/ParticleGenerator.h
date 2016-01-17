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
#include <random>
#include "../gtl/type_traits.h"
#include "../gtl/primitives.h"
#include "../numeric/rectangle_distribution.h"
#include "../numeric/multi_normal_distribution.h"
#include "../parallel/DistributedCounter.h"

namespace simpla { namespace particle
{

template<typename TV,
        typename TSeed,
        typename XDist,
        typename VDist>
struct ParticleGenerator : public parallel::DistributedCounter
{

public:
    typedef TV value_type;

    typedef TSeed seed_type;

    typedef XDist x_dist_engine;

    typedef VDist v_dist_engine;

    static constexpr int NUM_OF_SEED_PER_SAMPLE = 6;
private:
    TSeed m_seed_;

public:
    ParticleGenerator() { }

    ParticleGenerator(TSeed const &seed) : m_seed_(seed) { }

    virtual  ~ParticleGenerator() { }

    ParticleGenerator(ParticleGenerator const &other) = delete;

    using parallel::DistributedCounter::reserve;


    struct input_iterator : public std::iterator<std::input_iterator_tag, value_type>
    {
    private:

        x_dist_engine m_x_dist_;

        v_dist_engine m_v_dist_;

        size_t m_count_;

        std::function<void(nTuple<Real, 3> const &, nTuple<Real, 3> const &, value_type *)> m_func_;

        seed_type m_seed_;

        value_type m_value_;


    public:

        template<typename TFun>
        input_iterator(size_t start, TFun const &func)
                : m_count_(start), m_func_(func)
        {
            m_seed_.discard(m_count_ * NUM_OF_SEED_PER_SAMPLE);
            this->operator++();
            --m_count_;
        }

        template<typename TFun>
        input_iterator(size_t start, TFun const &func, seed_type const &seed)
                : m_seed_(seed), m_count_(start), m_func_(func)
        {
            m_seed_.discard(m_count_ * NUM_OF_SEED_PER_SAMPLE);
            this->operator++();
            --m_count_;
        }

        input_iterator(size_t start) : m_count_(start) { }

        input_iterator(input_iterator const &other) :
                m_count_(other.m_count_), m_seed_(other.m_seed_), m_func_(other.m_func_), m_value_(other.m_value_)
        {
        }

        virtual ~input_iterator() { }

        value_type const &operator*() const { return m_value_; }

        value_type const *operator->() const { return &m_value_; }

    private:
        HAS_MEMBER(_tags);

        void set_tags_(std::integral_constant<bool, false>) const { }

        void set_tags_(std::integral_constant<bool, true>) const { m_value_._tags = m_count_; }


    public:
        input_iterator &operator++()
        {

            nTuple<Real, 3> x = m_x_dist_(m_seed_);
            nTuple<Real, 3> v = m_v_dist_(m_seed_);

            size_t _tags = m_count_;

            if (m_func_) { m_func_(x, v, &m_value_); }
            set_tags_(std::integral_constant<bool, has_member__tags<size_t>::value>());
            ++m_count_;
            return *this;
        }

        size_t count() const { return m_count_; }

        bool operator==(input_iterator const &other) const { return m_count_ == other.m_count_; }

        bool operator!=(input_iterator const &other) const { return (m_count_ != other.m_count_); }


    }; //struct input_iterator

    template<typename TFunc> std::tuple<input_iterator, input_iterator>
    generate(size_t num, TFunc const &func)
    {
        size_t ib = parallel::DistributedCounter::get(num);
        return std::make_tuple(input_iterator(ib, func, m_seed_), input_iterator(ib + num));
    }

};

template<typename TP, typename TSeed, typename XDist, typename VDist>
class ParticleGeneratorPerCell
        : public ParticleGenerator<typename TP::sample_type, TSeed, XDist, VDist>
{
    typedef ParticleGenerator<typename TP::sample_type, TSeed, XDist, VDist> base_type;
    typedef ParticleGeneratorPerCell<TP, TSeed, XDist, VDist> this_type;
    typedef TP particle_type;
    typedef typename particle_type::mesh_type mesh_type;
    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::point_type point_type;

    particle_type const &m_particle_;
    size_t m_num_per_cell_ = 1;

    std::function<Real(point_type const &)> m_density_;
    std::function<Real(point_type const &)> m_temperature_;

public:
    template<typename ...Args>
    ParticleGeneratorPerCell(particle_type const &e, size_t num_per_cell, Args &&...args)
            : base_type(std::forward<Args>(args)  ...), m_particle_(e), m_num_per_cell_(num_per_cell)
    {
        base_type::reserve(m_particle_.mesh().template range<particle_type::iform>().size() * m_num_per_cell_);
    }

    ~ParticleGeneratorPerCell() { }

    size_t num_per_cell(size_t n) { m_num_per_cell_ = n; }

    size_t num_per_cell() const { return m_num_per_cell_; }

    template<typename TFun> void density(TFun const &fun) { m_density_ = fun; }

    template<typename TFun> void temperature(TFun const &fun) { m_temperature_ = fun; }

//    using base_type::input_iterator;
//    using base_type::value_type;

    std::tuple<typename base_type::input_iterator, typename base_type::input_iterator>
    operator()(id_type const &s)
    {
        Real m_mass_;
        Real m_charge_;
        Real mass = m_particle_.engine().mass();
        Real charge = m_particle_.engine().charge();
        mesh_type const &m = m_particle_.mesh();

        Real inv_sample_density = m.volume(s) / num_per_cell();

        DEFINE_PHYSICAL_CONST;


        return base_type::generate(
                num_per_cell(),
                [&](nTuple<Real, 3> const &x0, nTuple<Real, 3> const &v0,
                    typename base_type::value_type *p)
                {
                    nTuple<Real, 3> x = m.coordinates_local_to_global(std::make_tuple(s, x0));

                    nTuple<Real, 3> v;

//            v = v0 * std::sqrt(2.0 * m_temperature_(x) * boltzmann_constant / mass);
//
//            *p = m_particle_.engine().lift(x, v, m_density_(x) * inv_sample_density);
                });
    }
};

template<typename TV> using DefaultParticleGenerator=ParticleGeneratorPerCell<TV,
        std::mt19937,
        rectangle_distribution<3>,
        multi_normal_distribution<3> >;

}}//namespace simpla{namespace particle


#endif /* CORE_PARTICLE_PARTICLE_GENERATOR_H_ */
