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

template<typename TSeed= std::mt19937,
        typename XDist=rectangle_distribution<3>,
        typename VDist= multi_normal_distribution<3>>
struct ParticleGenerator : public parallel::DistributedCounter
{

public:
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


    template<typename TV>
    struct input_iterator
    {
    public:
        typedef std::input_iterator_tag iterator_category;
        /// The type "pointed to" by the iterator.
        typedef TV value_type;
        /// Distance between iterators is represented as this type.
        typedef ptrdiff_t difference_type;
        /// This type represents a pointer-to-value_type.
        typedef TV *pointer;
        /// This type represents a reference-to-value_type.
        typedef TV reference;
    private:


        x_dist_engine m_x_dist_;

        v_dist_engine m_v_dist_;

        size_t m_count_;

        std::function<void(nTuple<Real, 3> const &, nTuple<Real, 3> const &, value_type *)> m_func_;

        seed_type m_seed_;

        value_type m_value_;


    public:

        input_iterator() { }

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
        HAS_MEMBER(_tag);

        void set_tags_(std::integral_constant<bool, false>, value_type *v) const { }

        void set_tags_(std::integral_constant<bool, true>, value_type *v) const { v->_tag = m_count_; }


    public:
        input_iterator &operator++()
        {

            nTuple<Real, 3> x = m_x_dist_(m_seed_);
            nTuple<Real, 3> v = m_v_dist_(m_seed_);

            if (m_func_) { m_func_(x, v, &m_value_); }

            set_tags_(std::integral_constant<bool, has_member__tag<value_type>::value>(), &m_value_);

            ++m_count_;

            return *this;
        }

        size_t count() const { return m_count_; }

        bool operator==(input_iterator const &other) const { return m_count_ == other.m_count_; }

        bool operator!=(input_iterator const &other) const { return (m_count_ != other.m_count_); }


    }; //struct input_iterator

    template<typename TV, typename TFunc> std::tuple<input_iterator<TV>, input_iterator<TV>>
    generate(size_t num, TFunc const &func)
    {
        size_t ib = parallel::DistributedCounter::get(num);
        return std::make_tuple(input_iterator<TV>(ib, func, m_seed_), input_iterator<TV>(ib + num));
    }

};

template<typename TGen, typename TPart, typename TFun>
void generate_particle(TPart *part, TGen &gen, size_t number_of_pic, TFun const &fun)
{
    typedef typename TPart::value_type value_type;

    DEFINE_PHYSICAL_CONST;

    mesh::MeshEntityRange r0 = part->entity_id_range();

    gen.reserve(number_of_pic * r0.size());

    parallel::parallel_for(
            r0,
            [&](mesh::MeshEntityRange const &r)
            {
                for (auto const &s:r)
                {
                    typename TGen::template input_iterator<value_type> ib, ie;
                    std::tie(ib, ie) = gen.template generate<value_type>(number_of_pic, fun);
//                            [&](nTuple<Real, 3> const &x0, nTuple<Real, 3> const &v0, value_type *res)
//                            {
//                            nTuple<Real, 3> x = m.point_local_to_global(m.miminal_vertex(s), x0);
//                            nTuple<Real, 3> v;
//
//                            v = v0 * std::sqrt(2.0 * m_temperature_(x) * boltzmann_constant / mass);
//
//                            *res = part->lift(x, v, m_density_(x) * inv_sample_density);
//                            }
                    part->insert(s, ib, ie);
                }
            });
}


}}//namespace simpla{namespace particle


#endif /* CORE_PARTICLE_PARTICLE_GENERATOR_H_ */
