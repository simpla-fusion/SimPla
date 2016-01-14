/**
 * @file Particle.h
 *
 *  created on: 2012-11-1
 *      Author: salmon
 */

#ifndef CORE_PARTICLE_PARTICLE_H_
#define CORE_PARTICLE_PARTICLE_H_

#include <vector>
#include <list>
#include <map>

#include "../parallel/Parallel.h"
#include "../gtl/design_pattern/singleton_holder.h"
#include "../gtl/utilities/memory_pool.h"
#include "Particle.h"

namespace simpla { namespace particle
{


struct ParticleBase
{
private:
    typedef ParticleBase this_type;
public:
    ParticleBase() { }

    virtual ~ParticleBase() { }

    virtual std::ostream &print(std::ostream &os, int indent) const = 0;

    virtual size_t size() const = 0;

    virtual void deploy() = 0;

    virtual void rehash() = 0;

    virtual void sync() = 0;

    virtual Properties const &properties() const = 0;

    virtual Properties &properties() = 0;

    virtual data_model::DataSet data_set() const = 0;

    virtual void push(Real t0, Real t1) = 0;

    virtual void integral() = 0;

    std::ostream &operator<<(std::ostream &os) const { return this->print(os, 0); }

};

inline std::ostream &operator<<(std::ostream &os, ParticleBase const &p)
{
    return p.print(os, 0);
};

template<typename ...> struct ParticleContainer;
template<typename ...> struct Particle;

template<typename P, typename M, typename ...Policies>
struct Particle<P, M, Policies...> : public ParticleBase, public Policies ...
{
private:
    typedef ParticleContainer<P, M> container_type;
    typedef Particle<P, M, Policies...> this_type;

    typedef typename this_type::interpolate_policy interpolate_policy;

    std::shared_ptr<container_type> m_data_;
public:

    typedef M mesh_type;
    typedef P engine_type;

    typedef typename mesh_type::point_type point_type;
    typedef typename mesh_type::index_tuple index_tuple;
    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::range_type range_type;

    Particle(mesh_type &m, std::string const &s_name) : m_data_(new container_type(m, s_name)) { }

    ~Particle() { }

    Particle(this_type const &other) : m_data_(other.m_data_) { };

    Particle(this_type &&other) : m_data_(other.m_data_) { };

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    void swap(this_type const &other) { std::swap(other.m_data_, m_data_); }

    virtual Properties const &properties() const { return m_data_->properties(); };

    virtual Properties &properties() { return m_data_->properties(); };

    virtual size_t size() const { return m_data_->size(); }

    virtual void deploy() { m_data_->deploy(); }

    virtual void clear() { m_data_->clear(); }

    virtual void sync() { m_data_->sync(); }

    virtual void rehash() { m_data_->rehash(); }

    virtual void push(Real t0, Real t1) { }

    virtual void integral() { }

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return m_data_->print(os, indent); }

    virtual data_model::DataSet data_set() const { return m_data_->data_set(); }


    std::shared_ptr<container_type> data() { return m_data_; }

    std::shared_ptr<container_type> const data() const { return m_data_; }

    std::shared_ptr<typename mesh_type::AttributeEntity> attribute()
    {
        return std::dynamic_pointer_cast<typename mesh_type::AttributeEntity>(m_data_);
    }

    std::shared_ptr<typename mesh_type::AttributeEntity> const attribute() const
    {
        return std::dynamic_pointer_cast<typename mesh_type::AttributeEntity>(m_data_);
    }

    engine_type const &engine() const { return *std::dynamic_pointer_cast<const engine_type>(m_data_); }

    engine_type &engine() { return *std::dynamic_pointer_cast<engine_type>(m_data_); }

    template<typename ...Args>
    void generate(Args &&...args) { m_data_->generate(std::forward<Args>(args)...); }


    template<typename TField>
    void integral(TField *res) const;

    template<typename Pusher> void push(Pusher const &pusher);


};


//*******************************************************************************

template<typename P, typename M, typename ...Policies>
template<typename TField> void
Particle<P, M, Policies...>::integral(id_type const &s, TField *J) const
{
    static constexpr int f_iform = traits::iform<TField>::value;

    auto x0 = m_data_->mesh().point(s);


    id_type neighbours[mesh_type::MAX_NUM_OF_NEIGHBOURS];

    int num = m_data_->mesh().get_adjacent_cells(container_type::iform, s, neighbours);

    for (int i = 0; i < num; ++i)
    {
        typename container_type::const_accessor acc1;

        if (m_data_->find(acc1, neighbours[i]))
        {
            for (auto const &p:acc1->second)
            {
                typename ::simpla::traits::field_value_type<TField>::type v;

                engine_type::integral(x0, p, &v);

                (*J)[s] += interpolate_policy::template sample<f_iform>(m_data_->mesh(), s, v);
            }
        }
    }
};

template<typename P, typename M, typename ...Policies>
template<typename TField> void
Particle<P, M, Policies...>::integral(range_type const &r, TField *J) const
{
    // TODO cache J, Base on r
    for (auto const &s:r) { integral(s, J); }
};

template<typename P, typename M, typename ...Policies>
template<typename TField> void
Particle<P, M, Policies...>::integral(TField *J, Gather const &gather) const
{

    CMD << "integral particle [" << m_data_->name()
    << "] to Field [" << J->attribute()->name() << "<" << J->attribute()->center_type() << ","
    << J->attribute()->extent(0) << ">]" << std::endl;


    static constexpr int f_iform = traits::iform<TField>::value;
    m_data_->mesh().template for_each_boundary<f_iform>([&](range_type const &r) { integral(r, J, integrator); });

    parallel::DistributedObject dist_obj;
    dist_obj.add(*J);
    dist_obj.sync();

    m_data_->mesh().template for_each_center<f_iform>([&](range_type const &r) { integral(r, J, integrator); });

    dist_obj.wait();


}
//*******************************************************************************

template<typename P, typename M, typename ...Policies>
template<typename ...Args> void
Particle<P, M, Policies...>::push(id_type const &s, Args &&...args)
{
    typename container_type::accessor acc;

    if (m_data_->find(acc, s))
    {
        for (auto &p:acc->second)
        {
//            engine_type::push(&p, std::forward<Args>(args)...);
        }
    }


};

template<typename P, typename M, typename ...Policies>
template<typename ...Args> void
Particle<P, M, Policies...>::push(range_type const &r, Args &&...args)
{
    // TODO cache args, Base on s or r
    for (auto const &s:r) { push(s, std::forward<Args>(args)...); }
};

template<typename P, typename M, typename ...Policies>
template<typename ...Args> void
Particle<P, M, Policies...>::push(Args &&...args)
{


    CMD << "Push particle [" << m_data_->name() << "]" << std::endl;

    m_data_->mesh().template for_each_ghost<container_type::iform>(
            [&](range_type const &r) { push(r, std::forward<Args>(args)...); });

    m_data_->mesh().template for_each_boundary<container_type::iform>(
            [&](range_type const &r) { push(r, std::forward<Args>(args)...); });

    m_data_->mesh().template for_each_center<container_type::iform>(
            [&](range_type const &r) { push(r, std::forward<Args>(args)...); });

//    rehash();
}


//**************************************************************************************************

}} //namespace simpla


#endif /* CORE_PARTICLE_PARTICLE_H_ */
