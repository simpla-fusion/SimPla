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
#include "ParticleContainer.h"

namespace simpla { namespace particle
{
template<typename ...> struct Particle;
template<typename ...> struct ParticleEngine;
template<typename TAGS, typename M> using particle_t= Particle<ParticleEngine<TAGS>, M>;

template<typename P, typename M, typename ...Policies>
struct Particle<P, M, Policies...>
{
private:
    typedef ParticleContainer<P, M, Policies...> container_type;
    typedef Particle<P, M, Policies...> this_type;

    std::shared_ptr<container_type> m_data_;
public:

    typedef M mesh_type;
    typedef P engine_type;

    template<typename ...Args>
    Particle(mesh_type &m, std::string const &s_name, Args &&...args)
            : m_data_(new container_type(m, s_name, std::forward<Args>(args)...))
    {
        if (s_name != "") { m.enroll(s_name, m_data_); }
    };

    template<typename ...Args>
    Particle(mesh_type const &m, std::string const &s_name, Args &&...args)
            : m_data_(new container_type(m, s_name, std::forward<Args>(args)...))
    {
    };

    Particle(this_type const &other) : m_data_(other.m_data_) { };

    Particle(this_type &&other) : m_data_(other.m_data_) { };

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    void swap(this_type const &other) { std::swap(other.m_data_, m_data_); }

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

    template<typename ...Args>
    size_t size(Args &&...args) const { return m_data_->size(std::forward<Args>(args)...); }

    void deploy() { m_data_->deploy(); }

    void clear() { m_data_->clear(); }

    void sync() { m_data_->sync(); }

    void update() { m_data_->update(); }

    std::ostream &print(std::ostream &os, int indent = 0) const { m_data_->print(os, indent); }

    template<typename TDict> void load(TDict const &dict) { m_data_->load(dict); }


    engine_type const &engine() const { return *std::dynamic_pointer_cast<const engine_type>(m_data_); }

    engine_type &engine() { return *std::dynamic_pointer_cast<engine_type>(m_data_); }

    template<typename ...Args>
    void generator(Args &&...args) { m_data_->generator(std::forward<Args>(args)...); }


    template<typename ...Args>
    void push(Args &&...args) { m_data_->push(std::forward<Args>(args)...); }

    template<typename ...Args>
    void integral(Args &&...args) const { m_data_->integral(std::forward<Args>(args)...); }

    Properties &properties() { return m_data_->properties(); }

    Properties const &properties() const { return m_data_->properties(); }


};
}} //namespace simpla


#endif /* CORE_PARTICLE_PARTICLE_H_ */
