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

    virtual void integral() const = 0;

    virtual void load_filter(std::string const &key = "") = 0;

    std::ostream &operator<<(std::ostream &os) const { return this->print(os, 0); }

};

inline std::ostream &operator<<(std::ostream &os, ParticleBase const &p)
{
    return p.print(os, 0);
};

template<typename ...> struct ParticleContainer;
template<typename ...> struct Particle;

template<typename P, typename M>
struct Particle<P, M> : public ParticleBase
{
private:
    typedef ParticleContainer<P, M> container_type;
    typedef Particle<P, M> this_type;
    std::shared_ptr<container_type> m_data_;
public:

    typedef M mesh_type;
    typedef P engine_type;
    typedef typename engine_type::sample_type sample_type;
    typedef typename mesh_type::point_type point_type;
    typedef typename mesh_type::index_tuple index_tuple;
    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::range_type range_type;

    Particle(mesh_type &m, std::string const &s_name) : m_data_(new container_type(m, s_name))
    {
        if (s_name != "") { m.enroll(s_name, m_data_); }
    }

    ~Particle() { }

    Particle(this_type const &other) : m_data_(other.m_data_) { };

    Particle(this_type &&other) : m_data_(other.m_data_) { };

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    void swap(this_type const &other) { std::swap(other.m_data_, m_data_); }

    engine_type const &engine() const { return *std::dynamic_pointer_cast<const engine_type>(m_data_); }

    engine_type &engine() { return *std::dynamic_pointer_cast<engine_type>(m_data_); }

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

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return m_data_->print(os, indent); }

    virtual data_model::DataSet data_set() const { return m_data_->data_set(); }

    virtual Properties const &properties() const { return m_data_->properties(); };

    virtual Properties &properties() { return m_data_->properties(); };

    virtual size_t size() const { return m_data_->count(); }

    virtual void deploy() { m_data_->deploy(); }

    virtual void clear() { m_data_->clear(); }

    virtual void sync() { m_data_->sync(); }

    virtual void rehash() { m_data_->rehash(); }

    virtual void push(Real t0, Real t1)
    {
        VERBOSE << "[CMD] Push particle " << m_data_->name() << std::endl;

        /* m_data_->filter(engine_type::pusher(t0, t1));  */

        //  m_data_->rehash();
    }

    virtual void integral() const
    {
        VERBOSE << "[CMD] Integral particle " << m_data_->name() << std::endl;
        /*m_data_->(engine_type::gather()); */}

    template<typename ...Args>
    void generate(Args &&...args) { m_data_->generate(std::forward<Args>(args)...); }

    typedef std::function<void(sample_type *)> filter_fun;

    template<typename ...Args>
    void filter(Args &&...args) { m_data_->filter(std::forward<Args>(args)...); }


    template<typename TFun>
    void filter(std::tuple<TFun, range_type> const &f) { m_data_->filter(std::get<0>(f), std::get<1>(f)); }

    virtual void load_filter(std::string const &key = "")
    {
        if (key == "")
        {
            for (auto const &item:m_filter_list_) { filter(item.second); }
        }
        else
        {
            auto it = m_filter_list_.find(key);
            if (it != m_filter_list_.end()) { filter(it->second); }
        }
    }


    bool register_filter(std::string const &key, filter_fun const &f, range_type const &r)
    {
        return std::get<1>(m_filter_list_.insert(std::make_pair(key, std::make_tuple(f, r))));
    }

private:
    std::map<std::string, std::tuple<filter_fun, range_type >> m_filter_list_;
};


}} //namespace simpla


#endif /* CORE_PARTICLE_PARTICLE_H_ */
