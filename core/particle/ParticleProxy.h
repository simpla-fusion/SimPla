/**
 * @file ParticleProxy.h
 * @author salmon
 * @date 2015-11-26.
 */

#ifndef SIMPLA_PARTICLE_PROXY_H
#define SIMPLA_PARTICLE_PROXY_H

#include "../data_model/DataSet.h"

namespace simpla { namespace particle
{
template<typename...> struct Particle;
template<typename...> struct ParticleProxyBase;
template<typename...> struct ParticleProxy;

template<typename TP, typename TE, typename TB, typename TJ, typename TRho>
class ParticleProxy<TP, TE, TB, TJ, TRho> : public ParticleProxyBase<TE, TB, TJ, TRho>
{
    typedef ParticleProxy<TP, TE, TB, TJ, TRho> this_type;
public:
    typedef TP particle_type;

    std::shared_ptr<particle_type> m_self_;

    ParticleProxy(std::shared_ptr<particle_type> p) : m_self_(p) { }

    ParticleProxy(this_type const &other) : m_self_(other.m_self_) { }

    virtual  ~ParticleProxy() { }

    particle_type &self() { return *m_self_; }

    particle_type const &self() const { return *m_self_; }

    virtual Properties const &properties() const { return m_self_->properties(); }

    virtual Properties &properties() { return m_self_->properties(); }

    virtual void deploy() { m_self_->deploy(); }

    virtual void rehash() { m_self_->rehash(); }

    virtual void sync() { m_self_->sync(); }

    virtual data_model::DataSet data_set() const { return m_self_->data_set(); }

    virtual void push(Real dt, Real t, TE const &E, TB const &B) { m_self_->push(dt, t, E, B); }

    virtual void integral(TJ *J) const { m_self_->integral(J); }

    virtual void integral(TRho *n) const { m_self_->integral(n); }

    std::ostream &print(std::ostream &os, int indent) const { return m_self_->print(os, indent); }

};

template<typename TE, typename TB, typename TJ, typename TRho>
struct ParticleProxyBase<TE, TB, TJ, TRho>
{
private:
    typedef ParticleProxyBase<TE, TB, TJ, TRho> this_type;
public:
    ParticleProxyBase() { }

    virtual ~ParticleProxyBase() { }

    virtual void deploy() = 0;

    virtual void rehash() = 0;

    virtual void sync() = 0;

    virtual Properties const &properties() const = 0;

    virtual Properties &properties() = 0;

    virtual data_model::DataSet data_set() const = 0;

    virtual void push(Real dt, Real t, TE const &E, TB const &B) = 0;

    virtual void integral(TJ *J) const = 0;

    virtual void integral(TRho *n) const = 0;

    virtual std::ostream &print(std::ostream &os, int indent) const = 0;

    template<typename TP>
    static std::shared_ptr<this_type> create(std::shared_ptr<TP> p)
    {
        return std::dynamic_pointer_cast<this_type>(
                std::make_shared<ParticleProxy<TP, TE, TB, TJ, TRho>>(p));

    };
};

template<typename TE, typename TB, typename TJ, typename TRho>
std::ostream &operator<<(std::ostream &os, ParticleProxyBase<TE, TB, TJ, TRho> const &p)
{
    return p.print(os, 0);
};
}}//namespace simpla{namespace particle

#endif //SIMPLA_PARTICLE_PROXY_H
