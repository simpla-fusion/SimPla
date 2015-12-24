/**
 * @file ParticleFactory.h
 * @author salmon
 * @date 2015-12-21.
 */

#ifndef SIMPLA_PARTICLEFACTORY_H
#define SIMPLA_PARTICLEFACTORY_H

#include "ParticleProxy.h"

namespace simpla { namespace particle
{
template<typename TP, typename ...Args, typename TM, typename TDict>
std::shared_ptr<particle::ParticleProxyBase<Args...>>
create_particle(std::string const &key, TDict const &dict)
{
    typedef particle::ParticleProxyBase<Args...> particle_proxy_type;
    TP pic(m, key);

    dict.as(&pic.properties());

    pic.deploy();

    auto gen = particle::make_generator(pic.engine(), 1.0);

    pic.generator(plasma_region_volume, gen, pic.properties()["PIC"].template as<size_t>(10),
                  pic.properties()["T"].template as<Real>(1));


    return particle_proxy_type::create(pic.data());

}

template<typename ...Args, typename TMesh, typename TDict>
std::shared_ptr<particle::ParticleProxyBase<Args...>>
create(TMesh &mesh, TDict const &dict)
{
    create_particle(std::string const &key, TDict const &dict)

};
}}
#endif //SIMPLA_PARTICLEFACTORY_H
