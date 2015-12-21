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
typedef particle::ParticleProxyBase<TE, TB, TJv, TRho> particle_proxy_type;

template<typename TE, typename TB, typename TJ, typename TRho>
struct ParticleFactory
{
    typedef ParticleProxyBase<TE, TB, TJ, TRho> base_type;

    template<typename TDict>
    std::shared_ptr<base_type> create(TDict const &dict);
};

template<typename TE, typename TB, typename TJ, typename TRho>
struct ParticleFactory

template<typename TDict>
std::shared_ptr<base_type> ParticleFactorytemplate<TE, TB, TJ, TRho>::create(std::string const key, TDict const &dict)
{
    std::shared_ptr<base_type> res;

    if (dict.second["Type"].template as<std::string>() == "Boris")
    {
        particle::BorisParticle<mesh_type> pic(m, key);

        pic.engine().mass(mass);
        pic.engine().charge(charge);

        pic.properties()["DisableCheckPoint"] =
                dict.second["DisableCheckPoint"].template as<bool>(true);

        auto gen = particle::make_generator(pic.engine(), 1.0);

        pic.generator(gen, options["PIC"].as<size_t>(10), 1.0);

        res = base_type::create(pic.data());

    }
};
}}
#endif //SIMPLA_PARTICLEFACTORY_H
