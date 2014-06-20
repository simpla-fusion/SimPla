/*
 * particle_factory.h
 *
 *  Created on: 2014年6月13日
 *      Author: salmon
 */

#ifndef PARTICLE_FACTORY_H_
#define PARTICLE_FACTORY_H_

#include <string>

#include "../utilities/factory.h"
#include "particle_base.h"

namespace simpla
{
template<typename TM, typename ...Args>
struct ParticleFactory: public Factory<std::string, std::shared_ptr<ParticleBase<TM>>, TM const &, Args && ...>
{
	typedef Factory<std::string, std::shared_ptr<ParticleBase<TM>>, TM const &, Args && ...> base_type;
	typedef ParticleFactory<TM, Args ...> this_type;
	typedef typename base_type::product_type product_type;
	typedef typename base_type::create_fun_callback create_fun_callback;
	ParticleFactory()
	{
	}

	~ParticleFactory()
	{
	}

};

template<typename TM, typename ...Args>
bool RegisterParticle(std::string const & name,
        std::function<std::shared_ptr<ParticleBase<TM>>(TM const &, Args && ...)> callback)
{
	return SingletonHolder<ParticleFactory<TM, Args...>>::instance().Register(name, callback);
}

template<typename TM, typename ...Args>
std::shared_ptr<ParticleBase<TM>> CreateParticle(std::string const & name, TM const & m, Args && ... args)
{
	return SingletonHolder<ParticleFactory<TM, Args...>>::instance().Create(name, m, std::forward<Args >(args)...);
}

}  // namespace simpla

#endif /* PARTICLE_FACTORY_H_ */
