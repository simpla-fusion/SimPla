/*
 * save_particle.h
 *
 *  created on: 2013-12-21
 *      Author: salmon
 */

#ifndef SAVE_PARTICLE_H_
#define SAVE_PARTICLE_H_

#include <string>
#include <vector>

#include "../io/data_stream.h"

namespace simpla
{

template<typename, typename > class ParticlePool;

template<typename TM, typename TPoints> inline std::string //
save(std::string const & name, ParticlePool<TM, TPoints> const & d)
{
	std::vector<TPoints> res;

	for (auto const & l : d)
	{
		std::copy(l.second.begin(), l.second.end(), std::back_inserter(res));
	}

	return GLOBAL_DATA_STREAM.write(name,&res[0],res.size());
}

}
 // namespace simpla

#endif /* SAVE_PARTICLE_H_ */
