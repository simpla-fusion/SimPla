/*
 * save_particle.h
 *
 *  Created on: 2013年12月21日
 *      Author: salmon
 */

#ifndef SAVE_PARTICLE_H_
#define SAVE_PARTICLE_H_

#include <string>
#include <vector>

#include "../io/data_stream.h"

namespace simpla
{

template<typename > class DataDumper;
template<typename, typename > class ParticlePool;

template<typename TM, typename TPoints> inline std::string //
Dump(ParticlePool<TM, TPoints> const & d, std::string const & name)
{
	std::vector<TPoints> res;

	for (auto const & l : d.GetTree())
	{
		std::copy(l.begin(), l.end(), std::back_inserter(res));
	}

	return Dump(res, name);

}

}  // namespace simpla

#endif /* SAVE_PARTICLE_H_ */
