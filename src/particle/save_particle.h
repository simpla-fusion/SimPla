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

template<typename, typename > class ParticlePool;

template<typename TM, typename TPoints> inline std::string //
Save(std::string const & name, ParticlePool<TM, TPoints> const & d)
{
	//	std::vector<TPoints> res;
	//
	//	for (auto const & l : d.data())
	//	{
	//		std::copy(l.second.begin(), l.second.end(), std::back_inserter(res));
	//	}
	//
	//	return Save(name, res);
	///@FIXME  Particle Save is not implemented!!
	UNIMPLEMENT;
	return "";
}

}  // namespace simpla

#endif /* SAVE_PARTICLE_H_ */
