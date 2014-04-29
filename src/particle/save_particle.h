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
template<typename, typename > class Particle;

template<typename TEngine, typename TStorage> inline std::string //
Dump(Particle<TEngine, TStorage> const & d, std::string const & name, bool is_compact_store = false)
{
	std::vector<typename Particle<TEngine, TStorage>::value_type> res;

	for (auto const & l : d.GetTree())
	{
		std::copy(l.begin(), l.end(), std::back_inserter(res));
	}

	return Dump(res, name, is_compact_store);

}

}  // namespace simpla

#endif /* SAVE_PARTICLE_H_ */
