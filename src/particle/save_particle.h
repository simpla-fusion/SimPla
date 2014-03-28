/*
 * save_particle.h
 *
 *  Created on: 2013年12月21日
 *      Author: salmon
 */

#ifndef SAVE_PARTICLE_H_
#define SAVE_PARTICLE_H_

#include <string>

#include "../io/data_stream.h"
#include "particle.h"

namespace simpla
{

template<typename > class DataDumper;
template<typename > class Particle;

template<typename TEngine> inline std::string //
Dump(Particle<TEngine> const & d, std::string const & name, bool is_compact_store = false)
{
	auto t = d.DumpData();

	return DataDumper<typename TEngine::Point_s>(t.first.get(), name, 1, &(t.second), is_compact_store).GetName();

}

}  // namespace simpla

#endif /* SAVE_PARTICLE_H_ */
