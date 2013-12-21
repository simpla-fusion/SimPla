/*
 * save_particle.h
 *
 *  Created on: 2013年12月21日
 *      Author: salmon
 */

#ifndef SAVE_PARTICLE_H_
#define SAVE_PARTICLE_H_

#include <memory>
#include "../utilities/log.h"
namespace simpla
{

template<typename > class DataSet;
template<typename > class Particle;

template<typename TEngine> inline DataSet<typename TEngine::Point_s> //
Data(Particle<TEngine> const & d, std::string const & name, bool is_compact_store = false)
{
	size_t s = d.size();
	return std::move(
	        DataSet<typename TEngine::Point_s>(std::shared_ptr<typename TEngine::Point_s>(nullptr), name, 1, &s,
	                is_compact_store));
	UNIMPLEMENT;
}

}  // namespace simpla

#endif /* SAVE_PARTICLE_H_ */
