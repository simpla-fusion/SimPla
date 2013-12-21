/*
 * save_particle.h
 *
 *  Created on: 2013年12月21日
 *      Author: salmon
 */

#ifndef SAVE_PARTICLE_H_
#define SAVE_PARTICLE_H_

namespace simpla
{

template<typename > class Particle;
template<typename TEngine, typename TOther> inline DataSet<typename TEngine::Point_s>

Data(Particle<TEngine> const & d, std::string const & name, Other const &, bool flag)
{
	return std::move(DataSet<TV>(d.data(), name, d.GetShape(), flag));
}

}  // namespace simpla

#endif /* SAVE_PARTICLE_H_ */
