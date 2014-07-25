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
save(std::string const & name, ParticlePool<TM, TPoints> const & pool)
{
	std::vector<TPoints> res;
	res.clear();
	res.reserve(pool.size());

	for (auto const &p : pool)
	{
		std::copy(p.second.begin(), p.second.end(), std::back_inserter(res));
	}
	auto s = res.size();

	return GLOBAL_DATA_STREAM.write(name,&res[0],DataType::create<TPoints>(), 1,nullptr,&s,nullptr,nullptr,nullptr,nullptr,DataStream::SP_UNORDER );

}
}
 // namespace simpla

#endif /* SAVE_PARTICLE_H_ */
