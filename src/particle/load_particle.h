/*
 * load_particle.h
 *
 *  Created on: 2013年12月21日
 *      Author: salmon
 */

#ifndef LOAD_PARTICLE_H_
#define LOAD_PARTICLE_H_

namespace simpla
{
template<typename > class Particle;
template<typename TEngine>
bool LoadParticle(LuaObject const &obj, Particle<TEngine> *p)
{
	if (obj.empty())
		return false;

	typedef TEngine engine_type;

	typedef typename engine_type::mesh_type mesh_type;

	typedef Particle<engine_type> this_type;

	typedef typename engine_type::Point_s particle_type;

	typedef particle_type value_type;

	typedef typename mesh_type::scalar scalar;

	mesh_type const &mesh = p->mesh;





	return true;
}
}  // namespace simpla

#endif /* LOAD_PARTICLE_H_ */
