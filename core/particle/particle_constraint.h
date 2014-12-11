/**
 * \file particle_constraint.h
 *
 * \date    2014年9月2日  上午10:39:25 
 * \author salmon
 */

#ifndef PARTICLE_CONSTRAINT_H_
#define PARTICLE_CONSTRAINT_H_

#include <cstddef>
#include <vector>

namespace simpla
{
class PolicyProbeParticle;
class PolicyKineticParticle;
template<typename TM, typename Engine, typename Policy> class Particle;

class PolicyReflectSurface;

template<typename TM, typename Engine, typename Policy = PolicyReflectSurface>
class ParticleConstraint
{
public:
	typedef TM mesh_type;
	typedef Engine engine_type;
	typedef Policy policy_type;
	typedef typename engine_type::Poiont_s Poiont_s;
	typedef ParticleConstraint<mesh_type, engine_type, policy_type> this_type;

private:
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type::index_type index_type;

	static constexpr unsigned int NDIMS = mesh_type::NDIMS;

	typedef Vec3 normal_polar_vec_type;

	std::map<index_type, normal_polar_vec_type> surface_;

	typedef std::tuple<coordinates_type, Vec3, scalar_type> c_particle_type;

	std::function<void(c_particle_type, Vec3 const &)> op_;

public:
	mesh_type const & mesh;
	std::vector<index_type> range;

	template<typename Func>
	ParticleConstraint(mesh_type const & pm, engine_type const & pe, Func const & fun)
			: mesh(pm), op_(fun)
	{

	}
	~ParticleConstraint()
	{
	}
	/**
	 *
	 * @param p
	 * @return true if particle is changed
	 */
	bool operator()(Poiont_s *p) const
	{
		c_particle_type tp = engine_type::pull_back(*p);
		auto git = std::get<0>(mesh.coordiantes_global_to_local(std::get<0>(tp)));
		op_(&tp, surface_[git]);
		*p = engine_type::push_forward(tp);
		return true;
	}

}
;
}  // namespace simpla

#endif /* PARTICLE_CONSTRAINT_H_ */
