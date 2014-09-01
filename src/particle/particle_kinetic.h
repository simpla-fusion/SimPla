/**
 * \file particle_kinetic.h
 *
 * \date    2014年9月1日  下午2:25:26 
 * \author salmon
 */

#ifndef PARTICLE_KINETIC_H_
#define PARTICLE_KINETIC_H_

#include <iostream>
#include <string>
#include <type_traits>

#include "../utilities/ntuple.h"
#include "../utilities/primitives.h"
#include "../utilities/container_pool.h"

namespace simpla
{

template<typename TM, typename Engine>
class KineticParticle: public Engine
{
public:
	static constexpr unsigned int IForm = VERTEX;

	typedef TM mesh_type;
	typedef Engine engine_type;
	typedef KineticParticle<mesh_type, engine_type> this_type;
private:

	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type::template field<VERTEX, scalar_type> rho_type;

	HAS_MEMBER(J_at_the_center);
	typedef typename std::conditional<has_member_J_at_the_center<engine_type>::value,
	typename mesh_type::template field<VERTEX, nTuple<3, scalar_type>>,
	typename mesh_type::template field<EDGE, scalar_type> >::type J_type;

	typedef typename engine_type::Point_s particle_type;
	typedef typename mesh_type::compact_index_type mid_type; // id of mesh point

	typedef ContainerPool<mid_type, particle_type> storage_type;

	storage_type pic_;
public:
	mesh_type const & mesh;
	rho_type rho;
	J_type J;

	template<typename ...Others>
	KineticParticle(mesh_type const & pmesh, Others && ...);// Constructor

	~KineticParticle();// Destructor

	void load();

	template<typename TDict, typename ...Others>
	void load(TDict const & dict, Others && ...others);

	std::string save(std::string const & path) const;

	std::ostream& print(std::ostream & os) const;

	template<typename ...Args> void next_timestep(Args && ...args);

private:

	template<typename ... Args>
	struct next_timestep_rebind
	{
	private:
		HAS_MEMBER_FUNCTION(next_timestep);
	public:
		static constexpr int value =

		((has_member_function_next_timestep<engine_type, particle_type*, rho_type *,Real,Args...>::value)?1:0) +

		((has_member_function_next_timestep<engine_type,particle_type*, J_type *,Real,Args...>::value)?2:0 )

		;
	};

	template<typename TRange>
	void sort(TRange && range)
	{
		pic_.sort(std::forward<TRange>(range), [&mesh](particle_type const & p)->mid_type
				{
					return std::get<0>(mesh.CoordinatesGlobalToLocal(std::get<0>(engine_type::pull_back(p))));
				});
	}

public:
	template<typename ...Args>
	typename std::enable_if<next_timestep_rebind<Args...>::value==0,void>::value
	next_timestep (Real dt, Args && ...args)
	{

		LOGGER << "Push particles to  next step [ " << engine_type::get_type_as_string() << " ]";

		auto range = mesh.select(IForm);

		sort(range);

		pic_.modify(range, [this](particle_type & p)
				{
					this->engine_type::next_timestep(&p,dt, std::forward<Args>(args)...);
				});

		sort(range);

		if (engine_type::properties["ScatterJ"].template as<bool>(true))
		{
			J.clear();

			pic_.reduce(range, J, &J, [](particle_type const & ,J_type * t_J)
					{
						this->engine_type::scatterJ(&p, t_J);
					});

			update_ghosts(&J);
		}

		if (engine_type::properties["ScatterRho"].template as<bool>(false))
		{
			rho.clear();

			pic_.reduce(range, rho, &rho, [](particle_type const & ,rho_type * t_rho)
					{
						this->engine_type::scatterRho(&p, t_rho);
					});

			update_ghosts(&rho);
		}
		else if (properties["ContinuityEquation"].template as<bool>(false))
		{
			rho -= Diverge(MapTo<EDGE>(J)) * dt;
		}

	}

	template<typename ...Args>
	typename std::enable_if<next_timestep_rebind<Args...>::value==1,void>::value
	next_timestep (Real dt, Args && ...args)
	{
		auto range = mesh.select(IForm);

		sort(range);

		rho.clear();

		pic_.reduce(range, rho, &rho, [&](particle_type const & ,rho_type * t_rho)
				{
					this->engine_type::next_timestep(&p, t_rho,dt,std::forward<Args>(args)...);
				});

		update_ghosts(&rho);

	}

	template<typename ...Args>
	typename std::enable_if<next_timestep_rebind<Args...>::value==2,void>::value
	next_timestep ( Real dt, Args && ...args)
	{
		auto range = mesh.select(IForm);

		sort(range);

		J.clear();

		pic_.reduce(range, J, &J, [&](particle_type const & ,J_typr * t_J)
				{
					this->engine_type::next_timestep(&p, t_J,dt,std::forward<Args>(args)...);
				});

		update_ghosts(&J);

		if (properties["ScatterRho"].template as<bool>(false))
		{
			rho.clear();

			pic_.reduce(range, rho, &rho, [](particle_type const & ,rho_type * t_rho)
					{
						this->engine_type::scatterRho(&p, t_rho);
					});

			update_ghosts(&rho);
		}

	}

};

template<typename TM, typename Engine>
template<typename ... Others>
KineticParticle<TM, Engine>::KineticParticle(mesh_type const & pmesh, Others && ...others) :
		mesh(pmesh), rho(mesh), J(mesh)
{
	load(std::forward<Others>(others)...);
}

template<typename TM, typename Engine>
KineticParticle<TM, Engine>::~KineticParticle()
{
}
template<typename TM, typename Engine>
template<typename TDict, typename ...Others>
void KineticParticle<TM, Engine>::load(TDict const & dict, Others && ...others)
{
	engine_type::load(dict, std::forward<Others>(others)...);

	pic_.load(dict, std::forward<Others>(others)...);

	properties.set("DumpKineticParticle", dict["DumpKineticParticle"].template as<bool>(false));

	properties.set("ContinuityEquation", dict["ContinuityEquation"].template as<bool>(false));

	properties.set("ScatterRho", dict["ScatterRho"].template as<bool>(false));

}

//*************************************************************************************************

template<typename TM, typename Engine>
std::string KineticParticle<TM, Engine>::save(std::string const & path) const
{
	std::stringstream os;

	GLOBAL_DATA_STREAM.cd(path);

	os << "\n, n =" << simpla::save("rho", rho);

	os << "\n, J_ =" << simpla::save("J_", J);

	if (properties["DumpKineticParticle"].template as<bool>(false))
	{
		os << "\n, particles = " << pic_.save("particles");
	}

	return os.str();
}
template<typename TM, typename Engine>
std::ostream& KineticParticle<TM, Engine>::print(std::ostream & os) const
{
	engine_type::print(os);
	properties.print(os);
}

}  // namespace simpla

#endif /* PARTICLE_KINETIC_H_ */
