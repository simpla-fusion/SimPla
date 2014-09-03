/**
 * \file kinetic_particle.h
 *
 * \date    2014年9月1日  下午2:25:26 
 * \author salmon
 */

#ifndef KINETIC_PARTICLE_H_
#define KINETIC_PARTICLE_H_

#include <iostream>
#include <string>
#include <type_traits>

#include "../utilities/ntuple.h"
#include "../utilities/primitives.h"
#include "../utilities/container_pool.h"
#include "../utilities/properties.h"

namespace simpla
{
class PolicyKineticParticle;
template<typename TM, typename Engine, typename Policy> class Particle;

template<typename TM, typename Engine> using KineticParticle=Particle<TM, Engine, PolicyKineticParticle>;

template<typename TM, typename Engine>
class Particle<TM, Engine, PolicyKineticParticle> : public Engine, public ContainerPool<typename Engine::Point_s>
{
public:
	static constexpr unsigned int IForm = VERTEX;

	typedef TM mesh_type;
	typedef Engine engine_type;
	typedef Particle<mesh_type, engine_type, PolicyKineticParticle> this_type;
private:

	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type::template field<VERTEX, scalar_type> rho_type;

	HAS_MEMBER(J_at_the_center)

typedef	typename std::conditional<has_member_J_at_the_center<engine_type>::value,
	typename mesh_type::template field<VERTEX, nTuple<3, scalar_type>>,
	typename mesh_type::template field<EDGE, scalar_type> >::type J_type;

	typedef typename engine_type::Point_s particle_type;
	typedef typename mesh_type::compact_index_type mid_type; // id of mesh point

	typedef ContainerPool<mid_type, particle_type> storage_type;

	storage_type pic_;

	std::function<compact_index_type(particle_type const &)> hash_fun_;
public:
	mesh_type const & mesh;

	Properties properties;
	rho_type rho;
	J_type J;

	template<typename ...Others>
	Particle(mesh_type const & pmesh, Others && ...);// Constructor

	~Particle();// Destructor

	static std::string get_type_as_string()
	{
		return "Kinetic"+engine_type::get_type_as_string();
	}

	void load()
	{};

	template<typename TDict, typename ...Others>
	void load(TDict const & dict, Others && ...others);

	std::string save(std::string const & path) const;

	std::ostream& print(std::ostream & os) const;

	template<typename ...Args> void next_timestep(Args && ...args);

	template<typename TJ> void ScatterJ(TJ * pJ) const
	{
		*pJ+=MapTo<TJ>(J);
	}

	template<typename TRho> void ScatterRho(TRho * prho) const
	{
		*prho+=MapTo<TRho>(rho);
	}

private:

	template<typename ... Args>
	struct next_timestep_rebind
	{
	private:
		HAS_MEMBER_FUNCTION(next_timestep);
	public:
		static constexpr int value =

		((has_member_function_next_timestep<engine_type, particle_type*, rho_type *,Real,Args...>::value)?1:0) +

		((has_member_function_next_timestep<engine_type, particle_type*, J_type *,Real,Args...>::value)?2:0 )

		;
	};

public:
	template<typename ...Args>
	void next_timestep(Real dt, Args && ...args);

	template<typename ...Args>
	void next_timestep(Real dt, J_type * pJ, Args && ...args);

	template<typename ...Args>
	void next_timestep(Real dt, rho_type * pJ, Args && ...args);

	template<typename TRange,typename Fun >
	void modify(TRange const & range,Fun const& );

};

template<typename TM, typename Engine>
template<typename ... Others>
Particle<TM, Engine, PolicyKineticParticle>::Particle(mesh_type const & pmesh, Others && ...others)
		: mesh(pmesh), rho(mesh), J(mesh)
{
	load(std::forward<Others>(others)...);
	hash_fun_ = [& ](particle_type const & p)->mid_type
	{
		return std::get<0>(mesh.coordinates_global_to_local(std::get<0>(engine_type::pull_back(p))));
	};
}

template<typename TM, typename Engine>
Particle<TM, Engine, PolicyKineticParticle>::~Particle()
{
}
template<typename TM, typename Engine>
template<typename TDict, typename ...Others>
void Particle<TM, Engine, PolicyKineticParticle>::load(TDict const & dict, Others && ...others)
{
	engine_type::load(dict, std::forward<Others>(others)...);

	pic_.load(dict, std::forward<Others>(others)...);

	properties.set("DumpParticle", dict["DumpParticle"].template as<bool>(false));

	properties.set("ContinuityEquation", dict["ContinuityEquation"].template as<bool>(false));

	properties.set("ScatterRho", dict["ScatterRho"].template as<bool>(false));

}
template<typename TM, typename Engine>
std::string Particle<TM, Engine, PolicyKineticParticle>::save(std::string const & path) const
{
	std::stringstream os;

	GLOBAL_DATA_STREAM.cd(path);

	os << "\n, n =" << simpla::save("rho", rho);

	os << "\n, J_ =" << simpla::save("J_", J);

//	if (properties["DumpParticle"].template as<bool>(false))
//	{
//		os << "\n, particles = " << save(pic_, "particles");
//	}

	return os.str();
}
template<typename TM, typename Engine>
std::ostream& Particle<TM, Engine, PolicyKineticParticle>::print(std::ostream & os) const
{
	engine_type::print(os);
	properties.print(os);
}
template<typename TM, typename Engine>
template<typename TF>
void Particle<TM, Engine, PolicyKineticParticle>::scatter(TF * pf)
{
	pic_.reduce(mesh.select(IForm), *pf, pf, [this](particle_type const & p,TF * t_f)
	{
		scatter_cartesian(t_f,this->engine_type::pull_back( p ));
	});

	update_ghosts(pf);

}
template<typename TM, typename Engine>
template<typename ...Args>
void Particle<TM, Engine, PolicyKineticParticle>::next_timestep(Real dt, Args && ...args)
{

	LOGGER << "Push particles to  next step [ " << engine_type::get_type_as_string() << " ]";

	auto range = mesh.select(IForm);

	pic_.sort(hash_fun_);

	pic_.modify(range, [&](particle_type & p)
	{
		this->engine_type::next_timestep(&p,dt, std::forward<Args>(args)...);
	});

	pic_.sort(hash_fun_);

	if (engine_type::properties["ScatterJ"].template as<bool>(true))
	{
		J.clear();
		scatter(&J);
	}

	if (engine_type::properties["ScatterRho"].template as<bool>(false))
	{
		rho.clear();
		scatter(&rho);
	}
	else if (properties["ContinuityEquation"].template as<bool>(false))
	{
		rho -= Diverge(MapTo<EDGE>(J)) * dt;
	}

}
template<typename TM, typename Engine>
template<typename ...Args>
void Particle<TM, Engine, PolicyKineticParticle>::next_timestep(Real dt, J_type * pJ, Args && ...args)
{
	auto range = mesh.select(IForm);

	pic_.sort(hash_fun_);

	if (properties["ContinuityEquation"].template as<bool>(false))
	{
		rho -= Diverge(MapTo<EDGE>(J)) * dt * 0.5;
	}

	J.clear();

	pic_.reduce(range, &J, J, [&](particle_type & p,J_type * t_J)
	{
		this->engine_type::next_timestep(&p, dt,t_J,std::forward<Args>(args)...);
	});

	update_ghosts(&J);
	if (properties["ContinuityEquation"].template as<bool>(false))
	{
		rho -= Diverge(MapTo<EDGE>(J)) * dt * 0.5;
	}

}
template<typename TM, typename Engine>
template<typename ...Args>
void Particle<TM, Engine, PolicyKineticParticle>::next_timestep(Real dt, rho_type * pJ, Args && ...args)
{
	auto range = mesh.select(IForm);

	pic_.sort(hash_fun_);

	rho.clear();

	pic_.reduce(range, rho, &rho, [&](particle_type & p ,rho_type * t_rho)
	{
		this->engine_type::next_timestep(&p, dt,t_rho,std::forward<Args>(args)...);
	});

	update_ghosts(&rho);

}
template<typename TM, typename Engine>
template<typename TRange, typename Fun>
void Particle<TM, Engine, PolicyKineticParticle>::modify(TRange const & range, Fun const& fun)
{
	auto buffer = pic_.get_child();
	pic_.reduce(std::forward<TRange>(range), buffer, &buffer, fun);
	pic_.add(buffer, hash_fun_);
}

}  // namespace simpla

#endif /* KINETIC_PARTICLE_H_ */
