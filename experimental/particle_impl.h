/*
 * particle_impl.h
 *
 *  Created on: 2014年8月29日
 *      Author: salmon
 */

#ifndef PARTICLE_IMPL_H_
#define PARTICLE_IMPL_H_

#include <string>
#include <utility>

#include "../data_interface/container_pool.h"
#include "../utilities/log.h"

namespace simpla
{

/** \defgroup  Particle Particle
 *
 *   \page Particle Conecpt Particle
 *
 *   \code
 * template<typename TM, typename Engine>
 * class Particle
 * {
 *
 *	 public:
 *	 static constexpr unsigned int IForm = VERTEX;
 *
 *	 typedef TM mesh_type;
 *
 *	 typedef Particle<mesh_type, engine_type> this_type;
 *
 *	 mesh_type const & mesh;
 *
 *	 Properties properties;
 *
 *	 template<typename ...Others>
 *	 Particle(mesh_type const & pmesh, Others && ...); 	// Constructor
 *
 *	 ~Particle();	// Destructor
 *
 *	 void load();
 *
 *	 template<typename TDict, typename ...Others>
 *	 void load(TDict const & dict, Others && ...others);
 *
 *	 std::string save(std::string const & path) const;
 *
 *	 std::ostream& print(std::ostream & os) const;
 *
 *	 template<typename ...Args> void next_timestep(Args && ...);
 *
 *	 template<typename TJ> void ScatterJ(TJ * J) const;
 *
 *	 template<typename TJ> void ScatterRho(TJ * rho) const;
 *
 * };
 *
 *   \endcode
 *
 */

/**
 *  \brief Particle class
 *
 */
template<typename ...>class Particle;

template<typename TM, typename Engine>
class Particle<TM, Engine>
{

public:
	static constexpr unsigned int IForm = VERTEX;

	typedef TM mesh_type;
	typedef Engine engine_type;
	typedef Particle<mesh_type, engine_type> this_type;

	mesh_type const & mesh;

	template<typename ...Others>
	Particle(mesh_type const & pmesh, Others && ...); 	// Constructor

	~Particle();	// Destructor

	void load();

	template<typename TDict, typename ...Others>
	void load(TDict const & dict, Others && ...others);

	std::string save(std::string const & path) const;

	std::ostream& print(std::ostream & os) const;

	template<typename ...Args> void next_timestep(Args && ...);

	template<typename TJ> void ScatterJ(TJ * J) const;

	template<typename TJ> void ScatterRho(TJ * rho) const;

};

template<typename TM, typename Engine>
template<typename ... Others>
Particle<TM, Engine>::Particle(mesh_type const & model, Others && ...others) :
		storage_type(model), mesh(model)
{
	this_type::load(std::forward<Others>(others)...);
}

template<typename TM, typename Engine>
Particle<TM, Engine>::~Particle()
{
}
template<typename TM, typename Engine>
template<typename TDict, typename ...Others>
void Particle<TM, Engine>::load(TDict const & dict, Others && ...others)
{
	engine_type::load(dict, std::forward<Others>(others)...);

	storage_type::load(dict, std::forward<Others>(others)...);

	properties.set("DumpParticle",
			dict["DumpParticle"].template as<bool>(false));

	properties.set("DivergeJ", dict["DivergeJ"].template as<bool>(false));

	properties.set("ScatterN", dict["ScatterN"].template as<bool>(false));

	J.clear();

	rho.clear();

}
template<typename ...T>
std::ostream &operator<<(std::ostream&os, Particle<T...> const & p)
{
	p.print(os);
	return os;
}

//*************************************************************************************************

template<typename TM, typename Engine>
std::string Particle<TM, Engine>::save(std::string const & path) const
{
	std::stringstream os;

	GLOBAL_DATA_STREAM.cd(path);

	os << "\n, n =" << simpla::save("rho", rho);

	os << "\n, J =" << simpla::save("J", J);

	if (properties["DumpParticle"].template as<bool>(false))
	{
		os << "\n, particles = " << storage_type::save("particles");
	}

	return os.str();
}

template<typename TM, typename Engine>
template<typename ...Args>
void Particle<TM, Engine>::next_timestep(Args && ...args)
{

	LOGGER << "Push particles to half step [ "
			<< engine_type::get_type_as_string() << " ]";

	storage_type::Sort();

	Real dt = mesh.get_dt();

	for (auto & cell : *this)
	{
		//TODO add rw cache
		for (auto & p : cell.second)
		{
			this->engine_type::next_timestep(&p, std::forward<Args>(args)...);
		}
	}

	storage_type::set_changed();
}

template<typename TM, typename Engine>
void Particle<TM, Engine>::update_fields()
{

	LOGGER
	articles to
	fields[" << engine_type::get_type_as_string() << "]
	";

	Real dt = mesh.get_dt();

	storage_type::Sort();

	J.clear();

	for (auto & cell : *this)
	{
		//TODO add rw cache
		for (auto & p : cell.second)
		{
			this->engine_type::ScatterJ(p, &J);
		}
	}

	update_ghosts (&J);

	if (engine_type::properties["DivergeJ"].template as<bool>(false))
	{
		LOG_CMD(rho -= Diverge(MapTo < EDGE > (J)) * dt);
	}
	else if (engine_type::properties["ScatterN"].template as<bool>(false))
	{
		rho.clear();

		for (auto & cell : *this)
		{
			//TODO add rw cache
			for (auto & p : cell.second)
			{
				this->engine_type::ScatterRho(p, &rho);
			}
		}

		update_ghosts (&rho);
	}
}

template<typename TM, typename Engine>
template<typename TRange, typename TFun>
void Particle<TM, Engine>::remove(TRange const & range, TFun const & obj)
{
	auto f1 = TypeCast<
			std::function<bool(coordinates_type const &, Vec3 const &)>>(obj);

	std::function<bool(particle_type const&)> fun = [&](particle_type const & p)
	{

		auto z=engine_type::pull_back(p);

		return f1(std::get<0>(z),std::get<1>(z));

	};

	storage_type::remove(range, fun);

}

template<typename TM, typename Engine>
template<typename TRange, typename TFun>
void Particle<TM, Engine>::modify(TRange const & range, TFun const & obj)
{

	auto f1 = TypeCast<
			std::function<
					std::tuple<coordinates_type, Vec3>(coordinates_type const &,
							Vec3 const &)>>(obj);

	std::function<void(particle_type *)> fun =
			[&](particle_type * p)
			{
				auto z0=engine_type::pull_back(*p);

				auto z1=f1(std::get<0>(z0),std::get<1>(z0));

				*p=engine_type::push_forward( std::get<0>(z1),std::get<1>(z1),std::get<2>(z0));
			};

	storage_type::modify(range, fun);

}

//*************************************************************************************************
template<typename TX, typename TV, typename TE, typename TB> inline
void BorisMethod(Real dt, Real cmr, TE const & E, TB const &B, TX *x, TV *v)
{
// \note   Birdsall(1991)   p.62
// Bories Method

	Vec3 v_;

	auto t = B * (cmr * dt * 0.5);

	(*v) += E * (cmr * dt * 0.5);

	v_ = (*v) + Cross((*v), t);

	(*v) += Cross(v_, t) * (2.0 / (Dot(t, t) + 1.0));

	(*v) += E * (cmr * dt * 0.5);

	(*x) += (*v) * dt;

}

}
// namespace simpla

#endif /* PARTICLE_IMPL_H_ */
