/*
 * fluid_cold_engine.h
 *
 * \date  2014-4-15
 *      \author  salmon
 */

#ifndef FLUID_COLD_ENGINE_H_
#define FLUID_COLD_ENGINE_H_
#include <functional>
#include <typeinfo>
#include "../../src/fetl/fetl.h"
#include "../../src/fetl/load_field.h"
#include "../../src/fetl/save_field.h"
#include "../../src/particle/particle_base.h"
#include "../../src/utilities/properties.h"
#include "../../src/utilities/any.h"

#include "../../src/physics/physical_constants.h"

namespace simpla
{

template<typename > class ColdFluid;
template<typename > class Particle;

/**
 * \ingroup ParticleEngine
 * \brief Cold Plasma fluid
 */
template<typename TM>
class Particle<ColdFluid<TM> > : public ParticleBase
{
public:
	static constexpr unsigned int IForm = VERTEX;

	typedef TM mesh_type;

	typedef ColdFluid<mesh_type> engine_type;

	typedef Particle<engine_type> this_type;

	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type::iterator iterator;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type:: template field<VERTEX, scalar_type> n_type;

	typedef typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> J_type;

	typedef typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> E_type;

	typedef typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> B_type;

	enum
	{
		is_implicit_ = true
	};
	const Real m;
	const Real q;

	mesh_type const & mesh;

	n_type n;
	J_type J;

	template<typename TDict>
	Particle(TDict const & dict, mesh_type const & pmesh);

	~Particle();

	Properties properties;

	void set_property_(std::string const & name, Any const&v)
	{
		properties[name] = v;
	}
	Any const & get_property_(std::string const &name) const
	{
		return properties[name].template as<Any>();
	}

	template<typename T> void set_property(std::string const & name, T const&v)
	{
		set_property_(name, Any(v));
	}

	template<typename T> T get_property(std::string const & name) const
	{
		return get_property_(name).template as<T>();
	}

	template<typename OS>
	OS & print_(OS& os) const
	{
		DEFINE_PHYSICAL_CONST
		;

		os << "Engine = '" << get_type_as_string() << "' "

		<< " , " << "Mass = " << m / proton_mass << " * m_p"

		<< " , " << "Charge = " << q / elementary_charge << " * q_e"

		;

		return os;
	}

	template<typename TDict, typename TModel, typename ...Args>
	static std::shared_ptr<ParticleBase> create(TDict dict, TModel const & model, Args && ... args)
	{
		std::shared_ptr<this_type> res(new this_type(dict, model));

		try
		{
			res->J.clear();
			res->n.clear();

			load_field(dict["Density"], &(res->n));
			load_field(dict["Current"], &(res->J));

			res->n *= res->q;
		} catch (...)
		{
			PARSER_ERROR("Configure  Particle<ColdFluid> error!");
		}

		return std::dynamic_pointer_cast<ParticleBase>(res);
	}

	template<typename ...Args>
	static std::pair<std::string, std::function<std::shared_ptr<ParticleBase>(Args const &...)>> CreateFactoryFun()
	{
		std::function<std::shared_ptr<ParticleBase>(Args const &...)> call_back = []( Args const& ...args)
		{
			return this_type::create(args...);
		};
		return std::move(std::make_pair(get_type_as_string_static(), call_back));
	}

	void next_timestep_zero(E_type const & E, B_type const & B);

	void next_timestep_half(E_type const & E, B_type const & B);

	// interface

	bool same_mesh_type(std::type_info const & t_info) const
	{
		return t_info == typeid(mesh_type);
	}

	std::string save(std::string const & path) const;

	std::ostream& print(std::ostream & os) const
	{
		return print_(os);
	}

	Real get_mass() const
	{
		return m;
	}

	Real get_charge() const
	{
		return q;
	}

	bool is_implicit() const
	{
		return is_implicit_;
	}

	static std::string get_type_as_string_static()
	{
		return "ColdFluid";
	}

	std::string get_type_as_string() const
	{
		return get_type_as_string_static();
	}

	void const * get_n() const
	{
		return reinterpret_cast<void const*>(&n);
	}

	void const * get_J() const
	{
		return reinterpret_cast<void const*>(&J);
	}

	void next_timestep_zero_(void const * E, void const*B)
	{
		next_timestep_zero(*reinterpret_cast<E_type const*>(E), *reinterpret_cast<B_type const*>(B));
	}

	void next_timestep_half_(void const * E, void const*B)
	{
		next_timestep_half(*reinterpret_cast<E_type const*>(E), *reinterpret_cast<B_type const*>(B));
	}

private:

}
;

template<typename TM>
template<typename TDict> Particle<ColdFluid<TM>>::Particle(TDict const & dict, mesh_type const & pmesh)
		: mesh(pmesh),

		m(dict["Mass"].template as<Real>(1.0)),

		q(dict["Charge"].template as<Real>(1.0)),

		n(mesh), J(mesh)
{
}

template<typename TM>
Particle<ColdFluid<TM>>::~Particle()
{
}

template<typename TM>
std::string Particle<ColdFluid<TM>>::save(std::string const & path) const
{
	std::stringstream os;

	GLOBAL_DATA_STREAM.cd(path );

	DEFINE_PHYSICAL_CONST
	;
//
//	if (is_verbose)
//	{
//		os
//
//		<< "Engine = '" << get_type_as_string()
//
//		<< " , " << "Mass = " << m / proton_mass << " * m_p"
//
//		<< " , " << "Charge = " << q / elementary_charge << " * q_e"
//
//		;
//	}

	os << "\n, n =" << simpla::save("n", n);

	os << "\n, J =" << simpla::save("J", J);

	return os.str();
}
template<typename TM>
void Particle<ColdFluid<TM>>::next_timestep_zero(E_type const & E, B_type const & B)
{
	LOGGER << "Push particles Step Zero[ " << get_type_as_string() << "]";
	Real dt = mesh.get_dt();
	LOG_CMD(n -= Diverge(MapTo<EDGE>(J)) * dt);
}
template<typename TM>
void Particle<ColdFluid<TM>>::next_timestep_half(E_type const & E, B_type const & B)
{
	LOGGER << "Push particles Step Half[ " << get_type_as_string() << "]";

	auto K = mesh.template make_field<VERTEX, nTuple<3, scalar_type>>();

	Real as = 0.5 * q / m * mesh.get_dt();

	K = J + Cross(J, B) * as + 2.0 * as * n * E;

	J = (K + Cross(K, B) * as + B * (Dot(K, B) * as * as)) / (Dot(B, B) * as * as + 1);

}

}
// namespace simpla

#endif /* FLUID_COLD_ENGINE_H_ */
