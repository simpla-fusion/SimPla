/*
 * fluid_cold_engine.h
 *
 * @date  2014-4-15
 *      @author  salmon
 */

#ifndef FLUID_COLD_ENGINE_H_
#define FLUID_COLD_ENGINE_H_
#include <functional>
#include <typeinfo>
#include "../../core/field/field.h"
#include "../../core/field/loadField.h"
#include "../../core/field/saveField.h"
#include "../../core/particle/particle_base.h"
#include "../../core/utilities/properties.h"
#include "../../core/utilities/any.h"

#include "../../core/physics/physical_constants.h"

namespace simpla
{

class PolicyFluidParticle;
class ColdFluid;
template<typename ...> class Particle;

/**
 * @ingroup ParticleEngine
 * \brief Cold Plasma fluid
 */
template<typename TM>
class Particle<TM, ColdFluid, PolicyFluidParticle>
{
public:
	static constexpr std::size_t IForm = VERTEX;

	typedef TM mesh_type;

	typedef ColdFluid engine_type;

	typedef Particle<mesh_type, engine_type, PolicyFluidParticle> this_type;

	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type::iterator iterator;

	typedef typename mesh_type::coordinate_tuple coordinate_tuple;

	typedef typename mesh_type:: template field<VERTEX, scalar_type> rho_type;

	typedef typename mesh_type:: template field<VERTEX, nTuple<scalar_type, 3>> J_type;

	typedef typename mesh_type:: template field<VERTEX, nTuple<Real, 3> > E0_type;

	typedef typename mesh_type:: template field<VERTEX, nTuple<Real, 3> > B0_type;

	typedef typename mesh_type:: template field<VERTEX, nTuple<scalar_type, 3>> E1_type;

	typedef typename mesh_type:: template field<VERTEX, nTuple<scalar_type, 3>> B1_type;

	mesh_type const & mesh;

	Properties properties;
	enum
	{
		is_implicit_ = true
	};
	const Real m;
	const Real q;

	rho_type rho;
	J_type J;

	template<typename TDict, typename TModel, typename ...Args>
	Particle(TDict const & dict, TModel const & model, Args && ... args);

	~Particle();

	template<typename TDict, typename TModel>
	void load(TDict const & dict, TModel const & model);

	template<typename TDict, typename TModel, typename TN, typename TJ>
	void load(TDict const & dict, TModel const & model, TN const &, TJ const &);

	std::string save(std::string const & path) const;

	std::ostream& print(std::ostream & os) const
	{
		return print_(os);
	}

	template<typename OS>
	OS & print_(OS& os) const
	{
		DEFINE_PHYSICAL_CONST
		;

		os << "Engine = '" << get_type_as_string() << "' "

		<< " , " << "Mass = " << m / proton_mass << " * m_p"

		<< " , " << "Charge = " << q / elementary_charge << " * q_e"

		<< " , " << properties;

		return os;
	}
	void set_property_(std::string const & name, any const&v)
	{
		properties[name] = v;
	}
	any const & get_property_(std::string const &name) const
	{
		return properties[name].template as<any>();
	}

	template<typename T> void set_property(std::string const & name, T const&v)
	{
		set_property_(name, any(v));
	}

	template<typename T> T get_property(std::string const & name) const
	{
		return get_property_(name).template as<T>();
	}

	void next_timestep_zero(E0_type const & E0, B0_type const & B0,
			E1_type const & E1, B1_type const & B1);
	void next_timestep_half(E0_type const & E0, B0_type const & B0,
			E1_type const & E1, B1_type const & B1);

// interface

	bool check_mesh_type(std::type_info const & t_info) const
	{
		return t_info == typeid(mesh_type);
	}
	bool check_E_type(std::type_info const & t_info) const
	{
		return t_info == typeid(E1_type);
	}
	bool check_B_type(std::type_info const & t_info) const
	{
		return t_info == typeid(B1_type);
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

	void const * get_rho() const
	{
		return reinterpret_cast<void const*>(&rho);
	}

	void const * get_J() const
	{
		return reinterpret_cast<void const*>(&J);
	}

	void updateFields();

	void next_timestep()
	{
	}

private:

}
;

template<typename TM>
template<typename TDict, typename TModel, typename ...Args>
Particle<TM, ColdFluid, PolicyFluidParticle>::Particle(TDict const & dict,
		TModel const & model, Args && ... args)
		: mesh(model),

		m(dict["Mass"].template as<Real>(1.0)),

		q(dict["Charge"].template as<Real>(1.0)),

		rho(mesh), J(mesh)
{
	load(dict, model, std::forward<Args>(args)...);
}

template<typename TM>
Particle<TM, ColdFluid, PolicyFluidParticle>::~Particle()
{
}
template<typename TM>
template<typename TDict, typename TModel>
void Particle<TM, ColdFluid, PolicyFluidParticle>::load(TDict const & dict,
		TModel const & model)
{

	try
	{
		loadField(dict["Density"], &(rho));
		loadField(dict["Current"], &(J));

		if (!rho.empty())
		{
			rho *= get_charge() * dict["Ratio"].template as<Real>(1.0);
		}

	} catch (...)
	{
		PARSER_ERROR("Configure  Particle<ColdFluid> error!");
	}

	LOGGER << "Create Particles:[ Engine=" << get_type_as_string() << "]"
			<< DONE;
}

template<typename TM>
template<typename TDict, typename TModel, typename TN, typename TT>
void Particle<TM, ColdFluid, PolicyFluidParticle>::load(TDict const & dict,
		TModel const & model, TN const & pn, TT const & pT)
{

	load(dict, model);

	if (rho.empty())
	{
		rho.clear();

		auto range = model.select_by_config(rho_type::IForm, dict["Select"]);

		rho.pull_back(range, model, pn);

		rho *= get_charge() * dict["Ratio"].template as<Real>(1.0);
	}

	if (J.empty())
	{
		J.clear();
	}

//	LOGGER << "Create Particles:[ Engine=" << get_type_as_string() << "]" << DONE;
}

template<typename TM>
std::string Particle<TM, ColdFluid, PolicyFluidParticle>::save(
		std::string const & path) const
{
	cd(path);

	return

	"\n, n =" + simpla::save("rho", rho) +

	"\n, J =" + simpla::save("J", J)

	;
}
template<typename TM>
void Particle<TM, ColdFluid, PolicyFluidParticle>::next_timestep_zero(
		E0_type const & E0, B0_type const & B0, E1_type const & E1,
		B1_type const & B1)
{
}

template<typename TM>
void Particle<TM, ColdFluid, PolicyFluidParticle>::next_timestep_half(
		E0_type const & E0, B0_type const & B0, E1_type const & E1,
		B1_type const & B1)
{
	LOGGER << "Push particles Step Half[ " << get_type_as_string() << "]";

	auto K = mesh.template makeField<VERTEX, nTuple<scalar_type, 3>>();
	auto B2 = mesh.template makeField<VERTEX, Real>();

	Real as = 0.5 * q / m * mesh.get_dt();

	K = J + cross(J, B0) * as + 2.0 * as * rho * E1;

	B2 = dot(B0, B0);

	J = (K + cross(K, B0) * as + B0 * (dot(K, B0) * as * as))
			/ (B2 * as * as + 1);

}
template<typename TM>
void Particle<TM, ColdFluid, PolicyFluidParticle>::updateFields()
{
	LOGGER << "Push particles update fields[ " << get_type_as_string() << "]";

	Real dt = mesh.get_dt();

	if (properties["DivergeJ"].template as<bool>(true))
	{
		LOG_CMD(rho -= diverge(map_to<EDGE>(J)) * dt);
	}

}
}
// namespace simpla

#endif /* FLUID_COLD_ENGINE_H_ */
