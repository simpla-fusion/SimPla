/*
 * explicit_em_impl.h
 *
 * \date  2013-12-29
 *      \author  salmon
 */

#ifndef EXPLICIT_EM_IMPL_H_
#define EXPLICIT_EM_IMPL_H_

#include <cmath>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>

// Misc
#include "../../src/utilities/log.h"
#include "../../src/utilities/pretty_stream.h"
// Data IO
#include "../../src/io/data_stream.h"

// Field
#include "../../src/fetl/fetl.h"
#include "../../src/fetl/save_field.h"
#include "../../src/fetl/load_field.h"

// Particle
#include "../../src/particle/particle_base.h"
#include "../../src/particle/particle_factory.h"
// Model
#include "../../src/model/model.h"
#include "../../src/model/geqdsk.h"
#include "../../src/flow_control/context_base.h"
#include "../../src/numeric/geometric_algorithm.h"

// Solver
#include "../field_solver/pml.h"
#include "../field_solver/implicitPushE.h"
#include "../particle_solver/register_particle.h"

namespace simpla
{

/**
 * \ingroup Application
 *
 * \brief Electromagnetic solver
 */
template<typename TM>
struct ExplicitEMContext: public ContextBase
{
public:

	typedef TM mesh_type;
	typedef typename mesh_type::scalar_type scalar_type;
	typedef ExplicitEMContext<mesh_type> this_type;
	typedef ContextBase base_type;

	ExplicitEMContext();

	template<typename ...Args>
	ExplicitEMContext(Args && ...args)
			: ExplicitEMContext()
	{
		load(std::forward<Args >(args)...);
	}

	~ExplicitEMContext();

	double CheckCourantDt() const;

	template<typename ...Args>
	static std::shared_ptr<base_type> create(Args && ... args)
	{
		return std::dynamic_pointer_cast<base_type>(
		        std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...)));
	}

	template<typename ...Args>
	static std::pair<std::string, std::function<std::shared_ptr<base_type>(Args const &...)>> CreateFactoryFun()
	{
		std::function<std::shared_ptr<base_type>(Args const &...)> call_back = []( Args const& ...args)
		{
			return this_type::create(args...);
		};
		return std::move(std::make_pair(get_type_as_string_static(), call_back));
	}

	// interface begin

	template<typename TDict> void load(TDict const & dict);

	template<typename OS> OS& print_(OS &) const;

//	std::string load(std::string const & path = "");

	std::string save(std::string const & path = "") const;

	static std::string get_type_as_string_static()
	{
		return "ExplicitEMContext_" + mesh_type::get_type_as_string_static();
	}

	std::string get_type_as_string() const
	{
		return get_type_as_string_static();
	}

	std::ostream & print(std::ostream & os) const
	{
		print_(os);
		return os;
	}

	void next_timestep();

	bool pre_process();

	bool post_process();

	bool empty() const
	{
		return !model.is_ready();
	}

	operator bool() const
	{
		return model.is_ready();
	}

	// interface end

public:

	std::string description;

	Model<mesh_type> model;

	template<int iform, typename TV> using field=typename mesh_type::template field<iform, TV>;

	field<EDGE, scalar_type> E, dE;
	field<FACE, scalar_type> B, dB;
	field<VERTEX, scalar_type> n, phi; //!< electrostatic potential

	field<VERTEX, Real> n0; //!< density
	field<EDGE, scalar_type> J0; //!<background current density J0+Curl(B(t=0))=0
	field<EDGE, scalar_type> Jext; //!< current density

	field<VERTEX, nTuple<3, Real> > Bv;
private:
	typedef decltype(E) TE;
	typedef decltype(B) TB;
	typedef decltype(Jext) TJ;

	typedef std::map<std::string, std::shared_ptr<ParticleBase> > TParticles;

	std::function<void(Real, TE const &, TB const &, TE*)> E_plus_CurlB;

	std::function<void(Real, TE const &, TB const &, TB*)> B_minus_CurlE;

	std::function<void(TE const &, TB const &, TParticles const&, TE*)> Implicit_PushE;

	template<typename TBatch>
	void ExcuteCommands(TBatch const & batch)
	{
		VERBOSE << "Apply constraints";
		for (auto const & command : batch)
		{
			command();
		}
	}

	std::list<std::function<void()> > commandToE_;

	std::list<std::function<void()> > commandToB_;

	std::list<std::function<void()> > commandToJ_;

	std::map<std::string, std::shared_ptr<ParticleBase>> particles_;

}
;

template<typename TM>
ExplicitEMContext<TM>::ExplicitEMContext()
		: E(model), B(model), Jext(model), J0(model), dE(model), dB(model), n(model), n0(model), //
		phi(model), Bv(model)
{
}

template<typename TM>
ExplicitEMContext<TM>::~ExplicitEMContext()
{
}

template<typename TM>
template<typename OS>
OS &ExplicitEMContext<TM>::print_(OS & os) const
{

	os

	<< "Description = \"" << description << "\"," << std::endl

	<< " Model = { " << model << "} ," << std::endl;

	if (particles_.size() > 0)
	{

		os << "\n , Particles = { \n";
		for (auto const & p : particles_)
		{
			os << p.first << " = { ";
			p.second->print(os);
			os << "}," << std::endl;
//			os << " { " << p.second << " }, " << std::endl;
		}
		os << "\n} ";
	}

	return os;

}

template<typename TM>
std::string ExplicitEMContext<TM>::save(std::string const & path) const
{

	GLOBAL_DATA_STREAM.cd(path);

	VERBOSE << SAVE(E);
	VERBOSE << SAVE(B);
	VERBOSE << SAVE(dE);
	VERBOSE << SAVE(dB);
	VERBOSE << SAVE(Jext);

	for (auto const & p : particles_)
	{
		VERBOSE << p.second->save(path);
	}

	return path;
}
template<typename TM> template<typename TDict>
void ExplicitEMContext<TM>::load(TDict const & dict)
{
	DEFINE_PHYSICAL_CONST

	LOGGER << "Load ExplicitEMContext ";

	description = dict["Description"].template as<std::string>();

	LOGGER << description;

	field<VERTEX, Real> ne0(model);
	field<VERTEX, Real> Te0(model);
	field<VERTEX, Real> Ti0(model);

	if (dict["Model"]["GFile"])
	{
		model.mesh_type::load(dict["Model"]["Mesh"]);

		model.Update();

		GEqdsk geqdsk;

		geqdsk.load(dict["Model"]["GFile"].template as<std::string>());

		geqdsk.save("/Geqdsk/");

		typename mesh_type::coordinates_type src_min;
		typename mesh_type::coordinates_type src_max;
		typename mesh_type::coordinates_type min1, min2, max1, max2;

		std::tie(src_min, src_max) = geqdsk.get_extents();

		min1 = model.MapTo(geqdsk.InvMapTo(src_min));
		max1 = model.MapTo(geqdsk.InvMapTo(src_max));

		std::tie(min2, max2) = model.get_extents();

//		min2[(mesh_type::ZAxis + 1) % 3] = min1[(mesh_type::ZAxis + 1) % 3];
//		min2[(mesh_type::ZAxis + 2) % 3] = src_min[GEqdsk::RAxis];
//		min2[(mesh_type::ZAxis + 3) % 3] = src_min[GEqdsk::ZAxis];
//
//		max2[(mesh_type::ZAxis + 1) % 3] = max1[(mesh_type::ZAxis + 1) % 3];
//		max2[(mesh_type::ZAxis + 2) % 3] = src_max[GEqdsk::RAxis];
//		max2[(mesh_type::ZAxis + 3) % 3] = src_max[GEqdsk::ZAxis];

		Clipping(min1, max1, &min2, &max2);

		model.set_extents(min2, max2);

		model.Update();

		geqdsk.SetUpMaterial(&model);

		E.clear();
		B.clear();
		J0.clear();
		ne0.clear();
		Te0.clear();
		Ti0.clear();

		geqdsk.GetProfile("B", &B);
		geqdsk.GetProfile("ne", &ne0);
		geqdsk.GetProfile("Te", &Te0);
		geqdsk.GetProfile("Ti", &Ti0);

		description = description + "\n GEqdsk ID:" + geqdsk.Description();

		J0 = Curl(B) / mu0;

		Jext = J0;

	}
	else
	{
		if (!model.load(dict["Model"]))
		{
			PARSER_ERROR("Configure 'Model' fail!");
		}
		model.Update();

		B.clear();

		J0.clear();

		Jext.clear();

		E.clear();

		VERBOSE_CMD(load_field(dict["InitValue"]["B"], &B));

		VERBOSE_CMD(load_field(dict["InitValue"]["J"], &J0));

		VERBOSE_CMD(load_field(dict["InitValue"]["ne"], &ne0));

		VERBOSE_CMD(load_field(dict["InitValue"]["Te"], &Te0));

		VERBOSE_CMD(load_field(dict["InitValue"]["Ti"], &Ti0));

		Jext = J0;
	}

	dB.clear();

	dE.clear();

	VERBOSE_CMD(load_field(dict["InitValue"]["E"], &E));

	LOGGER << "Load Particles";

	auto particle_factory = RegisterAllParticles<mesh_type, TDict, Model<mesh_type> const &, decltype(ne0),
	        decltype(Te0)>();

	/**
	 * @todo load particle engine plugin
	 *
	 *  add new creator at here
	 *
	 */
	for (auto opt : dict["Particles"])
	{
		auto id = opt.first.template as<std::string>("unnamed");

		auto type_str = opt.second["Type"].template as<std::string>("");

		try
		{
			auto p = particle_factory.create(type_str, opt.second, model, ne0, Te0);

			if (p != nullptr)
			{

				particles_.emplace(id, p);

			}

		} catch (...)
		{

			PARSER_ERROR("Particle={" + id + " = { Type = " + type_str + "}}" + "  ");

		}

	}

	bool enableImplicit = false;

	for (auto const &p : particles_)
	{
		enableImplicit = enableImplicit || p.second->is_implicit();
	}

	LOGGER << "Load Constraints";

	for (auto const & item : dict["Constraints"])
	{
		try
		{

			auto dof = item.second["DOF"].template as<std::string>("");

			VERBOSE << "Add constraint to " << dof;

			if (dof == "E")
			{
				commandToE_.push_back(
				        E.CreateCommand(model.SelectByConfig(E.IForm, item.second["Select"]),
				                item.second["Operation"]));
			}
			else if (dof == "B")
			{

				commandToB_.push_back(
				        B.CreateCommand(model.SelectByConfig(B.IForm, item.second["Select"]),
				                item.second["Operation"]));
			}
			else if (dof == "J")
			{

				commandToJ_.push_back(
				        Jext.CreateCommand(model.SelectByConfig(Jext.IForm, item.second["Select"]),
				                item.second["Operation"]));
			}
			else
			{
				PARSER_ERROR("Unknown DOF!");
			}

		} catch (std::runtime_error const & e)
		{

			PARSER_ERROR("Load 'Constraints' error! ");
		}
	}

	bool enablePML = false;

	try
	{
		LOGGER << "Load electromagnetic fields solver";

		using namespace std::placeholders;

		Real ic2 = 1.0 / (mu0 * epsilon0);

		if (dict["FieldSolver"]["PML"])
		{
			auto solver = std::shared_ptr<PML<TM> >(new PML<TM>(model, dict["FieldSolver"]["PML"]));

			E_plus_CurlB = std::bind(&PML<TM>::next_timestepE, solver, _1, _2, _3, _4);

			B_minus_CurlE = std::bind(&PML<TM>::next_timestepB, solver, _1, _2, _3, _4);

		}
		else
		{
			E_plus_CurlB = [mu0 , epsilon0](Real dt, TE const & E , TB const & B, TE* pdE)
			{
				auto & dE=*pdE;
				VERBOSE_CMD(dE += Curl(B)/(mu0 * epsilon0) *dt);
			};

			B_minus_CurlE = [](Real dt, TE const & E, TB const &, TB* pdB)
			{
				auto & dB=*pdB;
				VERBOSE_CMD( dB -= Curl(E)*dt);
			};
		}

	} catch (std::runtime_error const & e)
	{
		PARSER_ERROR("Configure field solver error! ");
	}

	Implicit_PushE = [] ( TE const &, TB const &, TParticles const&, TE*)
	{};

	if (enableImplicit)
	{

		auto solver = std::shared_ptr<ImplicitPushE<mesh_type>>(new ImplicitPushE<mesh_type>(model));
		Implicit_PushE = [solver] ( TE const & pE, TB const & pB, TParticles const&p, TE*dE)
		{	solver->next_timestep( pE,pB,p,dE);};
	}

}
template<typename TM>
bool ExplicitEMContext<TM>::pre_process()
{
	return true;
}

template<typename TM>
bool ExplicitEMContext<TM>::post_process()
{
	return true;
}
template<typename TM>
void ExplicitEMContext<TM>::next_timestep()
{
	DEFINE_PHYSICAL_CONST

	INFORM

	<< "[" << model.get_clock() << "]"

	<< "Simulation Time = " << (model.get_time() / CONSTANTS["s"]) << "[s]";

	Real dt = model.get_dt();

	// Compute Cycle Begin

	LOG_CMD(B -= Curl(E) * dt / (mu0 * epsilon0));

	LOG_CMD(E += Curl(B) * dt);

	//	// E0 B0,
	//	LOG_CMD(Jext = J0);
	//	ExcuteCommands(commandToJ_);
	//
	//	//   particle 0-> 1/2 . To n[1/2], J[1/2]
	//	for (auto &p : particles_)
	//	{
	//		if (!p.second->is_implicit())
	//		{
	//			p.second->next_timestep_zero(E, B);
	//
	//			auto const & Js = p.second->template J<TJ>();
	//			LOG_CMD(Jext += Js);
	//		}
	//	}
	//
	//	LOG_CMD(B += dB * 0.5);	//  B(t=0 -> 1/2)
	//	ExcuteCommands(commandToB_);
	//
	//	dE.clear();
	//	E_plus_CurlB(dt, E, B, &dE);	// dE += Curl(B)*dt
	//
	//	LOG_CMD(dE -= Jext * (dt / epsilon0));
	//
	////   particle 1/2 -> 1  . To n[1/2], J[1/2]
	//	Implicit_PushE(E, B, particles_, &dE);
	//
	//	LOG_CMD(E += dE);	// E(t=0 -> 1)
	//	ExcuteCommands(commandToE_);
	//
	//	dB.clear();
	//	B_minus_CurlE(dt, E, B, &dB);
	//
	//	LOG_CMD(B += dB * 0.5);	//	B(t=1/2 -> 1)
	//	ExcuteCommands(commandToB_);

	// Compute Cycle End
	model.next_timestep();

}

}
// namespace simpla

#endif /* EXPLICIT_EM_IMPL_H_ */
