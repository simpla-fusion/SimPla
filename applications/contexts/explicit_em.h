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

	void InitPECboundary();

public:

	std::string description;

	Model<mesh_type> model;

	template<int iform, typename TV> using field=typename mesh_type::template field<iform, TV>;

	field<EDGE, scalar_type> E1, dE;
	field<FACE, scalar_type> B1, dB;

	field<EDGE, scalar_type> J1; //!< current density
	field<EDGE, Real> Jext; //!< external current

//	field<VERTEX, scalar_type>  phi; //!< electrostatic potential

	ImplicitPushE<mesh_type> implicit_push_E;

	field<VERTEX, Real> n0; //!< background  equilibrium electron density
	field<VERTEX, nTuple<3, Real>> E0; //!<background  equilibrium electoric field  (B0)=0
	field<VERTEX, nTuple<3, Real>> B0; //!<background  equilibrium magnetic field J0+Curl(B0)=0
//	field<EDGE, Real> J0; //!<background  equilibrium current density J0+Curl(B0)=0

//	PML<mesh_type> pml_push;

private:
	typedef decltype(E1) E_type;
	typedef decltype(B1) B_type;
	typedef decltype(J1) J_type;

	typedef std::map<std::string, std::shared_ptr<ParticleBase> > TParticles;

	template<typename TBatch>
	void ExcuteCommands(TBatch const & batch)
	{
		if (batch.size() == 0)
			return;
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
		: E1(model), B1(model), J1(model), dE(model), dB(model),

		B0(model), E0(model),

		n0(model), Jext(model),

		implicit_push_E(model)
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

	auto abs_path = (GLOBAL_DATA_STREAM.cd(path));

	VERBOSE << SAVE(E1);
	VERBOSE << SAVE(B1);
	VERBOSE << SAVE(J1);
	VERBOSE << SAVE(Jext);
	VERBOSE << SAVE(dE);
	VERBOSE << SAVE(dB);

	for (auto const & p : particles_)
	{
		VERBOSE << p.second->save(abs_path+p.first+"/");
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

		Clipping(min2, max2, &min1, &max1);

		auto dims = model.get_dimensions();

		if (model.enable_spectral_method)
		{

			/**
			 *  @bug Lua can not handle field with complex value!!
			 */

			for (int i = 0; i < model.NDIMS; ++i)
			{
				if (dims[i] <= 1)
				{
					min1[i] = min2[i];
					max1[i] = max2[i];
				}
			}
		}

		if (dims[2] > 1)
		{
			min1[2] = min2[2];
			max1[2] = max2[2];
		}

		model.set_extents(min1, max1);

		model.update();

		geqdsk.SetUpMaterial(&model);

		B1.clear();
		E1.clear();
		J1.clear();
		B0.clear();
		ne0.clear();
		Te0.clear();
		Ti0.clear();

		geqdsk.GetProfile("B", &B0);
		geqdsk.GetProfile("ne", &ne0);
		geqdsk.GetProfile("Te", &Te0);
		geqdsk.GetProfile("Ti", &Ti0);

		description = description + "\n GEqdsk ID:" + geqdsk.Description();
		InitPECboundary();
	}
	else
	{
		if (!model.load(dict["Model"]))
		{
			PARSER_ERROR("Configure 'Model' fail!");
		}

		B1.clear();
		E1.clear();
		J1.clear();
		B0.clear();

		VERBOSE_CMD(load_field(dict["InitValue"]["B"], &B1));

		VERBOSE_CMD(load_field(dict["InitValue"]["E"], &E1));

		VERBOSE_CMD(load_field(dict["InitValue"]["J"], &J1));

		VERBOSE_CMD(load_field(dict["InitValue"]["ne"], &ne0));

		VERBOSE_CMD(load_field(dict["InitValue"]["Te"], &Te0));

		VERBOSE_CMD(load_field(dict["InitValue"]["Ti"], &Ti0));

	}

	dB.clear();
	dE.clear();
	E0.clear();
	Jext.clear();
	GLOBAL_DATA_STREAM.cd("/Input/");

	VERBOSE << SAVE(ne0);
	VERBOSE << SAVE(Te0);
	VERBOSE << SAVE(Ti0);
	VERBOSE << SAVE(B0);
	VERBOSE << SAVE(E0);

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
				        E1.CreateCommand(model.SelectByConfig(E1.IForm, item.second["Select"]),
				                item.second["Operation"]));
			}
			else if (dof == "B")
			{

				commandToB_.push_back(
				        B1.CreateCommand(model.SelectByConfig(B1.IForm, item.second["Select"]),
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
//	bool enableImplicit = false;
//
//	for (auto const &p : particles_)
//	{
//		enableImplicit = enableImplicit || p.second->is_implicit();
//	}
//	bool enablePML = false;
//
//	try
//	{
//		LOGGER << "Load electromagnetic fields solver";
//
//		using namespace std::placeholders;
//
//		Real ic2 = 1.0 / (mu0 * epsilon0);
//
//		if (dict["FieldSolver"]["PML"])
//		{
//			auto solver = std::shared_ptr<PML<TM> >(new PML<TM>(model, dict["FieldSolver"]["PML"]));
//
//			E_plus_CurlB = std::bind(&PML<TM>::next_timestepE, solver, _1, _2, _3, _4);
//
//			B_minus_CurlE = std::bind(&PML<TM>::next_timestepB, solver, _1, _2, _3, _4);
//
//		}
//		else
//		{
//			E_plus_CurlB = [mu0 , epsilon0](Real dt, TE const & E , TB const & B, TE* pdE)
//			{
//				auto & dE=*pdE;
//				VERBOSE_CMD(dE += Curl(B)/(mu0 * epsilon0) *dt);
//			};
//
//			B_minus_CurlE = [](Real dt, TE const & E, TB const &, TB* pdB)
//			{
//				auto & dB=*pdB;
//				VERBOSE_CMD( dB -= Curl(E)*dt);
//			};
//		}
//
//	} catch (std::runtime_error const & e)
//	{
//		PARSER_ERROR("Configure field solver error! ");
//	}
//	if (enableImplicit)
//	{
//
//		auto solver = std::shared_ptr<ImplicitPushE<mesh_type>>(new);
//		Implicit_PushE = [solver] ( TE const & pE, TB const & pB, TParticles const&p, TE*dE)
//		{	solver->next_timestep( pE,pB,p,dE);};
//	}

}
template<typename TM>
bool ExplicitEMContext<TM>::pre_process()
{
	return true;
}
template<typename TM>
void ExplicitEMContext<TM>::InitPECboundary()
{
	std::vector<typename mesh_type::compact_index_type> conduct_wall_E_;

	for (auto s : model.Select(E_type::IForm))
	{
		if (model.get(s) == model.null_material)
		{
			conduct_wall_E_.push_back(s);
		}
	}
	if (conduct_wall_E_.size() > 0)
	{
		std::function<void()> fun = [=]()
		{
			VERBOSE << "Apply PEC to E1";
			for (auto s : conduct_wall_E_)
			{
				get_value(this->E1, s) = 0;

			};
		};
		commandToE_.push_back(fun);
	}

	std::vector<typename mesh_type::compact_index_type> conduct_wall_B_;

	for (auto s : model.Select(B_type::IForm))
	{
		if (model.get(s) == model.null_material)
		{
			conduct_wall_B_.push_back(s);
		}
	}

	if (conduct_wall_B_.size() > 0)
	{
		std::function<void()> fun = [=]()
		{
			VERBOSE << "Apply PEC to B1 ";
			for (auto s : conduct_wall_B_)
			{
				get_value(this->B1, s) = 0;

			};
		};
		commandToE_.push_back(fun);
	}

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

	J1.clear();

	//   particle 0-> 1/2 . To n[1/2], J[1/2]
	for (auto &p : particles_)
	{
		if (!p.second->is_implicit())
		{
			p.second->next_timestep_zero(E0, B0, E1, B1);
			p.second->update_fields();

			auto const & Js = p.second->template J<J_type>();
			LOG_CMD(J1 += Js);
		}
	}
	Jext.clear();
	ExcuteCommands(commandToJ_);
	J1 += Jext;

	LOG_CMD(B1 += dB * 0.5);	//  B(t=0 -> 1/2)
	ExcuteCommands(commandToB_);

	LOG_CMD(dE = (Curl(B1) / mu0 - J1) / epsilon0 * dt);

//   particle 1/2 -> 1  . To n[1/2], J[1/2]
	implicit_push_E.next_timestep(E0, B0, E1, B1, particles_, &dE);

	LOG_CMD(E1 += dE * 0.5);	// E(t=0 -> 1)
	ExcuteCommands(commandToE_);

	for (auto &p : particles_)
	{
		if (!p.second->is_implicit())
		{
			p.second->next_timestep_half(E0, B0, E1, B1);
		}
	}

	LOG_CMD(E1 += dE * 0.5);	// E(t=0 -> 1)
	ExcuteCommands(commandToE_);

	LOG_CMD(dB = -Curl(E1) * dt);

	LOG_CMD(B1 += dB * 0.5);	//	B(t=1/2 -> 1)
	ExcuteCommands(commandToB_);
// Compute Cycle End
	model.next_timestep();

}

}
// namespace simpla

#endif /* EXPLICIT_EM_IMPL_H_ */
