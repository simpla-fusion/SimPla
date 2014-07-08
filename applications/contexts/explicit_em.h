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
struct ExplicitEMContext
{
public:

	typedef TM mesh_type;

	typedef ExplicitEMContext<mesh_type> this_type;

	ExplicitEMContext();

	template<typename ...Args>
	ExplicitEMContext(Args && ...args)
			: ExplicitEMContext()
	{
		Load(std::forward<Args >(args)...);
	}
	~ExplicitEMContext();

	template<typename TDict> void Load(TDict const & dict);

	void NextTimeStep();

	std::string Save(std::string const & path = "", bool is_verbose = false) const;

	double CheckCourantDt() const;

public:

	typedef typename mesh_type::scalar_type scalar_type;

	std::string description;

	Model<mesh_type> model_;

	mesh_type & mesh;

	typename mesh_type::template field<EDGE, scalar_type> E, dE;
	typename mesh_type::template field<FACE, scalar_type> B, dB;
	typename mesh_type::template field<VERTEX, scalar_type> n, phi; //!< electrostatic potential

	typename mesh_type::template field<VERTEX, Real> n0; //!< density
	typename mesh_type::template field<EDGE, scalar_type> J0; //!<background current density J0+Curl(B(t=0))=0
	typename mesh_type::template field<EDGE, scalar_type> Jext; //!< current density

	typename mesh_type::template field<VERTEX, nTuple<3, Real> > Bv;

	typedef decltype(E) TE;
	typedef decltype(B) TB;
	typedef decltype(Jext) TJ;

	typedef std::map<std::string, std::shared_ptr<ParticleBase<mesh_type> > > TParticles;

	std::function<void(Real, TE const &, TB const &, TE*)> E_plus_CurlB;

	std::function<void(Real, TE const &, TB const &, TB*)> B_minus_CurlE;

	std::function<void(TE const &, TB const &, TParticles const&, TE*)> Implicit_PushE;

	template<typename TBatch>
	void ExcuteCommands(TBatch const & batch)
	{
		LOGGER << "Apply constraints";
		for (auto const & command : batch)
		{
			command();
		}
	}

private:

	std::list<std::function<void()> > commandToE_;

	std::list<std::function<void()> > commandToB_;

	std::list<std::function<void()> > commandToJ_;

	std::map<std::string, std::shared_ptr<ParticleBase<mesh_type>>>particles_;

}
;

template<typename TM>
ExplicitEMContext<TM>::ExplicitEMContext()
		: mesh(model_.mesh), E(mesh), B(mesh), Jext(mesh), J0(mesh), dE(mesh), dB(mesh), n(mesh), n0(mesh), //
		phi(mesh), Bv(mesh)
{
}

template<typename TM>
ExplicitEMContext<TM>::~ExplicitEMContext()
{
}
template<typename TM> template<typename TDict>
void ExplicitEMContext<TM>::Load(TDict const & dict)
{
	DEFINE_PHYSICAL_CONST

	LOGGER << "Load ExplicitEMContext ";

	description = "Description = \"" + dict["Description"].template as<std::string>() + "\"\n";

	LOGGER << description;

	auto ne0 = mesh.template make_field<VERTEX, Real>();
	auto Te0 = mesh.template make_field<VERTEX, Real>();
	auto Ti0 = mesh.template make_field<VERTEX, Real>();

	model_.mesh.Load(dict["Model"]["Grid"]);

	if (dict["Model"]["GFile"])
	{
		GEqdsk geqdsk;

		geqdsk.Load(dict["Model"]["GFile"]["File"].template as<std::string>());

		if (!model_.is_ready())
		{
			typename mesh_type::coordinates_type src_min;
			typename mesh_type::coordinates_type src_max;

			std::tie(src_min, src_max) = geqdsk.get_extents();

			typename mesh_type::coordinates_type min;
			typename mesh_type::coordinates_type max;

			std::tie(min, max) = model_.mesh.get_extents();

			min[(mesh_type::ZAxis + 2) % 3] = src_min[GEqdsk::RAxis];
			max[(mesh_type::ZAxis + 2) % 3] = src_max[GEqdsk::RAxis];

			min[mesh_type::ZAxis] = src_min[GEqdsk::ZAxis];
			max[mesh_type::ZAxis] = src_max[GEqdsk::ZAxis];

			model_.mesh.set_extents(min, max);
		}

		geqdsk.SetUpModel(&model_);

		model_.Update();

		ne0.clear();
		Te0.clear();
		Ti0.clear();

		geqdsk.GetProfile("B", &B);
		geqdsk.GetProfile("ne", &ne0);
		geqdsk.GetProfile("Te", &Te0);
		geqdsk.GetProfile("Ti", &Ti0);

		description = description + "\n GEqdsk ID:" + geqdsk.Description();

		geqdsk.Save("/Geqdsk");

		J0 = Curl(B) / mu0;

		Jext = J0;

	}
	else
	{
		model_.Update();

		B.clear();

		Jext.clear();

		J0.clear();

		LOG_CMD(LoadField(dict["InitValue"]["B"], &B));

		LOG_CMD(LoadField(dict["InitValue"]["J"], &Jext));

		LOG_CMD(LoadField(dict["InitValue"]["ne"], &ne0));

		LOG_CMD(LoadField(dict["InitValue"]["Te"], &Te0));

		LOG_CMD(LoadField(dict["InitValue"]["Ti"], &Ti0));
	}

	dB.clear();

	dE.clear();

	E.clear();

	LOG_CMD(LoadField(dict["InitValue"]["E"], &E));

//	LOGGER << simpla::Save("B", B);
//	LOGGER << simpla::Save("J0", J0);
//	LOGGER << simpla::Save("ne_", ne0);
//	LOGGER << simpla::Save("Te_", Te0);
//	LOGGER << simpla::Save("Ti_", Ti0);

	bool enableImplicit = false;

	bool enablePML = false;

	LOGGER << "Load Particles";

	auto particle_factory = RegisterAllParticles<mesh_type, TDict, Model<mesh_type> const &, decltype(ne0),
	        decltype(Te0)>();

	/**
	 * @todo load particle engine plugins
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
			auto p = particle_factory.Create(type_str, opt.second, model_, ne0, Te0);

			if (p != nullptr)
			{

				particles_.emplace(id, p);

			}

		} catch (...)
		{

			PARSER_ERROR("Particles={" + id + " = { Type = " + type_str + "}}" + "  ");

		}

	}

	for (auto const &p : particles_)
	{
		enableImplicit = enableImplicit || p.second->EnableImplicit();
	}

	LOGGER << "Load Constraints";

	for (auto const & item : dict["Constraints"])
	{
		try
		{

			auto dof = item.second["DOF"].template as<std::string>("");

			LOGGER << "Add constraint to " << dof;

			if (dof == "E")
			{

				commandToE_.push_back(E.CreateCommand(model_.Select(item.second["Select"]), item.second["Operation"]));
			}
			else if (dof == "B")
			{

				commandToB_.push_back(B.CreateCommand(model_.Select(item.second["Select"]), item.second["Operation"]));
			}
			else if (dof == "J")
			{

				commandToJ_.push_back(
				        Jext.CreateCommand(model_.Select(item.second["Select"]), item.second["Operation"]));
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

	try
	{
		LOGGER << "Load electromagnetic fields solver";

		using namespace std::placeholders;

		Real ic2 = 1.0 / (mu0 * epsilon0);

		if (dict["FieldSolver"]["PML"])
		{
			auto solver = std::shared_ptr<PML<TM> >(new PML<TM>(mesh, dict["FieldSolver"]["PML"]));

			E_plus_CurlB = std::bind(&PML<TM>::NextTimeStepE, solver, _1, _2, _3, _4);

			B_minus_CurlE = std::bind(&PML<TM>::NextTimeStepB, solver, _1, _2, _3, _4);

		}
		else
		{
			E_plus_CurlB = [mu0 , epsilon0](Real dt, TE const & E , TB const & B, TE* pdE)
			{
				auto & dE=*pdE;
				LOG_CMD(dE += Curl(B)/(mu0 * epsilon0) *dt);
			};

			B_minus_CurlE = [](Real dt, TE const & E, TB const &, TB* pdB)
			{
				auto & dB=*pdB;
				LOG_CMD( dB -= Curl(E)*dt);
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

		auto solver = std::shared_ptr<ImplicitPushE<mesh_type>>(new ImplicitPushE<mesh_type>(mesh));
		Implicit_PushE = [solver] ( TE const & pE, TB const & pB, TParticles const&p, TE*dE)
		{	solver->NextTimeStep( pE,pB,p,dE);};
	}

}

template<typename TM>
std::string ExplicitEMContext<TM>::Save(std::string const & path, bool is_verbose) const
{
	GLOBAL_DATA_STREAM.OpenGroup(path);

	std::stringstream os;

	os

	<< description

	<< "\n, Grid = { \n" << mesh.Save(path ) << " \n} "
	;

	os

	<< "\n, Fields = {" << "\n"

	<< "\n, E = " << SAVE(E )

	<< "\n, B = " << SAVE(B )

	<< "\n, J = " << simpla::Save("J",Jext)

	<< "\n} ";

	if (particles_.size() > 0)
	{

		os << "\n , Particles = { \n";
		for (auto const & p : particles_)
		{
			os << p.first << " = { " << p.second->Save(path + "/" + p.first,is_verbose) << "\n},";
		}
		os << "\n} ";
	}

	return os.str();

}

template<typename TM>
void ExplicitEMContext<TM>::NextTimeStep()
{
	DEFINE_PHYSICAL_CONST
	;

	INFORM

	<< "[" << mesh.get_clock() << "]"

	<< "Simulation Time = " << (mesh.get_time() / CONSTANTS["s"]) << "[s]";

	Real dt = mesh.get_dt();

	//***********************************************************
	// Compute Cycle Begin
	//***********************************************************
	// E0 B0,
	LOG_CMD(Jext = J0);
	ExcuteCommands(commandToJ_);

	//   particle 0-> 1/2 . To n[1/2], J[1/2]
	for (auto &p : particles_)
	{
		if (!p.second->EnableImplicit())
		{
			p.second->NextTimeStepZero(E, B);

			auto const & Js = p.second->J();
			LOG_CMD(Jext += Js);
		}
	}

	LOG_CMD(B += dB * 0.5);	//  B(t=0 -> 1/2)
	ExcuteCommands(commandToB_);

	dE.clear();
	E_plus_CurlB(dt, E, B, &dE);	// dE += Curl(B)*dt

	LOG_CMD(dE -= Jext * (dt / epsilon0));

	//   particle 1/2 -> 1  . To n[1/2], J[1/2]
	Implicit_PushE(E, B, particles_, &dE);

	LOG_CMD(E += dE);	// E(t=0 -> 1)
	ExcuteCommands(commandToE_);

	dB.clear();
	B_minus_CurlE(dt, E, B, &dB);

	LOG_CMD(B += dB * 0.5);	//	B(t=1/2 -> 1)
	ExcuteCommands(commandToB_);

	mesh.NextTimeStep();

//***********************************************************
// Compute Cycle End
//***********************************************************
}

}
// namespace simpla

#endif /* EXPLICIT_EM_IMPL_H_ */
