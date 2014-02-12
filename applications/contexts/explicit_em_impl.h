/*
 * explicit_em_impl.h
 *
 *  Created on: 2013年12月29日
 *      Author: salmon
 */

#ifndef EXPLICIT_EM_IMPL_H_
#define EXPLICIT_EM_IMPL_H_

#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <valarray>
#include <vector>
#include <map>
#include <unordered_map>

#include "../../src/engine/basecontext.h"

#include "../../src/fetl/fetl.h"

#include "../../src/fetl/load_field.h"
#include "../../src/fetl/save_field.h"

#include "../../src/mesh/field_convert.h"

#include "../../src/mesh/media_tag.h"

#include "../../src/fetl/field_function.h"

#include "../../src/utilities/log.h"
#include "../../src/utilities/lua_state.h"
#include "../../src/io/data_stream.h"

#include "../../src/particle/particle.h"

#include "../../src/particle/particle_factory.h"

#include "../solver/electromagnetic/solver.h"

#include "../../src/utilities/geqdsk.h"

namespace simpla
{

template<typename ...Args>
void NullFunction(Args const & ...)
{
}

template<typename TJ, typename TCfg, typename TM>
void CreateCurrentSrc(TCfg const & cfg, TM const & mesh, std::function<void(Real, TJ*)> *res)
{

	*res = [](Real, TJ*)
	{

	};
}

template<typename TM>
struct ExplicitEMContext: public BaseContext
{
public:
	typedef BaseContext base_type;
	typedef TM mesh_type;
	typedef LuaObject configure_type;

	DEFINE_FIELDS (TM)

	Real time_;
public:
	typedef ExplicitEMContext<TM> this_type;

	ExplicitEMContext();

	~ExplicitEMContext();

	void Load(configure_type const & cfg);

	std::ostream & Save(std::ostream & os) const;

	void NextTimeStep(double dt);

	void DumpData(std::string const & path = "") const;

	Real GetTime()
	{
		return time_;
	}

public:

	mesh_type mesh;

	std::string description;

	bool isCompactStored_;

	Form<EDGE> E, dE;
	Form<FACE> B, dB;
	Form<EDGE> J;

	typedef decltype(E) TE;
	typedef decltype(B) TB;
	typedef decltype(J) TJ;

	std::function<void(Real, TE const &, TB const &, TE*)> CalculatedE;

	std::function<void(Real, TE const &, TB const &, TB*)> CalculatedB;

	std::function<void(TE*)> ApplyBoundaryConditionToE;

	std::function<void(TB*)> ApplyBoundaryConditionToB;

	std::function<void(Real, TJ*)> ApplyCurrentSrcToJ;

	typedef ParticleWrap<TE, TB, TJ> ParticleType;

	std::map<std::string, ParticleType> particles_;

}
;

template<typename TM>
ExplicitEMContext<TM>::ExplicitEMContext()
		: isCompactStored_(true), time_(0), E(mesh), B(mesh), J(mesh), dE(mesh), dB(mesh)
{

	CalculatedE = [](Real dt, TE const & , TB const & pB, TE* pdE)
	{	LOG_CMD(*pdE += Curl(pB)*dt);};

	CalculatedB = [](Real dt, TE const & pE, TB const &, TB* pdB)
	{	LOG_CMD(*pdB -= Curl(pE)*dt);};

	ApplyCurrentSrcToJ = [](Real, TJ*)
	{};

	ApplyBoundaryConditionToE = [ ]( TE * pE)
	{};

	ApplyBoundaryConditionToB = [ ]( TB * pB)
	{};
}

template<typename TM>
ExplicitEMContext<TM>::~ExplicitEMContext()
{
}
template<typename TM>
void ExplicitEMContext<TM>::Load(LuaObject const & cfg)
{
	description = cfg["Description"].as<std::string>();

	mesh.Deserialize(cfg["Grid"]);
	B.Clear();
	J.Clear();
	E.Clear();
	if (cfg["GFile"])
	{

		GEqdsk geqdsk(cfg["GFile"].as<std::string>());

		mesh.SetExtent(geqdsk.GetMin(), geqdsk.GetMax());

		mesh.Update();

		RForm<EDGE> B1(mesh);

		B1.Clear();

		mesh.SerialTraversal(EDGE,

		[&](typename mesh_type::index_type s,typename mesh_type::coordinates_type const &x)
		{
			B1[s] = mesh.template GetWeightOnElement<FACE>(geqdsk.B(x),s);
		});

		MapTo(B1, &B);

	}

	if (cfg["InitValue"])
	{
		auto init_value = cfg["InitValue"];

		LOGGER << "Load E";
		LoadField(init_value["E"], &E);
		LOGGER << "Load B";
		LoadField(init_value["B"], &B);
		LOGGER << "Load J";
		LoadField(init_value["J"], &J);

		LOGGER << "Load Initial Fields." << DONE;
	}

	CreateEMSolver(cfg["FieldSolver"], mesh, &CalculatedE, &CalculatedB);

	CreateCurrentSrc(cfg["CurrentSrc"], mesh, &ApplyCurrentSrcToJ);

	if (mesh.tags() && cfg["Interface"])
	{

		for (auto const & item : cfg["Interface"])
		{
			std::shared_ptr<std::list<index_type> > edge(new std::list<index_type>);

			if (item.second["Type"].as<std::string>() == "PEC")
			{
				tag_type in = mesh.tags().GetTagFromString(item.second["In"].as<std::string>());

				tag_type out = mesh.tags().GetTagFromString(item.second["Out"].as<std::string>());

				auto selector = mesh.tags().template SelectInterface<EDGE>(in, out);

				this->mesh.SerialTraversal(EDGE, [&](index_type s)
				{	if(selector(s)) edge->push_back(s);});

				ApplyBoundaryConditionToE = [&]( TE * pE)
				{	for(auto s:*edge) (*pE)[s]=0;};

			};

		};
	}
	else
	{
		UNIMPLEMENT << "Unknown Interface type!"
//				<< " [function = " << function << " type= " << type << " object =" << object << " ]"
		        ;
	}

	LOGGER << "Setup interface" << DONE;

	for (auto const &opt : cfg["Particles"])
	{
		particles_.emplace(
		        std::make_pair(opt.first.template as<std::string>(),
		                CreateParticle<Mesh, TE, TB, TJ>(mesh, opt.second)));
	}

	LOGGER << ">>>>>>> Initialization  Complete! <<<<<<<< ";

}

template<typename TM>
std::ostream & ExplicitEMContext<TM>::Save(std::ostream & os) const
{

	os << "Description=\"" << description << "\" \n";

	os << "Grid = " << mesh << "\n";

//	os << " FieldSolver={ \n";
//
//	if (cold_fluid_ != nullptr)
//		os << *cold_fluid_ << ",\n";
//
//	if (pml_ != nullptr)
//		os << *pml_ << ",\n";
//
//	os << "} \n";
//
//	if (particles_ != nullptr)
//		os << *particles_ << "\n";
//
//	os << "Function={";
//	for (auto const & p : field_boundary_)
//	{
//		os << "\"" << p.first << "\",\n";
//	}
	os << "}\n"

	<< "Fields={" << "\n"

	<< "	E = " << Dump(E, "E", false) << ",\n"

	<< "	B = " << Dump(B, "B", false) << ",\n"

	<< "	J = " << Dump(J, "J", false) << ",\n"

	<< "}" << "\n"

	;
	return os;
}
template<typename TM>
void ExplicitEMContext<TM>::NextTimeStep(double dt)
{
	dt = std::isnan(dt) ? mesh.GetDt() : dt;

	time_ += dt;

	if (!mesh.CheckCourant(dt))
		VERBOSE << "dx/dt > c, Courant condition is violated! ";

	DEFINE_PHYSICAL_CONST(mesh.constants());

	LOGGER

	<< "Simulation Time = "

	<< (GetTime() / mesh.constants()["s"]) << "[s]"

	<< " dt = " << (dt / mesh.constants()["s"]) << "[s]";

//************************************************************
// Compute Cycle Begin
//************************************************************

	ApplyCurrentSrcToJ(GetTime(), &J);

	dE.Clear();

// dE = Curl(B)*dt
	CalculatedE(dt, E, B, &dE);

	LOG_CMD(dE -= J / epsilon0 * dt);

// E(t=0  -> 1/2  )
	LOG_CMD(E += dE * 0.5);

	ApplyBoundaryConditionToE(&E);

	for (auto &p : particles_)
	{
		p.second.NextTimeStep(dt, E, B);	// particle(t=0 -> 1)
	}

//  E(t=1/2  -> 1)
	LOG_CMD(E += dE * 0.5);

	ApplyBoundaryConditionToE(&E);

	Form<2> dB(mesh);

	dB.Clear();

	CalculatedB(dt, E, B, &dB);

//  B(t=1/2 -> 1)
	LOG_CMD(B += dB * 0.5);

	ApplyBoundaryConditionToB(&B);

	J.Clear();

	for (auto &p : particles_)
	{
		// B(t=0) E(t=0) particle(t=0) Jext(t=0)
		p.second.Collect(&J, E, B);
	}

// B(t=0 -> 1/2)
	LOG_CMD(B += dB * 0.5);

	ApplyBoundaryConditionToB(&B);

//************************************************************
// Compute Cycle End
//************************************************************

}
template<typename TM>
void ExplicitEMContext<TM>::DumpData(std::string const & path) const
{
	GLOBAL_DATA_STREAM.OpenGroup(path);

	LOGGER << DUMP(E);
	LOGGER << DUMP(B);
	LOGGER << DUMP(J);

}
}
// namespace simpla

#endif /* EXPLICIT_EM_IMPL_H_ */
