/*
 * explicit_em_impl.h
 *
 *  Created on: 2013年12月29日
 *      Author: salmon
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
// Field expression
#include "../../src/fetl/field.h"
#include "../../src/fetl/ntuple.h"
#include "../../src/fetl/primitives.h"

// Data IO
#include "../../src/fetl/save_field.h"
#include "../../src/io/data_stream.h"

// Modeling
#include "../../src/modeling/material.h"
#include "../../src/utilities/geqdsk.h"

// Field solver
#include "../../src/modeling/constraint.h"
#include "../field_solver/pml.h"
#include "../field_solver/implicitPushE.h"

// Particle
#include "../../src/particle/particle_base.h"
#include "../particle_solver/particle_factory.h"

namespace simpla
{
template<typename TM>
struct ExplicitEMContext
{
public:

	typedef TM mesh_type;

	DEFINE_FIELDS (TM)

	typedef ExplicitEMContext<mesh_type> this_type;

	ExplicitEMContext();

	~ExplicitEMContext();

	template<typename TDict> void Load(TDict const & dict);

	template<typename OS> void Print(OS & os) const;

	void NextTimeStep();

	void Dump(std::string const & path = "") const;

	double CheckCourantDt() const;

public:

	mesh_type mesh;

	std::string description;

	Material<mesh_type> model_;

	bool isCompactStored_;

	Form<EDGE> E, dE;
	Form<FACE> B, dB;
	Form<VERTEX> phi; // electrostatic potential

	Form<EDGE> Jext;     // current density
	Form<EDGE> J0;     //background current density J0+Curl(B(t=0))=0
	Form<VERTEX> n; // charge density
	Form<VERTEX> n0; // charge density

	typedef decltype(E) TE;
	typedef decltype(B) TB;
	typedef decltype(Jext) TJ;

	typedef std::map<std::string, std::shared_ptr<ParticleBase<mesh_type> > > TParticles;

	std::function<void(Real, TE const &, TB const &, TE*)> PushE;

	std::function<void(Real, TE const &, TB const &, TB*)> PushB;

	std::function<void(Real, TE const &, TB const &, TParticles const&, TE*)> AddCurrent;

	void ApplyConstraintToE(TE* pE)
	{
		if (constraintToE_.size() > 0)
		{
			LOGGER << "Apply Constraint to E";
			for (auto const & foo : constraintToE_)
			{
				foo(pE);
			}
		}
	}
	void ApplyConstraintToB(TB* pB)
	{
		if (constraintToB_.size() > 0)
		{
			LOGGER << "Apply Constraint to B";
			for (auto const & foo : constraintToB_)
			{
				foo(pB);
			}
		}
	}
	void ApplyConstraintToJ(TJ* pJ)
	{
		if (constraintToJ_.size() > 0)
		{
			LOGGER << "Apply Constraint to J";
			for (auto const & foo : constraintToJ_)
			{
				foo(pJ);
			}
		}
	}

private:

	std::list<std::function<void(TE*)> > constraintToE_;

	std::list<std::function<void(TB*)> > constraintToB_;

	std::list<std::function<void(TJ*)> > constraintToJ_;

	std::map<std::string, std::shared_ptr<ParticleBase<mesh_type>>>particles_;

}
;

template<typename TM>
ExplicitEMContext<TM>::ExplicitEMContext() :
		isCompactStored_(true), model_(mesh),

		E(mesh), B(mesh), Jext(mesh), J0(mesh), dE(mesh), dB(mesh), n(mesh), n0(
				mesh), phi(mesh)
{
}

template<typename TM>
ExplicitEMContext<TM>::~ExplicitEMContext()
{
}
template<typename TM> template<typename TDict>
void ExplicitEMContext<TM>::Load(TDict const & dict)
{

	LOGGER << "Load ExplicitEMContext ";

	description = "Description = \""
			+ dict["Description"].template as<std::string>() + "\"\n";

	LOGGER << description;

	mesh.Load(dict["Grid"]);

	Form<VERTEX> ne0(mesh);
	Form<VERTEX> Te0(mesh);
	Form<VERTEX> Ti0(mesh);

	E.Clear();
	B.Clear();
	Jext.Clear();
	n.Clear();

	dB.Clear();
	dE.Clear();
	J0.Clear();
	n0.Clear();
	if (dict["Model"])
	{
		model_.Update();

		if (dict["Model"]["GFile"])
		{

			GEqdsk geqdsk(dict["Model"]["GFile"].template as<std::string>());

			nTuple<3, Real> xmin, xmax;

			xmin[0] = geqdsk.GetMin()[0];
			xmin[1] = geqdsk.GetMin()[1];
			xmin[2] = 0;
			xmax[0] = geqdsk.GetMax()[0];
			xmax[1] = geqdsk.GetMax()[1];
			xmax[2] = 0;

			mesh.SetExtent(xmin, xmax);

			mesh.Update();

			model_.Add("Plasma", geqdsk.Boundary());
			model_.Add("Vacuum", geqdsk.Limiter());
			model_.Update();

			geqdsk.Save(std::cout);

			B.Clear();

			for (auto s : mesh.GetRange(FACE))
			{
				auto x = mesh.CoordinatesToCartesian(mesh.GetCoordinates(s));
				B[s] = mesh.template Sample<FACE>(Int2Type<FACE>(), s,
						geqdsk.B(x[0], x[1]));

			}

			ne0.Clear();
			Te0.Clear();
			Ti0.Clear();

			for (auto s : model_.template SelectCell<VERTEX>("Plasma"))
			{
				auto x = mesh.CoordinatesToCartesian(mesh.GetCoordinates(s));
				auto p = geqdsk.psi(x[0], x[1]);

				ne0[s] = geqdsk.Profile("ne", p);
				Te0[s] = geqdsk.Profile("Te", p);
				Ti0[s] = geqdsk.Profile("Ti", p);

			}

			J0 = Curl(B) / mesh.constants()["permeability of free space"];

			description = description + "\n GEqdsk ID:" + geqdsk.Description();

			LOGGER << simpla::Dump(ne0, "ne", false);
			LOGGER << simpla::Dump(Te0, "Te", false);
			LOGGER << simpla::Dump(Ti0, "Ti", false);
		}

	}

	if (mesh.CheckCourantDt() < mesh.GetDt())
	{
		mesh.SetDt(mesh.CheckCourantDt());
	}

	LOG_CMD(LoadField(dict["InitValue"]["E"], &E));

	LOG_CMD(LoadField(dict["InitValue"]["B"], &B));

	LOG_CMD(LoadField(dict["InitValue"]["J"], &Jext));

	LOG_CMD(LoadField(dict["InitValue"]["ne"], &ne0));

	LOG_CMD(LoadField(dict["InitValue"]["Te"], &Te0));

	LOG_CMD(LoadField(dict["InitValue"]["Ti"], &Ti0));

	bool enableImplicitPushE = false;

	bool enablePML = false;

	DEFINE_PHYSICAL_CONST(mesh.constants());

	if (dict["Particles"])
	{
		LOGGER << "Load Particles";
		for (auto const &opt : dict["Particles"])
		{

			auto key = opt.first.template as<std::string>("unnamed");

			auto p = CreateParticle<mesh_type>(
					opt.second["Type"].template as<std::string>("Default"),
					mesh, opt.second, ne0, Te0);

			if (p != nullptr)
				particles_.emplace(key, p);
		}
	}

	if (dict["Constraints"])
	{
		LOGGER << "Load Constraints";
		for (auto const & item : dict["Constraints"])
		{

			auto dof = item.second["DOF"].template as<std::string>();

			LOGGER << "Add constraint to " << dof;

			if (dof == "E")
			{
				constraintToE_.push_back(
						CreateConstraint<TE>(model_, item.second));
			}
			else if (dof == "B")
			{
				constraintToB_.push_back(
						CreateConstraint<TB>(model_, item.second));
			}
			else if (dof == "J")
			{
				constraintToJ_.push_back(
						CreateConstraint<TJ>(model_, item.second));
			}
			else
			{
				//TODO Add particles constraints
				UNIMPLEMENT2("Unknown Constraints!!");
				continue;
			}

		}
	}

	if (dict["FieldSolver"])
	{
		auto dict_ = dict["FieldSolver"];
		LOGGER << "Load Electromagnetic fields solver";

		using namespace std::placeholders;

		Real ic2 = 1.0 / (mu0 * epsilon0);

		if (dict["FieldSolver"]["PML"])
		{
			auto solver = std::shared_ptr<PML<TM> >(
					new PML<TM>(mesh, dict["FieldSolver"]["PML"]));

			PushE = std::bind(&PML<TM>::NextTimeStepE, solver, _1, _2, _3, _4);

			PushB = std::bind(&PML<TM>::NextTimeStepB, solver, _1, _2, _3, _4);

		}
		else
		{
			PushE =
					[mu0 , epsilon0](Real dt, TE const & E , TB const & B, TE* pdE)
					{
						auto & dE=*pdE;
						LOG_CMD(dE += Curl(B)/(mu0 * epsilon0) *dt);
					};

			PushB = [](Real dt, TE const & E, TB const &, TB* pdB)
			{
				auto & dB=*pdB;
				LOG_CMD( dB -= Curl(E)*dt);
			};
		}

	}

	if (!enableImplicitPushE)
	{
		AddCurrent =
				[mu0 , epsilon0](Real dt, TE const & E , TB const & B, TParticles const & sp, TE* pdE)
				{
					auto & dE=*pdE;
					for(auto const &p :sp)
					{
						LOG_CMD(dE -= (dt/epsilon0)*p.second->J);
					}
				};
	}
	else
	{
		AddCurrent = &ImplicitPushE<TE, TB, TParticles>;
	}

}

template<typename TM>
template<typename OS>
void ExplicitEMContext<TM>::Print(OS & os) const
{

	os

	<< "\n"

	<< description << "\n"

	<< "Grid = { \n" << mesh << " \n}\n"

	<< "InitValue={" << "\n"

	<< "	E = " << simpla::Dump(E, "E", false) << ",\n"

	<< "	B = " << simpla::Dump(B, "B", false) << ",\n"

	<< "	J = " << simpla::Dump(Jext, "J", false) << ",\n"

	<< "	J0 = " << simpla::Dump(J0, "J0", false) << ",\n"

	<< "}" << "\n";

	if (particles_.size() > 0)
	{
		os << "Particles = { \n";
		for (auto const & p : particles_)
		{
			os << p.first << " = {";
			p.second->Print(os);
			os << "},";
		}
		os << "\n}\n";
	}

}
template<typename OS, typename TM>
OS &operator<<(OS & os, ExplicitEMContext<TM> const& self)
{
	self.Print(os);
	return os;
}
template<typename TM>
void ExplicitEMContext<TM>::NextTimeStep()
{
	Real dt = mesh.GetDt();

	mesh.NextTimeStep();

	DEFINE_PHYSICAL_CONST(mesh.constants());

	VERBOSE

	<< "Simulation Time = "

	<< (mesh.GetTime() / mesh.constants()["s"]) << "[s]"

	<< " dt = " << (dt / mesh.constants()["s"]) << "[s]";

	//************************************************************
	// Compute Cycle Begin
	//************************************************************

	dB.Clear();

	PushB(dt, E, B, &dB);

	//  B(t=1/2 -> 1)
	LOG_CMD(B += dB * 0.5);

	ApplyConstraintToB(&B);

	//   x=-1/2 -> 1/2 , v=0 -> 1
	for (auto &p : particles_)
	{
		p.second->NextTimeStep(dt, E, B);
	}

	// B(t=0 -> 1/2)
	LOG_CMD(B += dB * 0.5);

	ApplyConstraintToB(&B);

	LOG_CMD(Jext = J0);

	ApplyConstraintToJ(&Jext);

	dE.Clear();

	LOG_CMD(dE -= Jext * (dt / epsilon0));

	AddCurrent(dt, E, B, particles_, &dE);

	// dE += Curl(B)*dt
	PushE(dt, E, B, &dE);

	// E(t=0  -> 1/2  )
	LOG_CMD(E += dE);

	ApplyConstraintToE(&E);

//************************************************************
// Compute Cycle End
//************************************************************
}
template<typename TM>
void ExplicitEMContext<TM>::Dump(std::string const & path) const
{
	GLOBAL_DATA_STREAM.OpenGroup(path);

	LOGGER << DUMP(E);
	LOGGER << DUMP(B);
	LOGGER << DUMP(Jext);

	for (auto &p : particles_)
	{
		p.second->Dump( p.first,true);
	}

}
}
// namespace simpla

#endif /* EXPLICIT_EM_IMPL_H_ */
