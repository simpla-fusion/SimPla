/*
 * cold_fluid.h
 *
 *  Created on: 2013年11月13日
 *      Author: salmon
 */

#ifndef COLD_FLUID_H_
#define COLD_FLUID_H_

#include "../../../src/fetl/fetl.h"
#include "../../../src/engine/fieldsolver.h"
#include "../../../src/utilities/load_field.h"
#include "../../../src/utilities/log.h"
#include "../../../src/utilities/pretty_stream.h"
namespace simpla
{

template<typename TM>
class ColdFluidEM
{
public:
	DEFINE_FIELDS(TM)

	typedef Mesh mesh_type;

	typedef ColdFluidEM<mesh_type> this_type;

	mesh_type const & mesh;
private:

	struct Species
	{
		Real m;
		Real Z;
		Form<0> n;
		VectorForm<0> J;

		Species(Real pm, Real pZ, mesh_type const &mesh)
				: m(pm), Z(pZ), n(mesh), J(mesh)
		{
		}
		~Species()
		{
		}

	};
	std::map<std::string, std::shared_ptr<Species>> sp_list_;
	VectorForm<0> Ev;
public:

	ColdFluidEM(mesh_type const & pmesh)
			: mesh(pmesh), Ev(pmesh)
	{
	}

	~ColdFluidEM()
	{
	}

	inline bool IsEmpty()
	{
		return sp_list_.empty();
	}
	inline bool empty() // STL style
	{
		return sp_list_.empty();
	}
	void Deserialize(LuaObject const&cfg);
	std::ostream & Serialize(std::ostream & os) const;

	template<typename TE, typename TB> inline
	void NextTimeStep(Real dt, TE const &dE, TB const &B0, TE *E);

	void DumpData() const;

}
;
template<typename TM>
template<typename TE, typename TB> inline
void ColdFluidEM<TM>::NextTimeStep(Real dt, TE const &E, TB const &B0, TE *dE)
{
	if (sp_list_.empty())
		return;

	const double mu0 = mesh.constants["permeability of free space"];
	const double epsilon0 = mesh.constants["permittivity of free space"];
	const double speed_of_light = mesh.constants["speed of light"];
	const double proton_mass = mesh.constants["proton mass"];
	const double elementary_charge = mesh.constants["elementary charge"];

	LOGGER << "Push Cold Fluid.";
	if (Ev.empty())
		MapTo(E, &Ev);

	VectorForm<0> K(mesh);

	Form<0> a(mesh);
	Form<0> b(mesh);
	Form<0> c(mesh);

	Form<0> BB(mesh);

	BB = Dot(B0, B0);

	VectorForm<0> B0v(mesh);

	MapTo(B0, &B0v);

	a.Fill(1.0);
	b.Fill(0);
	c.Fill(0);

	VectorForm<0> dEv(mesh);

	MapTo(*dE, &dEv);

	Ev += dEv * 0.5;

	K = Ev;

	for (auto &v : sp_list_)
	{

		auto & ns = v.second->n;
		auto & Js = v.second->J;
		auto ms = v.second->m * proton_mass;
		auto Zs = v.second->Z * elementary_charge;

		Real as = 2.0 * ms / (dt * Zs);

		a += ns * Zs / as * (0.5 * dt) / epsilon0;

		b += ns * Zs / (BB + as * as) * (0.5 * dt) / epsilon0;

		c += ns * Zs / ((BB + as * as) * as) * (0.5 * dt) / epsilon0;

		auto K_ = Cross(Js, B0v) + (Ev * ns) * Zs;

		K -= (Js

		+ (K_ / as

		+ Cross(K_, B0v) / (BB + as * as)

		+ Cross(Cross(K_, B0v), B0v) / (as * (BB + as * as)))

		) * (0.5 * dt / epsilon0);

	}

	Ev = K / a

	- Cross(K, B0v) * b / ((c * BB - a) * (c * BB - a) + b * b * BB)

	- Cross(Cross(K, B0v), B0v) * (-c * c * BB + c * a - b * b)

	/ (a * ((c * BB - a) * (c * BB - a) + b * b * BB))

	;

	for (auto &v : sp_list_)
	{
		auto & ns = v.second->n;
		auto & Js = v.second->J;
		auto ms = v.second->m * proton_mass;
		auto Zs = v.second->Z * elementary_charge;

		Real as = 2.0 * ms / (dt * Zs);

		auto K_ = Cross(Js, B0v) + Ev * ns * Zs;

		Js = K_ / as

		- Cross(K_, B0v) / (BB + as * as)

		- Cross(Cross(K_, B0v), B0v) / (as * (BB + as * as));

	}

	Ev += dEv * (0.5);

	MapTo(Ev, dE);

	*dE -= E;

}

template<typename TM>
inline void ColdFluidEM<TM>::Deserialize(LuaObject const&cfg)
{
	if (cfg.empty())
		return;

	for (auto const & p : cfg)
	{

		std::string key;

		if (!p.first.is_number())
		{
			key = p.first.as<std::string>();
		}
		else
		{
			p.second.GetValue("Name", &key);
		}

		std::shared_ptr<Species> sp(
		        new Species(p.second["m"].template as<Real>(1.0), p.second["Z"].template as<Real>(1.0), mesh));

		sp->n.Init();
		sp->J.Init();

		if (!LoadField(p.second["n"], &(sp->n)))
			sp->n.Fill(0);

		if (!LoadField(p.second["J"], &(sp->J)))
			sp->J.Fill(0);

		sp_list_.emplace(std::make_pair(key, sp));

	}

	LOGGER << "Load Cold Fluid solver" << DONE;

}

template<typename TM>
void ColdFluidEM<TM>::DumpData() const
{
	GLOBAL_DATA_STREAM.OpenGroup("/DumpData");

	for (auto const & p : sp_list_)
	{
		LOGGER << "Dump " << "n_" + p.first << " to "
		        << Data(p.second->n.data(), "n_" + p.first, p.second->n.GetShape(), true);

		LOGGER << "Dump " << "J_" + p.first << " to "
		        << Data(p.second->J.data(), "J_" + p.first, p.second->J.GetShape(), true);
	}
}

template<typename TM>
std::ostream & ColdFluidEM<TM>::Serialize(std::ostream & os) const
{
	os << "-- Cold Fluid -------------------\n";
	os << "  ColdFluid={\n";

	for (auto const & p : sp_list_)
	{
		os << "\t" << p.first

		<< " = { " << " m =" << p.second->m << "," << " Z =" << p.second->Z << ",\n"

		<< "\t n0 = " << Data(p.second->n.data(), "n_" + p.first, p.second->n.GetShape()) << "\n"

		<< "\t J0 = " << Data(p.second->J.data(), "J_" + p.first, p.second->J.GetShape()) << "\n"

		<< "\t},\n";
	}
	os << "}\n";

	return os;
}

template<typename TM>
inline std::ostream & operator<<(std::ostream & os, ColdFluidEM<TM> const &self)
{
	return self.Serialize(os);
}

}  // namespace simpla

#endif /* COLD_FLUID_H_ */
