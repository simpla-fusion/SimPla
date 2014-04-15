/*
 * implicit_pic.h
 *
 *  Created on: 2014年4月9日
 *      Author: salmon
 */

#ifndef IMPLICIT_PIC_H_
#define IMPLICIT_PIC_H_

namespace simpla
{

#include <map>
#include <string>
#include <utility>
#include <memory>

#include "../../../src/fetl/fetl.h"
#include "../../../src/fetl/load_field.h"
#include "../../../src/fetl/save_field.h"
#include "../../../src/utilities/log.h"
#include "../../../src/utilities/pretty_stream.h"
#include "../../../src/physics/physical_constants.h"

#include "../../../src/engine/fieldsolver.h"
#include "../../../src/mesh/field_convert.h"
namespace simpla
{

template<typename TM>
class ImplicitPIC
{
public:
	DEFINE_FIELDS(TM)

	typedef Mesh mesh_type;

	typedef ImplicitPIC<mesh_type> this_type;

	mesh_type const & mesh;

private:

	struct Species
	{
		Real m;
		Real q;
		RForm<0> n;
		VectorForm<0> J;

		Species(Real pm, Real pZ, mesh_type const &mesh)
				: m(pm), q(pZ), n(mesh), J(mesh)
		{
			n.Clear();
			J.Clear();
		}
		~Species()
		{
		}

	};
	std::map<std::string, std::shared_ptr<Species>> sp_list_;
	RVectorForm<0> B0, Ev;
	RForm<0> BB;
	RForm<0> a;
	RForm<0> b;
	RForm<0> c;

	bool enableNonlinear_;
public:

	ImplicitPIC(mesh_type const & pmesh)
			: mesh(pmesh), B0(pmesh), BB(pmesh), enableNonlinear_(false),

			Ev(mesh), a(pmesh), b(pmesh), c(pmesh), dt_(mesh.GetDt())
	{
	}

	~ImplicitPIC()
	{
	}

	inline bool empty() const  // STL style
	{
		return sp_list_.empty();
	}

	template<typename TDict, typename ...Args>
	void Load(TDict const&dict, RForm<0> const & ne, Args const & ...);

	template<typename OS>
	void Print(OS & os) const;

	void Dump(std::string const & path) const;

	template<typename TE, typename TB>
	void NextTimeStepE(Real dt, TE const &E, TB const &B0, TE *dE);

private:

	Real dt_;

}
;
template<typename TM>
template<typename TE, typename TB>
void ImplicitPIC<TM>::NextTimeStepE(Real dt, TE const &E, TB const &B, TE *pdE)
{

	DEFINE_PHYSICAL_CONST(mesh.constants());

	if (sp_list_.empty())
		return;

	LOGGER << "Push E: Cold Fluid. [ Species Number=" << sp_list_.size() << "]";
//	VERBOSE << "Nonlinear is " << ((enableNonlinear_) ? "opened" : "closed") << ".";

	TE & dE = *pdE;

	if (BB.empty() /*|| enableNonlinear_*/)
	{
		B0 = MapTo<VERTEX>(B);
		BB = Dot(B0, B0);
	}
	if (a.empty() || dt_ != dt)
	{
		dt_ = dt;
		a.Clear();
		b.Clear();
		c.Clear();

		for (auto &v : sp_list_)
		{
			auto & ns = v.second->n;
			auto & Js = v.second->J;
			Real ms = v.second->m;
			Real qs = v.second->q;
			Real as = (dt * qs) / (2.0 * ms);

			a += ns * qs * as / (BB * as * as + 1);
			b += ns * qs * as * as / (BB * as * as + 1);
			c += ns * qs * as * as * as / (BB * as * as + 1);
		}

		a *= 0.5 * dt / epsilon0;
		b *= 0.5 * dt / epsilon0;
		c *= 0.5 * dt / epsilon0;
		a += 1;
	}

	VectorForm<0> Q(mesh);
	VectorForm<0> K(mesh);

	Q.Clear();
	K.Clear();

	Ev = MapTo<VERTEX>(E + dE * 0.5);

	for (auto &v : sp_list_)
	{
		auto & ns = v.second->n;
		auto & Js = v.second->J;
		Real ms = v.second->m;
		Real qs = v.second->q;
		Real as = (dt * qs) / (2.0 * ms);

		Q -= Js;
		K = Cross(Js, B0) * as + Ev * ns * qs * as + Js;
		Js = (K + Cross(K, B0) * as + B0 * (Dot(K, B0) * as * as)) / (BB * as * as + 1);
		Q -= Js;
	}

	Q *= 0.5 * dt / epsilon0;
	Q += Ev;
	Ev = (Q * a - Cross(Q, B0) * b + B0 * (Dot(Q, B0) * (b * b - c * a) / (a + c * BB))) / (b * b * BB + a * a);

	for (auto &v : sp_list_)
	{
		auto & ns = v.second->n;
		auto & Js = v.second->J;
		auto ms = v.second->m;
		auto qs = v.second->q;

		Real as = (dt * qs) / (2.0 * ms);
		Js += (Ev + Cross(Ev, B0) * as + B0 * (Dot(Ev, B0) * as * as)) * (as * ns * qs) / (BB * as * as + 1);

	}

	dE = (MapTo<EDGE>(Ev) - E) + dE * 0.5;

	LOGGER << DONE;

	Dump("/DumpData");

}

template<typename TM>
template<typename TDict, typename ...Args>
void ImplicitPIC<TM>::Load(TDict const&dict, RForm<0> const & ne, Args const & ...)
{
	if (!dict)
		return;

	LOGGER << "Create ImplicitPIC solver";

	enableNonlinear_ = dict["Nonlinear"].template as<bool>(false);

	for (auto const & p : dict["Species"])
	{
		std::string key;

		if (!p.first.is_number())
		{
			key = p.first.template as<std::string>();
		}
		else
		{
			p.second.GetValue("Name", &key);
		}

		std::shared_ptr<Species> sp(
		        new Species(p.second["Mass"].template as<Real>(1.0), p.second["Charge"].template as<Real>(1.0), mesh));

		if (ne.empty())
		{
			LoadField(p.second["Density"], &(sp->n));
		}
		else
		{
			sp->n = ne;
			if (p.second["Density"].is_number())
				sp->n *= p.second["Density"].template as<Real>(1.0);
		}

		LoadField(p.second["Current"], &(sp->J));

		sp_list_.emplace(key, sp);

	}

	LOGGER << DONE;

}

template<typename TM>
void ImplicitPIC<TM>::Dump(std::string const & path) const
{
	GLOBAL_DATA_STREAM.OpenGroup(path);

	for (auto const & p : sp_list_)
	{
		LOGGER << simpla::Dump(p.second->n, "n_" + p.first, true);
		LOGGER << simpla::Dump(p.second->J, "J_" + p.first, true);
	}
}

template<typename TM>
template<typename OS>
void ImplicitPIC<TM>::Print(OS & os) const
{
	os << "ColdFluid = { Nonlinear = " << std::boolalpha << enableNonlinear_ << ",\n"

	<< "  Species = { \n ";

	for (auto const & p : sp_list_)
	{
		os << "\n\t" << p.first

		<< " = { " << " Mass =" << p.second->m << "," << " Charge =" << p.second->q << ",\n"

		<< "\t n0 = " << simpla::Dump(p.second->n.data(), "n_" + p.first, p.second->n.GetShape(), false) << "\n"

		<< "\t J0 = " << simpla::Dump(p.second->J.data(), "J_" + p.first, p.second->J.GetShape(), false) << "\n"

		<< "\t},\n";
	}
	os << "\t}\n}";

}

template<typename OS, typename TM>
OS &operator<<(OS & os, ImplicitPIC<TM> const& self)
{
	self.Save(os);
	return os;
}
}  // namespace simpla

#endif /* IMPLICIT_PIC_H_ */
