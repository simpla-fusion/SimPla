/*
 * pic_engine_implicit.h
 *
 * \date  2014-4-10
 *      \author  salmon
 */

#ifndef PIC_ENGINE_IMPLICIT_H_
#define PIC_ENGINE_IMPLICIT_H_

#include <string>
#include "../../src/physics/physical_constants.h"
#include "../../src/utilities/primitives.h"
#include "../../src/utilities/ntuple.h"
#include "../../src/utilities/log.h"

namespace simpla
{

template<typename TM, typename TS = Real>
struct PICEngineImplicit
{

public:
	typedef PICEngineImplicit<TM, TS> this_type;

	typedef TM mesh_type;
	typedef TS scalar_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef nTuple<8, Real> storage_value_type;

	struct Point_s
	{
		coordinates_type x;
		Vec3 v;
		Real f;
		scalar_type w;

	};

private:
	Real m_, cmr_, q_, q_kT_;
public:
	mesh_type const &mesh;

public:
	PICEngineImplicit(mesh_type const &pmesh) :
			mesh(pmesh), m_(1.0), q_(1.0), cmr_(1.0), q_kT_(1.0)
	{
	}
	~PICEngineImplicit()
	{
	}

	static std::string TypeName()
	{
		return "DeltaF";
	}
	static std::string GetTypeAsString()
	{
		return "DeltaF";
	}

	Real GetMass() const
	{
		return m_;
	}

	Real GetCharge() const
	{
		return q_;
	}
	size_t GetAffectedRange() const
	{
		return 2;
	}
	template<typename TDict>
	void Load(TDict const &dict)
	{

		DEFINE_PHYSICAL_CONST

		m_ = dict["Mass"].template as<Real>(1.0);
		q_ = dict["Charge"].template as<Real>(1.0);
		cmr_ = q_ / m_;
		q_kT_ = q_ / (dict["Temperature"].template as<Real>(1.0) * boltzmann_constant);
		CHECK(q_kT_);
	}

	std::string Save(std::string const & path = "", bool is_verbose = false) const
	{
		std::stringstream os;

		DEFINE_PHYSICAL_CONST
		;

		os << "Engine = '" << GetTypeAsString() << "' "

		<< " , " << "Mass = " << m_

		<< " , " << "Charge = " << q_

		<< " , " << "Temperature = " << q_ / q_kT_ / elementary_charge << "* eV"

		;

		return os.str();
	}
	void Update()
	{
	}
	static Point_s DefaultValue()
	{
		Point_s p;
		p.f = 1.0;
		p.w = 0.0;
		return std::move(p);
	}

	template<typename TB, typename TE, typename ... Others> inline
	void NextTimeStep(Point_s * p, Real dt, TE const &fE, TB const & fB, Others const &...others) const
	{

		// $ x_{1/2} - x_{0} = v_0   \Delta t /2$
		p->x += p->v * dt * 0.5;

		auto B = real(fB(p->x));
		auto E = real(fE(p->x));

		Vec3 v_;

		auto t = B * (cmr_ * dt * 0.5);

		p->v += E * (cmr_ * dt * 0.5);

		v_ = p->v + Cross(p->v, t);

		v_ = Cross(v_, t) / (Dot(t, t) + 1.0);

		p->v += v_;

		// FIXME miss one term E\cross B \cdot \Grad n
		// @NOTE Nonlinear delta-f
		auto a = (-Dot(fE(p->x), p->v) * q_kT_ * dt);
		p->w = (-a + (1 + 0.5 * a) * p->w) / (1 - 0.5 * a);

		p->v += v_;

		p->v += E * (cmr_ * dt * 0.5);

		// $ x_{1} - x_{1/2} = v_1   \Delta t /2$
		p->x += p->v * dt * 0.5;
//		BorisMethod(dt, cmr_, fE, fB, &(p->x), &(p->v));
////		// FIXME miss one term E\cross B \cdot \Grad n
//		auto a = (-Dot(fE(p->x), p->v) * q_kT_ * dt);
//		p->w = (-a + (1 + 0.5 * a) * p->w) / (1 - 0.5 * a);

	}

	template<typename TV, typename ... Others>
	inline typename std::enable_if<!is_ntuple<TV>::value, void>::type Scatter(Point_s const &p,
	        Field<mesh_type, VERTEX, TV>* n, Others const &... others) const
	{
		mesh.Scatter(n, std::make_tuple(p.x, p.f * p.w * q_));
	}

	template<unsigned int IFORM, typename TV, typename ...Others>
	inline void Scatter(Point_s const &p, Field<mesh_type, IFORM, TV>* J, Others const &... others) const
	{
		mesh.Scatter(J, std::make_tuple(p.x, p.v), p.w * q_);
	}

	static inline Point_s make_point(coordinates_type const & x, Vec3 const &v, Real f)
	{
		return std::move(Point_s( { x, v, f, 0 }));
	}

};

template<typename OS, typename TM, typename TS> OS&
operator<<(OS& os, typename PICEngineImplicit<TM, TS>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " , w=" << p.w << " }";

	return os;
}

} // namespace simpla

#endif /* PIC_ENGINE_IMPLICIT_H_ */
