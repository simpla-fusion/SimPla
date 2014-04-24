/*
 * pic_engine_ggauge.h
 *
 *  Created on: 2013年10月23日
 *      Author: salmon
 */

#ifndef PIC_ENGINE_GGAUGE_H_
#define PIC_ENGINE_GGAUGE_H_

#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>

#include "../../src/fetl/fetl.h"
#include "../../src/fetl/primitives.h"
#include "../../src/physics/physical_constants.h"
#include "../../src/physics/constants.h"
#include "../../src/utilities/utilities.h"

namespace simpla
{
template<typename TM, typename TS = Real, int NMATE = 8, typename Interpolator = typename TM::interpolator_type>
class PICEngineGGauge
{

public:
	typedef PICEngineGGauge<TM, TS, NMATE> this_type;
	typedef TM mesh_type;
	typedef TS scalar_type;
	typedef Interpolator interpolator_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
private:
	Real m_, q_, cmr_, T_, vT_;
	Real cosdq[NMATE], sindq[NMATE];
public:
	mesh_type const &mesh;

	struct Point_s
	{
		coordinates_type x;
		Vec3 v;
		Real f;
		nTuple<NMATE, scalar_type> w;

		static std::string DataTypeDesc()
		{
			std::ostringstream os;
			os

			<< "H5T_COMPOUND {          "

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"x\" : " << (offsetof(Point_s, x)) << ";"

			<< "   H5T_ARRAY { [3] H5T_NATIVE_DOUBLE}    \"v\" :  " << (offsetof(Point_s, v)) << ";"

			<< "   H5T_NATIVE_DOUBLE    \"f\" : " << (offsetof(Point_s, f)) << ";"

			<< "   H5T_ARRAY { [" << NMATE << "] H5T_NATIVE_DOUBLE}    \"w\" :  " << (offsetof(Point_s, w)) << ";"

			<< "}";

			return os.str();
		}
	};

	template<typename ...Args>
	PICEngineGGauge(mesh_type const &pmesh, Args const & ...args)
			: mesh(pmesh), m_(1.0), q_(1.0), cmr_(1.0), T_(1.0), vT_(1.0)
	{
		Load(std::forward<Args const &>(args)...);
	}

	~PICEngineGGauge()
	{
	}

	size_t GetAffectedRange() const
	{
		return 4;
	}
	static inline std::string TypeName()
	{
		return "GGauge" + ToString(NMATE);
	}
	static std::string GetTypeAsString()
	{
		return TypeName();
	}

	template<typename TDict, typename ...Others>
	void Load(TDict const &dict, Others const &...)
	{
		m_ = dict["Mass"].template as<Real>(1.0);
		q_ = dict["Charge"].template as<Real>(1.0);
		T_ = dict["Temperature"].template as<Real>(1.0);

		cmr_ = q_ / m_;

		vT_ = std::sqrt(2.0 * T_ / m_);

		constexpr Real theta = TWOPI / static_cast<Real>(NMATE);

		for (int i = 0; i < NMATE; ++i)
		{
			sindq[i] = std::sin(theta * i);
			cosdq[i] = std::cos(theta * i);
		}
	}

	std::string Dump(std::string const & path = "", bool is_verbose = false) const
	{
		std::stringstream os;
		DEFINE_PHYSICAL_CONST(mesh.constants());

		os << "Engine = '" << GetTypeAsString() << "' "

		<< " , " << "Mass = " << m_ / proton_mass << " * m_p"

		<< " , " << "Charge = " << q_ / elementary_charge << " * q_e"

		<< " , " << "Temperature = " << T_ * boltzmann_constant / elementary_charge << "* eV";

		return os.str();
	}

	Real GetMass() const
	{
		return m_;
	}

	Real GetCharge() const
	{
		return q_;
	}
	static Point_s DefaultValue()
	{
		Point_s p;
		p.f = 1.0;
		return std::move(p);
	}

	template<typename TJ, typename TB, typename TE, typename ... Others>
	inline void NextTimeStep(Point_s * p, Real dt, TJ *J, TE const &fE, TB const & fB, Others const &...others) const
	{
		RVec3 B0 = interpolator_type::Gather(fB, p->x);
		Real BB = Dot(B0, B0);

		RVec3 b = B0 / std::sqrt(BB);
		Vec3 v0, v1, r0, r1;
		Vec3 Vc;
		Vc = (Dot(p->v, B0) * B0) / BB;
		v1 = Cross(p->v, b);
		v0 = -Cross(v1, b);
		r0 = -Cross(v0, B0) / (cmr_ * BB);
		r1 = -Cross(v1, B0) / (cmr_ * BB);

		for (int ms = 0; ms < NMATE; ++ms)
		{
			Vec3 v, r;
			v = Vc + v0 * cosdq[ms] + v1 * sindq[ms];
			r = (p->x + r0 * cosdq[ms] + r1 * sindq[ms]);
			p->w[ms] += 0.5 * Dot(fE(r), v) * dt;
		}

		Vec3 t, V_;

		t = B0 * cmr_ * dt * 0.5;

		V_ = p->v + Cross(p->v, t);

		p->v += Cross(V_, t) / (Dot(t, t) + 1.0) * 2.0;

		Vc = (Dot(p->v, B0) * B0) / BB;

		p->x += Vc * dt * 0.5;

		v1 = Cross(p->v, b);
		v0 = -Cross(v1, b);
		r0 = -Cross(v0, B0) / (cmr_ * BB);
		r1 = -Cross(v1, B0) / (cmr_ * BB);

		for (int ms = 0; ms < NMATE; ++ms)
		{
			Vec3 v, r;
			v = Vc + v0 * cosdq[ms] + v1 * sindq[ms];
			r = (p->x + r0 * cosdq[ms] + r1 * sindq[ms]);
			p->w[ms] += 0.5 * Dot(interpolator_type::Gather(fE, r), v) * q_ / T_ * dt;

		}
		p->x += Vc * dt * 0.5;

		Vc = (Dot(p->v, B0) * B0) / BB;

		v1 = Cross(p->v, b);
		v0 = -Cross(v1, b);
		r0 = -Cross(v0, B0) / (cmr_ * BB);
		r1 = -Cross(v1, B0) / (cmr_ * BB);

		for (int ms = 0; ms < NMATE; ++ms)
		{
			Vec3 v, r;
			v = Vc + v0 * cosdq[ms] + v1 * sindq[ms];
			r = (p->x + r0 * cosdq[ms] + r1 * sindq[ms]);

			interpolator_type::Scatter(r, v * p->w[ms] * q_ * p->f, J);
		}

	}

	inline void Reflect(nTuple<3, Real> const & x, nTuple<3, Real> const & nv, Point_s * p) const
	{

	}
	template<typename TJ, typename ...Args>
	void Scatter(Point_s const & p, Field<mesh_type, EDGE, TJ> * J, Args const & ...) const
	{
		UNIMPLEMENT;
	}

	template<typename TJ, typename ...Args>
	void Scatter(Point_s const & p, Field<mesh_type, VERTEX, TJ> * n, Args const & ...) const
	{
		UNIMPLEMENT;
	}
	inline Real PullBack(Point_s const & p, nTuple<3, Real> *x, nTuple<3, Real> * v) const
	{
		return 1.0;
	}
	inline void PushForward(nTuple<3, Real> const&x, nTuple<3, Real> const& v, Point_s * p) const
	{
	}
	static inline Point_s make_point(coordinates_type const & x, Vec3 const &v, Real f)
	{
		Point_s res;
		res.x = x;
		res.v = v;
		res.f = f;
		res.w *= 0;
		return std::move(res);
	}

}
;

template<typename OS, typename TM, typename TS> OS&
operator<<(OS& os, typename PICEngineGGauge<TM, TS>::Point_s const & p)
{
	os << "{ x= {" << p.x << "} , v={" << p.v << "}, f=" << p.f << " , w=" << p.w << " }";

	return os;
}

} // namespace simpla

#endif /* PIC_ENGINE_GGAUGE_H_ */
