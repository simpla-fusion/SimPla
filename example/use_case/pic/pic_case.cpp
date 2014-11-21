/*
 * pic_case.cpp
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#include <string>

#include "../../../core/application/use_case.h"
#include "../../../core/simpla_defs.h"
#include "../../../core/utilities/log.h"
#include "../../../core/utilities/parse_command_line.h"

//#include "../../../core/common.h"
//#include "../../../core/particle/particle.h"
//#include "../../../core/manifold/fetl.h"
//#include "../../../core/physics/physical_constants.h"
//#include "../../../core/io/data_stream.h"
//#include "../../../core/manifold/domain_dummy.h"
//#include "../../../core/utilities/log.h"
//#include "../../../core/utilities/ntuple.h"
//#include "../../../core/utilities/parse_command_line.h"
//#include "../../../core/parallel/message_comm.h"

using namespace simpla;
//
//struct PICDemo
//{
//	typedef PICDemo this_type;
//	typedef Vec3 coordinates_type;
//	typedef Vec3 vector_type;
//	typedef Real scalar_type;
//
//	SP_DEFINE_POINT_STRUCT(Point_s,
//			coordinates_type ,x,
//			Vec3, v,
//			Real, f,
//			scalar_type, w)
//
//	SP_DEFINE_PROPERTIES(
//			Real, mass,
//			Real, charge,
//			Real, temperature
//	)
//
//private:
//	Real cmr_, q_kT_;
//public:
//
//	PICDemo() :
//			mass(1.0), charge(1.0), temperature(1.0)
//	{
//		update();
//	}
//
//	void update()
//	{
//		DEFINE_PHYSICAL_CONST
//		cmr_ = charge / mass;
//		q_kT_ = charge / (temperature * boltzmann_constant);
//	}
//
//	~PICDemo()
//	{
//	}
//
//	static std::string get_type_as_string()
//	{
//		return "PICDemo";
//	}
//
//	template<typename TE, typename TB>
//	void next_timestep(Point_s const* p0, Point_s * p1, Real dt, TE const &fE,
//			TB const & fB) const
//	{
//		p1->x += p0->v * dt * 0.5;
//
//		auto B = fB(p0->x);
//		auto E = fE(p0->x);
//
//		Vec3 v_;
//
//		auto t = B * (cmr_ * dt * 0.5);
//
//		p1->v += E * (cmr_ * dt * 0.5);
//
//		v_ = p0->v + cross(p1->v, t);
//
//		v_ = cross(v_, t) / (dot(t, t) + 1.0);
//
//		p1->v += v_;
//		auto a = (-dot(E, p1->v) * q_kT_ * dt);
//		p1->w = (-a + (1 + 0.5 * a) * p1->w) / (1 - 0.5 * a);
//
//		p1->v += v_;
//		p1->v += E * (cmr_ * dt * 0.5);
//
//		p1->x += p1->v * dt * 0.5;
//
//	}
//
//	static inline Point_s push_forward(coordinates_type const & x,
//			Vec3 const &v, scalar_type f)
//	{
//		return std::move(Point_s(
//		{ x, v, f }));
//	}
//
//	static inline auto pull_back(Point_s const & p)
//	DECL_RET_TYPE((std::make_tuple(p.x,p.v,p.f)))
//};

//class SPUseCase_pic: public UseCase
//{
//public:
//
//	typedef SPUseCase_pic this_type;
//	SPUseCase_pic()
//	{
//	}
//
//	virtual ~SPUseCase_pic()
//	{
//	}
//
//	SPUseCase_pic(this_type const &) = delete;
//
//	static const std::string case_info;
//private:
//
//	virtual void case_body();
//};
//
//const std::string SPUseCase_pic::case_info = use_case_register<SPUseCase_pic>(
//		("pic"));
//
//void SPUseCase_pic::case_body()

USE_CASE(pic)
{
//	typedef Manifold<CartesianCoordinates<StructuredMesh> > TManifold;
//
//	typedef TManifold manifold_type;

	bool is_configure_test_;

	parse_cmd_line(

	[&](std::string const & opt,std::string const & value)->int
	{

		if (opt=="t"|| opt=="test")
		{
			is_configure_test_=true;

		}
		else if(opt=="h" || opt=="help")
		{

			SHOW_OPTIONS("-n <NUM>","number of steps");
			SHOW_OPTIONS("-s <NUM>","recorder per <NUM> steps");
//					SHOW_OPTIONS("-g,--generator","generator a demo input script file");
			SHOW_OPTIONS("-t,--test ","only read and parse input file");

			return TERMINATE;
		}
		return CONTINUE;
	}

	);

}

