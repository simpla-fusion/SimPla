/**
 * \file particle_engine.h
 *
 * \date    2014年8月29日  上午10:36:23 
 * \author salmon
 */

#ifndef PARTICLE_ENGINE_H_
#define PARTICLE_ENGINE_H_
#include <stddef.h>
#include "../physics/physical_constants.h"
#include "../utilities/utilities.h"
#include "../dataset/datatype.h"
namespace simpla
{

/**
 * @ingroup particle
 * @addtogroup particle_engine  Particle Engine
 * ## Summary
 *
 * ## Requirements
 * The following table lists the requirements for a @particle_engine type E.
 *
 * Pseudo-Signature   |Semantics| optional
 * ------------- |----------|----------
 * \code E::Point_s \endcode |explicit |data structure and description of single particle/sample point
 * \code E( ) \endcode  |explicit | Constructor
 * \code ~E( ) \endcode |explicit | Destructor
 * \code void  next_timestep(Point_s * p, args ...) const; \endcode |explicit | push one particle to next timestep
 * \code void  update();\endcode | (optional) |update charge/mass and properties cache
 * \code static Point_s  push_forward(Vec3 const & x, Vec3 const &v, Real f);\endcode| (optional)| push forward Cartesian Coordinates x , velocity vector v  and sample weight f to paritlce's coordinates
 * \code static std::tuple<Vec3,Vec3,Real>  pull_back(Point_s const & p); \endcode| (optional)| pull back particle coordinates to Cartesian coordinates;
 * \code DataType  Point_s::datatype() \endcode |(implicit)| get the description of Point_s's data structure
 * \code Properties  properties \endcode |(implicit)| properties of engine
 *
 *
 * ## Example:
 \code
 struct DemoParticleEngine
 {

 SP_DEFINE_POINT_STRUCT(Point_s,
 Vec3 ,x,
 double[3], v,
 Real, f,
 Real, w
 )

 SP_DEFINE_PROPERTIES(
 Real, mass,
 Real, charge,
 Real, temperature,
 Real[3][3], pressure
 )


 static constexpr size_t memory_length = 0; //!  declare this engine is memoryless

 private:
 Real cmr_, q_kT_;
 public:

 DemoParticleEngine() :
 mass(1.0), charge(1.0), temperature(1.0)
 {
 pressure = 0;
 update();
 }

 ~DemoParticleEngine()
 {
 }

 static std::string get_type_as_string()
 {
 return "DemoParticleEngine";
 }


 void update()
 {
 DEFINE_PHYSICAL_CONST

 cmr_ = charge / mass;
 q_kT_ = charge / (temperature * boltzmann_constant);
 }

 template<typename Point_p, typename TE, typename TB>
 void next_timestep(Point_p p, Real dt, TE const &fE, TB const & fB) const
 {
 p->x+=p->v*dt;
 ....
 }

 };
 \endcode
 *
 *
 */

//*******************************************************************************************************
/**
 @ingroup particle
 @addtogroup particle_engine Particle Engine
 @{
 @brief @ref particle_engine describes the individual behavior of one sample.
 @details
 @}
 */
/*
 * Count the number of arguments passed to MACRO, very carefully
 * tiptoeing around an MSVC bug where it improperly expands __VA_ARGS__ as a
 * single token in argument lists.  See these URLs for details:
 *
 *   http://stackoverflow.com/questions/9183993/msvc-variadic-macro-expansion/9338429#9338429
 *   http://connect.microsoft.com/VisualStudio/feedback/details/380090/variadic-macro-replacement
 *   http://cplusplus.co.il/2010/07/17/variadic-macro-to-count-number-of-arguments/#comment-644
 */
#define COUNT_MACRO_ARGS_IMPL2(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18, count, ...) count
#define COUNT_MACRO_ARGS_IMPL(args) COUNT_MACRO_ARGS_IMPL2 args
#define COUNT_MACRO_ARGS(...) COUNT_MACRO_ARGS_IMPL((__VA_ARGS__,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0))

/* Pick the right helper macro to invoke. */
#define SP_PARTICLE_DEFINE_MEMBER_HELPER2(_T0_,_N0_) typename array_to_ntuple_convert<_T0_>::type _N0_;
#define SP_PARTICLE_DEFINE_MEMBER_HELPER4(_T0_,_N0_,_T1_,_N1_) SP_PARTICLE_DEFINE_MEMBER_HELPER2(_T0_,_N0_) \
	  SP_PARTICLE_DEFINE_MEMBER_HELPER2(_T1_,_N1_)
#define SP_PARTICLE_DEFINE_MEMBER_HELPER6(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_) SP_PARTICLE_DEFINE_MEMBER_HELPER2(_T0_,_N0_) \
	  SP_PARTICLE_DEFINE_MEMBER_HELPER4(_T1_,_N1_,_T2_,_N2_)
#define SP_PARTICLE_DEFINE_MEMBER_HELPER8(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_) SP_PARTICLE_DEFINE_MEMBER_HELPER2(_T0_,_N0_) \
	  SP_PARTICLE_DEFINE_MEMBER_HELPER6(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_)
#define SP_PARTICLE_DEFINE_MEMBER_HELPER10(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_) SP_PARTICLE_DEFINE_MEMBER_HELPER2(_T0_,_N0_) \
	  SP_PARTICLE_DEFINE_MEMBER_HELPER8(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_)
#define SP_PARTICLE_DEFINE_MEMBER_HELPER12(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_) SP_PARTICLE_DEFINE_MEMBER_HELPER2(_T0_,_N0_) \
	  SP_PARTICLE_DEFINE_MEMBER_HELPER10(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_)
#define SP_PARTICLE_DEFINE_MEMBER_HELPER14(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_) SP_PARTICLE_DEFINE_MEMBER_HELPER2(_T0_,_N0_) \
	  SP_PARTICLE_DEFINE_MEMBER_HELPER12(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_)
#define SP_PARTICLE_DEFINE_MEMBER_HELPER16(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_) SP_PARTICLE_DEFINE_MEMBER_HELPER2(_T0_,_N0_) \
	  SP_PARTICLE_DEFINE_MEMBER_HELPER14(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_)
#define SP_PARTICLE_DEFINE_MEMBER_HELPER18(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_) SP_PARTICLE_DEFINE_MEMBER_HELPER2(_T0_,_N0_) \
	  SP_PARTICLE_DEFINE_MEMBER_HELPER16(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_)

#define SP_PARTICLE_DEFINE_MEMBER_CHOOSE_HELPER1(count) SP_PARTICLE_DEFINE_MEMBER_HELPER##count
#define SP_PARTICLE_DEFINE_MEMBER_CHOOSE_HELPER(count) SP_PARTICLE_DEFINE_MEMBER_CHOOSE_HELPER1(count)
#define SP_PARTICLE_DEFINE_MEMBER(...) SP_PARTICLE_DEFINE_MEMBER_CHOOSE_HELPER(COUNT_MACRO_ARGS(__VA_ARGS__)) (__VA_ARGS__)

//#define SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) _N0_
//#define SP_PARTICLE_GET_NAME_HELPER4(_T0_,_N0_,_T1_,_N1_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER2(_T1_,_N1_)
//#define SP_PARTICLE_GET_NAME_HELPER6(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_)  SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER4(_T1_,_N1_,_T2_,_N2_)
//#define SP_PARTICLE_GET_NAME_HELPER8(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER6(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_)
//#define SP_PARTICLE_GET_NAME_HELPER10(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER8(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_)
//#define SP_PARTICLE_GET_NAME_HELPER12(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER10(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_)
//#define SP_PARTICLE_GET_NAME_HELPER14(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER12(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_)
//#define SP_PARTICLE_GET_NAME_HELPER16(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER14(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_)
//#define SP_PARTICLE_GET_NAME_HELPER18(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_) SP_PARTICLE_GET_NAME_HELPER2(_T0_,_N0_) , \
//	  SP_PARTICLE_GET_NAME_HELPER16(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_)
//
//#define SP_PARTICLE_GET_NAME_CHOOSE_HELPER1(count) SP_PARTICLE_GET_NAME_HELPER##count
//#define SP_PARTICLE_GET_NAME_CHOOSE_HELPER(count) SP_PARTICLE_GET_NAME_CHOOSE_HELPER1(count)
//#define SP_PARTICLE_GET_NAME(...) SP_PARTICLE_GET_NAME_CHOOSE_HELPER(COUNT_MACRO_ARGS(__VA_ARGS__)) (__VA_ARGS__)

#define SP_PARTICLE_DEFINE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) d_type.push_back(make_datatype<typename array_to_ntuple_convert<_T0_>::type>(), #_N0_, offsetof(_S_NAME_, _N0_));
#define SP_PARTICLE_DEFINE_DESC_HELPER4(_S_NAME_,_T0_,_N0_,_T1_,_N1_) SP_PARTICLE_DEFINE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  SP_PARTICLE_DEFINE_DESC_HELPER2(_S_NAME_,_T1_,_N1_)
#define SP_PARTICLE_DEFINE_DESC_HELPER6(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_)  SP_PARTICLE_DEFINE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  SP_PARTICLE_DEFINE_DESC_HELPER4(_S_NAME_,_T1_,_N1_,_T2_,_N2_)
#define SP_PARTICLE_DEFINE_DESC_HELPER8(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_) SP_PARTICLE_DEFINE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  SP_PARTICLE_DEFINE_DESC_HELPER6(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_)
#define SP_PARTICLE_DEFINE_DESC_HELPER10(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_) SP_PARTICLE_DEFINE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  SP_PARTICLE_DEFINE_DESC_HELPER8(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_)
#define SP_PARTICLE_DEFINE_DESC_HELPER12(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_) SP_PARTICLE_DEFINE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  SP_PARTICLE_DEFINE_DESC_HELPER10(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_)
#define SP_PARTICLE_DEFINE_DESC_HELPER14(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_) SP_PARTICLE_DEFINE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  SP_PARTICLE_DEFINE_DESC_HELPER12(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_)
#define SP_PARTICLE_DEFINE_DESC_HELPER16(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_) SP_PARTICLE_DEFINE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  SP_PARTICLE_DEFINE_DESC_HELPER14(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_)
#define SP_PARTICLE_DEFINE_DESC_HELPER18(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_) SP_PARTICLE_DEFINE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  SP_PARTICLE_DEFINE_DESC_HELPER16(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_)

#define SP_PARTICLE_DEFINE_DESC_CHOOSE_HELPER1(count) SP_PARTICLE_DEFINE_DESC_HELPER##count
#define SP_PARTICLE_DEFINE_DESC_CHOOSE_HELPER(count) SP_PARTICLE_DEFINE_DESC_CHOOSE_HELPER1(count)
#define SP_PARTICLE_DEFINE_DESC(_S_NAME_,...) SP_PARTICLE_DEFINE_DESC_CHOOSE_HELPER(COUNT_MACRO_ARGS(__VA_ARGS__)) (_S_NAME_,__VA_ARGS__)

/**
 *
 *  \brief Define Point_s struct:
 *     MAX number of members is 9
 *  *Usage:
 *  \code SP_DEFINE_POINT_STRUCT(Point_s, double, x, double, y, double, z, int, a, int, b, int, c)  \endcode
 *  *Macro Expansion
 *  \code
 struct Point_s
 {
 double x;
 double y;
 double z;
 int a;
 int b;
 int c;
 static DataType create_datadesc()
 {
 auto d_type = DataType::create<Point_s>();
 d_type.push_back<double>("x", offsetof(Point_s, x));
 d_type.push_back<double>("y", offsetof(Point_s, y));
 d_type.push_back<double>("z", offsetof(Point_s, z));
 d_type.push_back<int>("a", offsetof(Point_s, a));
 d_type.push_back<int>("b", offsetof(Point_s, b));
 d_type.push_back<int>("c", offsetof(Point_s, c));
 ;
 return std::move(d_type);
 }
 };
 \endcode
 */
#define SP_DEFINE_POINT_STRUCT(_S_NAME_,...)                                 \
struct _S_NAME_                                                  \
{                                                                \
	SP_PARTICLE_DEFINE_MEMBER(__VA_ARGS__)                                   \
	static DataType datatype()                             \
	{                                                             \
		auto d_type = DataType::create_opaque_type<Point_s>(#_S_NAME_);  \
		SP_PARTICLE_DEFINE_DESC(_S_NAME_,__VA_ARGS__);        \
		return std::move(d_type);                                 \
	}                                                             \
};

//#define SP_PARTICLE_ADD_PROP_HELPER2(_S_NAME_,_T0_,_N0_) _N0_=p_##_N0_;_S_NAME_.set<typename array_to_ntuple_convert<_T0_>::type>(#_N0_,_N0_);
//#define SP_PARTICLE_ADD_PROP_HELPER4(_S_NAME_,_T0_,_N0_,_T1_,_N1_) SP_PARTICLE_ADD_PROP_HELPER2(_S_NAME_,_T0_,_N0_) \
//	  SP_PARTICLE_ADD_PROP_HELPER2(_S_NAME_,_T1_,_N1_)
//#define SP_PARTICLE_ADD_PROP_HELPER6(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_)  SP_PARTICLE_ADD_PROP_HELPER2(_S_NAME_,_T0_,_N0_) \
//	  SP_PARTICLE_ADD_PROP_HELPER4(_S_NAME_,_T1_,_N1_,_T2_,_N2_)
//#define SP_PARTICLE_ADD_PROP_HELPER8(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_) SP_PARTICLE_ADD_PROP_HELPER2(_S_NAME_,_T0_,_N0_) \
//	  SP_PARTICLE_ADD_PROP_HELPER6(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_)
//#define SP_PARTICLE_ADD_PROP_HELPER10(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_) SP_PARTICLE_ADD_PROP_HELPER2(_S_NAME_,_T0_,_N0_) \
//	  SP_PARTICLE_ADD_PROP_HELPER8(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_)
//#define SP_PARTICLE_ADD_PROP_HELPER12(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_) SP_PARTICLE_ADD_PROP_HELPER2(_S_NAME_,_T0_,_N0_) \
//	  SP_PARTICLE_ADD_PROP_HELPER10(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_)
//#define SP_PARTICLE_ADD_PROP_HELPER14(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_) SP_PARTICLE_ADD_PROP_HELPER2(_S_NAME_,_T0_,_N0_) \
//	  SP_PARTICLE_ADD_PROP_HELPER12(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_)
//#define SP_PARTICLE_ADD_PROP_HELPER16(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_) SP_PARTICLE_ADD_PROP_HELPER2(_S_NAME_,_T0_,_N0_) \
//	  SP_PARTICLE_ADD_PROP_HELPER14(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_)
//#define SP_PARTICLE_ADD_PROP_HELPER18(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_) SP_PARTICLE_ADD_PROP_HELPER2(_S_NAME_,_T0_,_N0_) \
//	  SP_PARTICLE_ADD_PROP_HELPER16(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_)
//
//#define SP_PARTICLE_ADD_PROP_CHOOSE_HELPER1(count) SP_PARTICLE_ADD_PROP_HELPER##count
//#define SP_PARTICLE_ADD_PROP_CHOOSE_HELPER(count) SP_PARTICLE_ADD_PROP_CHOOSE_HELPER1(count)
//#define SP_PARTICLE_ADD_PROP(_S_NAME_,...) SP_PARTICLE_ADD_PROP_CHOOSE_HELPER(COUNT_MACRO_ARGS(__VA_ARGS__)) (_S_NAME_,__VA_ARGS__)

#define SP_PARTICLE_LOAD_DICT_HELPER2(_S1_,_S2_,_T0_,_N0_) _N0_=_S2_[#_N0_].template as<typename array_to_ntuple_convert<_T0_>::type>();
#define SP_PARTICLE_LOAD_DICT_HELPER4(_S1_,_S2_,_T0_,_N0_,_T1_,_N1_) SP_PARTICLE_LOAD_DICT_HELPER2(_S1_,_S2_,_T0_,_N0_) \
	  SP_PARTICLE_LOAD_DICT_HELPER2(_S1_,_S2_,_T1_,_N1_)
#define SP_PARTICLE_LOAD_DICT_HELPER6(_S1_,_S2_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_)  SP_PARTICLE_LOAD_DICT_HELPER2(_S1_,_S2_,_T0_,_N0_) \
	  SP_PARTICLE_LOAD_DICT_HELPER4(_S1_,_S2_,_T1_,_N1_,_T2_,_N2_)
#define SP_PARTICLE_LOAD_DICT_HELPER8(_S1_,_S2_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_) SP_PARTICLE_LOAD_DICT_HELPER2(_S1_,_S2_,_T0_,_N0_) \
	  SP_PARTICLE_LOAD_DICT_HELPER6(_S1_,_S2_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_)
#define SP_PARTICLE_LOAD_DICT_HELPER10(_S1_,_S2_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_) SP_PARTICLE_LOAD_DICT_HELPER2(_S1_,_S2_,_T0_,_N0_) \
	  SP_PARTICLE_LOAD_DICT_HELPER8(_S1_,_S2_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_)
#define SP_PARTICLE_LOAD_DICT_HELPER12(_S1_,_S2_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_) SP_PARTICLE_LOAD_DICT_HELPER2(_S1_,_S2_,_T0_,_N0_) \
	  SP_PARTICLE_LOAD_DICT_HELPER10(_S1_,_S2_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_)
#define SP_PARTICLE_LOAD_DICT_HELPER14(_S1_,_S2_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_) SP_PARTICLE_LOAD_DICT_HELPER2(_S1_,_S2_,_T0_,_N0_) \
	  SP_PARTICLE_LOAD_DICT_HELPER12(_S1_,_S2_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_)
#define SP_PARTICLE_LOAD_DICT_HELPER16(_S1_,_S2_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_) SP_PARTICLE_LOAD_DICT_HELPER2(_S1_,_S2_,_T0_,_N0_) \
	  SP_PARTICLE_LOAD_DICT_HELPER14(_S1_,_S2_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_)
#define SP_PARTICLE_LOAD_DICT_HELPER18(_S1_,_S2_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_) SP_PARTICLE_LOAD_DICT_HELPER2(_S1_,_S2_,_T0_,_N0_) \
	  SP_PARTICLE_LOAD_DICT_HELPER16(_S1_,_S2_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_)

#define SP_PARTICLE_LOAD_DICT_CHOOSE_HELPER1(count) SP_PARTICLE_LOAD_DICT_HELPER##count
#define SP_PARTICLE_LOAD_DICT_CHOOSE_HELPER(count) SP_PARTICLE_LOAD_DICT_CHOOSE_HELPER1(count)
#define SP_PARTICLE_LOAD_DICT(_S1_,_S2_,...) SP_PARTICLE_LOAD_DICT_CHOOSE_HELPER(COUNT_MACRO_ARGS(__VA_ARGS__)) (_S1_,_S2_,__VA_ARGS__)

#define SP_PARTICLE_UPDATE_PROP_HELPER2(_S1_,_T0_,_N0_) _S1_.set(#_N0_,_N0_);
#define SP_PARTICLE_UPDATE_PROP_HELPER4(_S1_,_T0_,_N0_,_T1_,_N1_) SP_PARTICLE_UPDATE_PROP_HELPER2(_S1_,_T0_,_N0_) \
	  SP_PARTICLE_UPDATE_PROP_HELPER2(_S1_,_T1_,_N1_)
#define SP_PARTICLE_UPDATE_PROP_HELPER6(_S1_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_)  SP_PARTICLE_UPDATE_PROP_HELPER2(_S1_,_T0_,_N0_) \
	  SP_PARTICLE_UPDATE_PROP_HELPER4(_S1_,_T1_,_N1_,_T2_,_N2_)
#define SP_PARTICLE_UPDATE_PROP_HELPER8(_S1_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_) SP_PARTICLE_UPDATE_PROP_HELPER2(_S1_,_T0_,_N0_) \
	  SP_PARTICLE_UPDATE_PROP_HELPER6(_S1_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_)
#define SP_PARTICLE_UPDATE_PROP_HELPER10(_S1_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_) SP_PARTICLE_UPDATE_PROP_HELPER2(_S1_,_T0_,_N0_) \
	  SP_PARTICLE_UPDATE_PROP_HELPER8(_S1_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_)
#define SP_PARTICLE_UPDATE_PROP_HELPER12(_S1_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_) SP_PARTICLE_UPDATE_PROP_HELPER2(_S1_,_T0_,_N0_) \
	  SP_PARTICLE_UPDATE_PROP_HELPER10(_S1_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_)
#define SP_PARTICLE_UPDATE_PROP_HELPER14(_S1_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_) SP_PARTICLE_UPDATE_PROP_HELPER2(_S1_,_T0_,_N0_) \
	  SP_PARTICLE_UPDATE_PROP_HELPER12(_S1_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_)
#define SP_PARTICLE_UPDATE_PROP_HELPER16(_S1_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_) SP_PARTICLE_UPDATE_PROP_HELPER2(_S1_,_T0_,_N0_) \
	  SP_PARTICLE_UPDATE_PROP_HELPER14(_S1_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_)
#define SP_PARTICLE_UPDATE_PROP_HELPER18(_S1_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_) SP_PARTICLE_UPDATE_PROP_HELPER2(_S1_,_T0_,_N0_) \
	  SP_PARTICLE_UPDATE_PROP_HELPER16(_S1_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_)

#define SP_PARTICLE_UPDATE_PROP_CHOOSE_HELPER1(count) SP_PARTICLE_UPDATE_PROP_HELPER##count
#define SP_PARTICLE_UPDATE_PROP_CHOOSE_HELPER(count) SP_PARTICLE_UPDATE_PROP_CHOOSE_HELPER1(count)
#define SP_PARTICLE_UPDATE_PROP(_S1_,...) SP_PARTICLE_UPDATE_PROP_CHOOSE_HELPER(COUNT_MACRO_ARGS(__VA_ARGS__)) (_S1_,__VA_ARGS__)

//#define SP_PARTICLE_CONVERT_PARAMETER_HELPER2(_S_NAME_,_T0_,_N0_)  typename array_to_ntuple_convert<_T0_>::type p_##_N0_
//#define SP_PARTICLE_CONVERT_PARAMETER_HELPER4(_S_NAME_,_T0_,_N0_,_T1_,_N1_) SP_PARTICLE_CONVERT_PARAMETER_HELPER2(_S_NAME_,_T0_,_N0_), \
//	  SP_PARTICLE_CONVERT_PARAMETER_HELPER2(_S_NAME_,_T1_,_N1_)
//#define SP_PARTICLE_CONVERT_PARAMETER_HELPER6(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_)  SP_PARTICLE_CONVERT_PARAMETER_HELPER2(_S_NAME_,_T0_,_N0_), \
//	  SP_PARTICLE_CONVERT_PARAMETER_HELPER4(_S_NAME_,_T1_,_N1_,_T2_,_N2_)
//#define SP_PARTICLE_CONVERT_PARAMETER_HELPER8(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_) SP_PARTICLE_CONVERT_PARAMETER_HELPER2(_S_NAME_,_T0_,_N0_), \
//	  SP_PARTICLE_CONVERT_PARAMETER_HELPER6(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_)
//#define SP_PARTICLE_CONVERT_PARAMETER_HELPER10(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_) SP_PARTICLE_CONVERT_PARAMETER_HELPER2(_S_NAME_,_T0_,_N0_), \
//	  SP_PARTICLE_CONVERT_PARAMETER_HELPER8(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_)
//#define SP_PARTICLE_CONVERT_PARAMETER_HELPER12(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_) SP_PARTICLE_CONVERT_PARAMETER_HELPER2(_S_NAME_,_T0_,_N0_), \
//	  SP_PARTICLE_CONVERT_PARAMETER_HELPER10(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_)
//#define SP_PARTICLE_CONVERT_PARAMETER_HELPER14(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_) SP_PARTICLE_CONVERT_PARAMETER_HELPER2(_S_NAME_,_T0_,_N0_), \
//	  SP_PARTICLE_CONVERT_PARAMETER_HELPER12(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_)
//#define SP_PARTICLE_CONVERT_PARAMETER_HELPER16(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_) SP_PARTICLE_CONVERT_PARAMETER_HELPER2(_S_NAME_,_T0_,_N0_), \
//	  SP_PARTICLE_CONVERT_PARAMETER_HELPER14(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_)
//#define SP_PARTICLE_CONVERT_PARAMETER_HELPER18(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_) SP_PARTICLE_CONVERT_PARAMETER_HELPER2(_S_NAME_,_T0_,_N0_), \
//	  SP_PARTICLE_CONVERT_PARAMETER_HELPER16(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_)
//
//#define SP_PARTICLE_CONVERT_PARAMETER_CHOOSE_HELPER1(count) SP_PARTICLE_CONVERT_PARAMETER_HELPER##count
//#define SP_PARTICLE_CONVERT_PARAMETER_CHOOSE_HELPER(count) SP_PARTICLE_CONVERT_PARAMETER_CHOOSE_HELPER1(count)
//#define SP_PARTICLE_CONVERT_PARAMETER(_S_NAME_,...) SP_PARTICLE_CONVERT_PARAMETER_CHOOSE_HELPER(COUNT_MACRO_ARGS(__VA_ARGS__)) (_S_NAME_,__VA_ARGS__)

/**
 *  \brief Define Property variables:
 *     MAX number of variable is 9
 * * Usage:
 *  \code SP_DEFINE_PROPERTIES(Real, mass,	Real, charge,Real, temperature)  \endcode
 *
 * * Macro Expansion
 \code
 Properties properties;
 Real  mass;
 Real  charge;
 Real  temperature;
 void load(Real  p_mass,
 Real  p_charge,
 Real  p_temperature)
 {
 mass=p_mass;properties.set<Real>("mass",mass);
 charge=p_charge;properties.set<Real>("charge",charge);
 emperature=p_temperature;properties.set<Real>("temperature",temperature);
 update();
 }
 template<typename TDict,typename ...Others>
 void load(TDict const & dict,Others && ...)
 {
 mass=dict["mass"].template as<Real>();properties.template set<Real>("mass",mass);
 charge=dict["charge"].template as<Real>();properties.template set<Real>("charge",charge);
 temperature=dict["temperature"].template as<Real>();properties.template set<Real>("temperature",temperature);
 update();
 }
 \endcode
 */
#define SP_DEFINE_PROPERTIES(...)                            \
Properties properties;                                       \
SP_PARTICLE_DEFINE_MEMBER(__VA_ARGS__)                       \
template<typename TDict,typename ...Others>                  \
void load(TDict const & dict,Others && ...)                  \
{                                                            \
	SP_PARTICLE_LOAD_DICT(properties,dict,__VA_ARGS__)       \
}                                                            \
void update_properties()                                     \
{                                                            \
	SP_PARTICLE_UPDATE_PROP(properties,__VA_ARGS__)          \
	properties.update();                                     \
}                                                            \
template<typename OS> OS& print(OS &os)const                 \
{  properties.print(os);		return os;	}

struct SimpleParticleEngine
{
	typedef double scalar_type;
	typedef nTuple<scalar_type, 3> coordinates_type;
	typedef nTuple<scalar_type, 3> vector_type;

//	SP_DEFINE_POINT_STRUCT(Point_s,
//			coordinates_type,x,
//			vector_type, v,
//			scalar_type, f )

	struct Point_s
	{
		coordinates_type x;
		vector_type v;
		scalar_type f;

		static DataType datatype()
		{
			auto d_type = DataType::create_opaque_type<Point_s>("Point_s");
			d_type.push_back(
					make_datatype<
							typename array_to_ntuple_convert<coordinates_type>::type>(),
					"x",
					(reinterpret_cast<size_t>(&reinterpret_cast<const volatile char &>(static_cast<Point_s*>(0)->x))));
			d_type.push_back(
					make_datatype<
							typename array_to_ntuple_convert<vector_type>::type>(),
					"v",
					(reinterpret_cast<size_t>(&reinterpret_cast<const volatile char &>(static_cast<Point_s*>(0)->v))));
			d_type.push_back(
					make_datatype<
							typename array_to_ntuple_convert<scalar_type>::type>(),
					"f",
					(reinterpret_cast<size_t>(&reinterpret_cast<const volatile char &>(static_cast<Point_s*>(0)->f))));
			;
			return std::move(d_type);
		}
	};

	SP_DEFINE_PROPERTIES(
			scalar_type, mass,
			scalar_type, charge,
			scalar_type, temperature
	)

private:
	Real cmr_, q_kT_;
public:

	SimpleParticleEngine()
			: mass(1.0), charge(1.0), temperature(1.0)
	{
		update();
	}

	void update()
	{
		DEFINE_PHYSICAL_CONST
		cmr_ = charge / mass;
		q_kT_ = charge / (temperature * boltzmann_constant);
	}

	~SimpleParticleEngine()
	{
	}

	static std::string get_type_as_string()
	{
		return "DeltaF";
	}

	template<typename TE, typename TB, typename TJ>
	void next_timestep(Point_s * p, Real dt, TE const &fE, TB const & fB,
			TJ * J) const
	{
		p->x += p->v * dt * 0.5;

		auto B = fB(p->x);
		auto E = fE(p->x);

		Vec3 v_;

		auto t = B * (cmr_ * dt * 0.5);

		p->v += p->v + E * (cmr_ * dt * 0.5);

		v_ = p->v + Cross(p->v, t);

		v_ = Cross(v_, t) / (Dot(t, t) + 1.0) * 2;

		p->v += v_;

		p->v += E * (cmr_ * dt * 0.5);

		p->x += p->v * dt * 0.5;

		J->scatter_cartesian(std::forward_as_tuple(p->x, p->v, p->f * charge));

	}

	Point_s push_forward(coordinates_type const & x, vector_type const &v,
			Real f = 1.0) const
	{
		return Point_s( { x, v, f });
	}

};
}
// namespace simpla

#endif /* PARTICLE_ENGINE_H_ */
