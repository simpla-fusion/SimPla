/**
 * \file particle_engine.h
 *
 * \date    2014年8月29日  上午10:36:23 
 * \author salmon
 */

#ifndef PARTICLE_ENGINE_H_
#define PARTICLE_ENGINE_H_

namespace simpla
{

/**
 *\ingroup  ParticleEngine
 *\page ParticleEngine Particle Engine Conecpt
 * - Summary
 *z
 * - Requirements
 * The following table lists the requirements for a Particle Engine type E.
 *
 * Pseudo-Signature  |Semantics
 * ------------- | -------------
 * \code E::Point_s \endcode | data structure and description of single particle/sample point
 * \code E::Point_s::x \endcode | coordiantes of particle
 * \code DataType E::Point_s::create_datadesc() \endcode | get the description of Point_s's data strcut
 * \code E::E(...) \endcode  | Constructor
 * \code E::~E(...) \endcode | Destructor
 * \code void E::load(TDict const & dict);\endcode | Load configure information from dict
 * \code Real get_mass() const;\endcode | get mass
 * \code Real get_charge() const;\endcode | get charge
 * \code void E::next_timestep(Point_s * p, Real dt, TE const & E, TB const &  B) const; \endcode | Using field E,B push particle p, a  time step dt
 * \code void E::ScatterJ(Point_s const & p, TJ * J) const; \endcode | Scatter current density (v*f) to field J
 * \code void E::ScatterRho(Point_s const & p, TJ * rho) const; \endcode | Scatter density ( f) to field rho
 * \code static Point_s E::push_forward(Vec3 const & x, Vec3 const &v, Real f);\endcode| push forward Cartesian Coordiantes x , velocity vector v  and sample weight f to paritlce's coordinates
 * \code static std::tuple<Vec3,Vec3,Real>  E::pull_back(Point_s const & p); \endcode| pull back particle coordiantes to Cartesian coordinates;
 *
 *
 * example:
 * \code
 * 	struct E
 *	{
 *		typedef PICEngineDefault this_type;
 *		struct Point_s
 *		{
 *			Vec3 x;
 *			Vec3 v;
 *			double f ;
 *
 *			static DataType create_datadesc()
 *			{
 *			 auto d_type = DataType::create<Point_s>();
 * 		     d_type.push_back<Vec3>("x", offsetof(Point_s, x));
 * 		     d_type.push_back<Vec3>("v", offsetof(Point_s, v));
 * 		     d_type.push_back<double>("f", offsetof(Point_s, f));
 * 		     return std::move(d_type);
 *			};
 *		};
 *
 *		PICEngineDefault(...);
 *
 *		~PICEngineDefault();
 *
 *		template<typename TDict> void load(TDict const & dict);
 *
 *		static std::string get_type_as_string();
 *
 *		Real get_mass() const;
 *		Real get_charge() const;
 *
 *		template<typename TE, typename TB>
 *		inline void next_timestep(Point_s * p, Real dt, TE const &fE, TB const & fB) const;
 *
 *		template<typename TJ> void ScatterJ(Point_s const & p, TJ * J) const;
 *
 *		template<typename TJ> void ScatterRho(Point_s const & p, TJ * rho) const;
 *
 *		static inline Point_s push_forward(Vec3 const & x, Vec3 const &v, Real f);
 *
 *		static inline auto pull_back(Point_s const & p);
 *
 *	}
 *	;
 * \endcode
 *
 *
 **/

//*******************************************************************************************************
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
#define DECLARE_MEMBER_HELPER2(_T0_,_N0_) _T0_ _N0_;
#define DECLARE_MEMBER_HELPER4(_T0_,_N0_,_T1_,_N1_) DECLARE_MEMBER_HELPER2(_T0_,_N0_) \
	  DECLARE_MEMBER_HELPER2(_T1_,_N1_)
#define DECLARE_MEMBER_HELPER6(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_) DECLARE_MEMBER_HELPER2(_T0_,_N0_) \
	  DECLARE_MEMBER_HELPER4(_T1_,_N1_,_T2_,_N2_)
#define DECLARE_MEMBER_HELPER8(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_) DECLARE_MEMBER_HELPER2(_T0_,_N0_) \
	  DECLARE_MEMBER_HELPER6(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_)
#define DECLARE_MEMBER_HELPER10(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_) DECLARE_MEMBER_HELPER2(_T0_,_N0_) \
	  DECLARE_MEMBER_HELPER8(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_)
#define DECLARE_MEMBER_HELPER12(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_) DECLARE_MEMBER_HELPER2(_T0_,_N0_) \
	  DECLARE_MEMBER_HELPER10(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_)
#define DECLARE_MEMBER_HELPER14(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_) DECLARE_MEMBER_HELPER2(_T0_,_N0_) \
	  DECLARE_MEMBER_HELPER12(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_)
#define DECLARE_MEMBER_HELPER16(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_) DECLARE_MEMBER_HELPER2(_T0_,_N0_) \
	  DECLARE_MEMBER_HELPER14(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_)
#define DECLARE_MEMBER_HELPER16(_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_) DECLARE_MEMBER_HELPER2(_T0_,_N0_) \
	  DECLARE_MEMBER_HELPER14(_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_)

#define DECLARE_MEMBER_CHOOSE_HELPER1(count) DECLARE_MEMBER_HELPER##count
#define DECLARE_MEMBER_CHOOSE_HELPER(count) DECLARE_MEMBER_CHOOSE_HELPER1(count)
#define DECLARE_MEMBER(...) DECLARE_MEMBER_CHOOSE_HELPER(COUNT_MACRO_ARGS(__VA_ARGS__)) (__VA_ARGS__)

#define DECLARE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) d_type.push_back<_T0_>(#_N0_, offsetof(_S_NAME_, _N0_));
#define DECLARE_DESC_HELPER4(_S_NAME_,_T0_,_N0_,_T1_,_N1_) DECLARE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  DECLARE_DESC_HELPER2(_S_NAME_,_T1_,_N1_)
#define DECLARE_DESC_HELPER6(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_)  DECLARE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  DECLARE_DESC_HELPER4(_S_NAME_,_T1_,_N1_,_T2_,_N2_)
#define DECLARE_DESC_HELPER8(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_) DECLARE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  DECLARE_DESC_HELPER6(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_)
#define DECLARE_DESC_HELPER10(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_) DECLARE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  DECLARE_DESC_HELPER8(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_)
#define DECLARE_DESC_HELPER12(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_) DECLARE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  DECLARE_DESC_HELPER10(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_)
#define DECLARE_DESC_HELPER14(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_) DECLARE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  DECLARE_DESC_HELPER12(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_)
#define DECLARE_DESC_HELPER16(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_) DECLARE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  DECLARE_DESC_HELPER14(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_)
#define DECLARE_DESC_HELPER16(_S_NAME_,_T0_,_N0_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_) DECLARE_DESC_HELPER2(_S_NAME_,_T0_,_N0_) \
	  DECLARE_DESC_HELPER14(_S_NAME_,_T1_,_N1_,_T2_,_N2_,_T3_,_N3_,_T4_,_N4_,_T5_,_N5_,_T6_,_N6_,_T7_,_N7_,_T8_,_N8_)

#define DECLARE_DESC_CHOOSE_HELPER1(count) DECLARE_DESC_HELPER##count
#define DECLARE_DESC_CHOOSE_HELPER(count) DECLARE_DESC_CHOOSE_HELPER1(count)
#define DECLARE_DESC(_S_NAME_,...) DECLARE_DESC_CHOOSE_HELPER(COUNT_MACRO_ARGS(__VA_ARGS__)) (_S_NAME_,__VA_ARGS__)

/** \ingroup Particle
 *
 *  \brief Define Point_s :
 *     MAX number of members is 9
 *  * example
 *  \code DEFINE_POINT_STRUCT(Point_s, double, x, double, y, double, z, int, a, int, b, int, c)  \endcode
 *  Macro Expansion
 *  \code
 *  struct Point_s
 *  {
 *  	double x;
 *  	double y;
 *  	double z;
 *  	int a;
 *  	int b;
 *  	int c;
 *  	static DataType create_datadesc()
 *  	{
 *  		auto d_type = DataType::create<Point_s>();
 *  		d_type.push_back<double>("x", offsetof(Point_s, x));
 *  		d_type.push_back<double>("y", offsetof(Point_s, y));
 *  		d_type.push_back<double>("z", offsetof(Point_s, z));
 *  		d_type.push_back<int>("a", offsetof(Point_s, a));
 *  		d_type.push_back<int>("b", offsetof(Point_s, b));
 *  		d_type.push_back<int>("c", offsetof(Point_s, c));
 *  		;
 *  		return std::move(d_type);
 *  	}
 *  };
 *  \endcode
 */
#define DEFINE_POINT_STRUCT(_S_NAME_,...)                                 \
struct _S_NAME_                                                  \
{                                                                \
	DECLARE_MEMBER(__VA_ARGS__)                                   \
	static DataType create_datadesc()                             \
	{                                                             \
		auto d_type = DataType::create<Point_s>();                \
		DECLARE_DESC(_S_NAME_,__VA_ARGS__);                       \
		return std::move(d_type);                                 \
	}                                                             \
};

}
// namespace simpla

#endif /* PARTICLE_ENGINE_H_ */
