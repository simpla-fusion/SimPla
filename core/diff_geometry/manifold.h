/**
 * @file  manifold.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef MANIFOLD_H_
#define MANIFOLD_H_
#include <memory>
#include <utility>
#include <vector>
#include <ostream>

#include "../data_representation/data_set.h"
#include "../utilities/utilities.h"
namespace simpla
{

/**
 * @ingroup diff_geo
 * \addtogroup  manifold Manifold
 *  @{
 *    \brief   Discrete spatial-temporal space
 *
 *  ## Summary
 *   \note In mathematics, a _manifold_ is a topological space that resembles
 *   Euclidean space near each point. A _differentiable manifold_ is a type of
 *    manifold that is locally similar enough to a linear space to allow one to do calculus.
 *
 * ## Requirements
 *
 ~~~~~~~~~~~~~{.cpp}
 template <typename Geometry,template<typename> class Policy1,template<typename> class Policy2>
 class Manifold:
 public Geometry,
 public Policy1<Geometry>,
 public Policy2<Geometry>
 {
 .....
 };
 ~~~~~~~~~~~~~
 * The following table lists requirements for a Manifold type `M`,
 *
 *  Pseudo-Signature  		| Semantics
 *  -------------------|-------------
 *  `M( const M& )` 		| Copy constructor.
 *  `~M()` 				| Destructor.
 *  `geometry_type`		| Geometry type of manifold, which describes coordinates and metric
 *  `topology_type`		| Topology structure of manifold,   topology of grid points
 *  `coordiantes_type` 	| data type of coordinates, i.e. nTuple<3,Real>
 *  `index_type`			| data type of the index of grid points, i.e. unsigned long
 *  `Domain  domain()`	| Root domain of manifold
 *
 *
 * Manifold policy concept {#concept_manifold_policy}
 * ================================================
 *   Poilcies define the behavior of manifold , such as  interpolator or calculus;
 ~~~~~~~~~~~~~{.cpp}
 template <typename Geometry > class P;
 ~~~~~~~~~~~~~
 *
 *  The following table lists requirements for a Manifold policy type `P`,
 *
 *  Pseudo-Signature  		| Semantics
 *  -----------------------|-------------
 *  `P( Geometry  & )` 	| Constructor.
 *  `P( P const  & )`	| Copy constructor.
 *  `~P( )` 				| Copy Destructor.
 *
 * ## Interpolator policy
 *   Interpolator, map between discrete space and continue space, i.e. Gather & Scatter
 *
 *    Pseudo-Signature  		| Semantics
 *  ---------------------------|-------------
 *  `gather(field_type const &f, coordinates_type x  )` 	| gather data from `f` at coordinates `x`.
 *  `scatter(field_type &f, coordinates_type x ,value_type v)` 	| scatter `v` to field  `f` at coordinates `x`.
 *
 * ## Calculus  policy
 *  Define calculus operation of  fields on the manifold, such  as algebra or differential calculus.
 *  Differential calculus scheme , i.e. FDM,FVM,FEM,DG ....
 *
 *
 *  Pseudo-Signature  		| Semantics
 *  -----------------------|-------------
 *  `calculate(TOP op, field_type const &f, field_type const &f, index_type s ) `	| `calculate`  binary operation `op` at grid point `s`.
 *  `calculate(TOP op, field_type const &f,  index_type s )` 	| `calculate`  unary operation  `op`  at grid point `s`.
 *
 *  *
 *  @}
 */

template<typename > class FiniteDiffMethod;
template<typename > class InterpolatorLinear;

/**
 *  \ingroup manifold
 *  \brief Manifold
 */

template<size_t IFORM, typename TG, //
		template<typename > class Policy1 = FiniteDiffMethod, //
		template<typename > class Policy2 = InterpolatorLinear>
class Manifold: public TG,
				public Policy1<TG>,
				public Policy2<TG>,
				public enable_create_from_this<
						Manifold<IFORM, TG, Policy1, Policy2>>
{
public:

	typedef Manifold<IFORM, TG, Policy1, Policy2> this_type;

	typedef TG geometry_type;
	typedef Policy1<geometry_type> calculate_policy;
	typedef Policy2<geometry_type> interpolatpr_policy;

	static constexpr size_t iform = IFORM;
//
//	typedef typename geometry_type::topology_type topology_type;
//	typedef typename geometry_type::coordinates_type coordinates_type;
//	typedef typename geometry_type::id_type id_type;
//	typedef typename geometry_type::scalar_type scalar_type;

	template<typename ...Args>
	Manifold(Args && ... args) :
			geometry_type(std::forward<Args>(args)...)
	{
		calculate_policy::geometry(this);
		interpolatpr_policy::geometry(this);
	}

	Manifold(this_type const & r) = delete;

	~Manifold() = default;

	this_type & operator=(this_type const &) = delete;

	using geometry_type::load;
	using geometry_type::update;
	using geometry_type::sync;

	template<typename ...Args>
	size_t hash(Args && ...args) const
	{
		return 0;
	}

	size_t max_hash() const
	{
		return 0;
	}

	// True if domain can be partitioned into two sub-domains.
	bool is_divisible() const
	{
		return false; //range_.is_divisible();
	}

	size_t size() const
	{
		return manifold_->template dataspace<iform>().size();
	}

	template<typename TV>
	std::shared_ptr<TV> allocate()const
	{
		return sp_make_shared_array<TV>(size());
	}

	template<typename TV>
	DataSet dataset(container_type<TV> const& data_,Properties const& prop)const
	{

		return DataSet(
				{	data_, make_datatype<TV>(),
					dataspace(), prop});
	}
	template<typename TV>
	auto access (container_type<TV> & v,id_type s)const
	DECL_RET_TYPE(v.get()[s])

	template<typename TV>
	auto access (TV* v,id_type s)const
	DECL_RET_TYPE(v[s])

	template<typename TV>
	auto access (TV const & v,id_type s)const
	DECL_RET_TYPE(get_value(v,s))

	template<typename ...Args>
	auto coordinates(Args && ...args) const
	DECL_RET_TYPE((geometry_type::coordinates(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto calculate(Args && ...args) const
	DECL_RET_TYPE((calculate_policy::calculate(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto gather(Args && ...args) const
	DECL_RET_TYPE((interpolatpr_policy::gather(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto scatter(Args && ...args) const
	DECL_RET_TYPE((interpolatpr_policy::scatter(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto sample(Args && ...args)const
	DECL_RET_TYPE((interpolatpr_policy::template sample<iform>(std::forward<Args>(args)...)))

	template<typename TFun >
	void foreach(TFun const & fun )const
	{
		for(auto s:range_)
		{
			fun(s);
		}
	}
	template< typename TFun>
	void pull_back( TFun const &fun)const
	{
		for(auto s:range_)
		{
			fun(geometry_type:: coordinates(s) );
		}
	}

};

template<typename TM, typename ...Args>
std::shared_ptr<TM> make_manifold(Args && ...args)
{
	return std::make_shared<TM>(std::forward<Args>(args)...);
}

}
// namespace simpla

#endif /* MANIFOLD_H_ */
