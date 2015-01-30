/**
 * @file  mesh.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef CORE_DIFF_GEOMETRY_MESH_H_
#define CORE_DIFF_GEOMETRY_MESH_H_
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
 * \addtogroup  mesh Mesh
 *  @{
 *    \brief   Discrete spatial-temporal space
 *
 * ## Summary
 *  Mesh
 * ## Requirements
 *
 ~~~~~~~~~~~~~{.cpp}
 template <typename Geometry,template<typename> class Policy1,template<typename> class Policy2>
 class Mesh:
 public Geometry,
 public Policy1<Geometry>,
 public Policy2<Geometry>
 {
 .....
 };
 ~~~~~~~~~~~~~
 * The following table lists requirements for a Mesh type `M`,
 *
 *  Pseudo-Signature  		| Semantics
 *  ------------------------|-------------
 *  `M( const M& )` 		| Copy constructor.
 *  `~M()` 				    | Destructor.
 *  `geometry_type`		    | Geometry type of manifold, which describes coordinates and metric
 *  `topology_type`		    | Topology structure of manifold,   topology of grid points
 *  `coordiantes_type` 	    | data type of coordinates, i.e. nTuple<3,Real>
 *  `index_type`			| data type of the index of grid points, i.e. unsigned long
 *  `Domain  domain()`	    | Root domain of manifold
 *
 *
 * Mesh policy concept {#concept_manifold_policy}
 * ================================================
 *   Poilcies define the behavior of manifold , such as  interpolator or calculus;
 ~~~~~~~~~~~~~{.cpp}
 template <typename Geometry > class P;
 ~~~~~~~~~~~~~
 *
 *  The following table lists requirements for a Mesh policy type `P`,
 *
 *  Pseudo-Signature  	   | Semantics
 *  -----------------------|-------------
 *  `P( Geometry  & )` 	   | Constructor.
 *  `P( P const  & )`	   | Copy constructor.
 *  `~P( )` 			   | Copy Destructor.
 *
 * ## Interpolator policy
 *   Interpolator, map between discrete space and continue space, i.e. Gather & Scatter
 *
 *    Pseudo-Signature  	   | Semantics
 *  ---------------------------|-------------
 *  `gather(field_type const &f, coordinates_type x  )` 	    | gather data from `f` at coordinates `x`.
 *  `scatter(field_type &f, coordinates_type x ,value_type v)` 	| scatter `v` to field  `f` at coordinates `x`.
 *
 * ## Calculus  policy
 *  Define calculus operation of  fields on the manifold, such  as algebra or differential calculus.
 *  Differential calculus scheme , i.e. FDM,FVM,FEM,DG ....
 *
 *
 *  Pseudo-Signature  		| Semantics
 *  ------------------------|-------------
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
 *  \brief Mesh
 */

template<size_t IFORM, typename TG, //
		template<typename > class CalculusPolicy = FiniteDiffMethod, //
		template<typename > class InterpolatorPlolicy = InterpolatorLinear, //
		template<typename > class ContainerPolicy = std::shared_ptr>
class Mesh: public CalculusPolicy<TG>, public InterpolatorPlolicy<TG>
{

public:

	typedef Mesh<IFORM, TG, CalculusPolicy, InterpolatorPlolicy> this_type;

	typedef TG geometry_type;
	typedef typename geometry_type::id_type id_type;
	typedef CalculusPolicy<geometry_type> calculate_policy;
	typedef InterpolatorPlolicy<geometry_type> interpolatpr_policy;

	static constexpr size_t iform = IFORM;
	template<typename TV> using field_value_type=typename
	std::conditional<iform==VERTEX || iform ==VOLUME,TV,nTuple<TV,3>>::type;

	typedef typename geometry_type::template Range<iform> range_type;
	typedef typename range_type::const_iterator const_iterator;

private:
	typename geometry_type::const_holder geometry_;
	range_type range_;
public:

	Mesh() :
			geometry_(nullptr)
	{
	}

	Mesh(geometry_type const & geo) :
			geometry_(geo.shared_from_this()), range_(
					geo.template range<iform>())
	{
	}

	Mesh(this_type const & other) :
			geometry_(other.geometry_)
	{
	}

	~Mesh() = default;

	std::string get_type_as_string() const
	{
		return "Mesh<" + geometry_->get_type_as_string() + ">";
	}
	this_type & operator=(this_type const & other)
	{
		geometry_ = other.geometry_->shared_from_this();
		return *this;
	}

	template<size_t J> using clone_type= Mesh<J,TG, CalculusPolicy, InterpolatorPlolicy>;

	template<size_t J>
	clone_type<J> clone() const
	{
		return clone_type<J>(*geometry_);
	}

	/** @name Range Concept
	 * @{
	 */

	Mesh(this_type & other, op_split) :
			geometry_(other.geometry_)
//	, range_(other.range_, op_split())
	{
	}
	bool is_divisible() const
	{
		return false; //range_.is_divisible();
	}
	bool empty() const
	{
		return false;
	}

	template<typename TFun>
	void serial_foreach(TFun const & fun) const
	{
		for (auto s : range_)
		{
			CHECK(s);
			fun(s);
		}
	}

	template<typename TFun, typename ...Args>
	void serial_foreach(TFun const &fun, Args &&...args) const
	{
		for (auto s : range_)
		{
			fun(get_value(std::forward<Args>(args),s)...);
		}
	}

	template<typename ...Args>
	void foreach(Args &&...args) const
	{
//		parallel_for(*this, [&](this_type const & sub_m)
//		{
		this->serial_foreach(std::forward<Args>(args) ...);
//		});
	}

	template<typename TFun, typename TContainre>
	void serial_pull_back(TFun const &fun, TContainre & data) const
	{
//		for (auto s : range_)
//		{
//			access(data, s) = sample(fun(geometry_->coordinates(s)), s);
//		}
	}

	template<typename ...Args>
	void pull_back(Args &&...args) const
	{
//		parallel_for(*this, [&](this_type const & sub_m)
//		{
//			sub_m.serial_pull_back(std::forward<Args>(args) ...);
//		});

	}

	/**
	 * @}
	 */

	template<typename ...Args>
	size_t hash(Args && ...args) const
	{
		return geometry_->hash(std::forward<Args>(args)...);
	}

	size_t max_hash() const
	{
		return geometry_->template max_hash<iform>();
	}
	template<typename ...Args>
	id_type id(Args && ...args) const
	{
		return geometry_->id(std::forward<Args>(args)...);
	}

	constexpr id_type id(id_type s) const
	{
		return s;
	}

	template<typename ...Args>
	auto calculate(Args && ...args) const
	DECL_RET_TYPE((calculate_policy::calculate(
							*geometry_,std::forward<Args>(args)...)))

	template<typename TOP, typename TL, typename TR>
	void calculate(
			_Field<AssignmentExpression<TOP, TL, TR> > const & fexpr) const
	{
		CHECK("=====");
		for (auto s : range_)
		{
			CHECK(s);
			fexpr.op_(fexpr.lhs, fexpr.rhs, s);
		}
	}

	range_type const & range() const
	{
		return range_;
	}
	const_iterator begin() const
	{
		return range_.begin();
	}
	const_iterator end() const
	{
		return range_.end();
	}

	template<typename ...Args>
	auto coordinates(Args && ...args) const
	DECL_RET_TYPE(( geometry_->coordinates(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto gather(Args && ...args) const
	DECL_RET_TYPE((interpolatpr_policy::gather(
							*geometry_,std::forward<Args>(args)...)))

	template<typename ...Args>
	auto scatter(Args && ...args) const
	DECL_RET_TYPE((interpolatpr_policy::scatter(
							*geometry_,std::forward<Args>(args)...)))

//	template<typename ...Args>
//	auto sample(Args && ...args) const
//	DECL_RET_TYPE(
//			(geometry_type:: sample<iform>(
//							*geometry_,std::forward<Args>(args)...)))

};

template<size_t IFORM, size_t I, typename ...Args>
std::shared_ptr<Mesh<IFORM, Args...>> clone_mesh(Mesh<I, Args...> const& m)
{

}

template<size_t IFORM, typename TG>
std::shared_ptr<Mesh<IFORM, TG>> create_mesh(TG const & geo)
{
	return std::make_shared<Mesh<IFORM, TG>>(geo);
}

}
// namespace simpla

#endif /* CORE_DIFF_GEOMETRY_MESH_H_ */
