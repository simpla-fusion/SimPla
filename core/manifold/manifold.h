/*
 * manifold.h
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
#include "../utilities/utilities.h"
#include "../data_interface/data_set.h"
namespace simpla
{

/**
 * \addtogroup  manifold Manifold
 *    \brief   Discrete spatial-temporal space \see @ref manifold_concept
 *    @file manifold.md
 */

template<typename > class FiniteDiffMethod;
template<typename > class InterpolatorLinear;

/**
 *  \ingroup manifold
 *  \brief Manifold
 */

template<typename TG, //
		template<typename > class Policy1 = FiniteDiffMethod, //
		template<typename > class Policy2 = InterpolatorLinear>
class Manifold: public TG,
		public Policy1<TG>,
		public Policy2<TG>,
		public std::enable_shared_from_this<Manifold<TG, Policy1, Policy2>>
{
public:

	typedef Manifold<TG, Policy1, Policy2> this_type;

	typedef std::shared_ptr<this_type> holder_type;

	typedef TG geometry_type;
	typedef Policy1<geometry_type> policy1;
	typedef Policy2<geometry_type> policy2;

	typedef typename geometry_type::topology_type topology_type;
	typedef typename geometry_type::coordinates_type coordinates_type;
	typedef typename geometry_type::id_type id_type;
	typedef typename geometry_type::scalar_type scalar_type;

	static constexpr size_t ndims = topology_type::ndims;

	template<typename ...Args>
	Manifold(Args && ... args) :
			geometry_type(std::forward<Args>(args)...)
	{
		policy1::geometry(this);
		policy2::geometry(this);
	}

	~Manifold() = default;

	Manifold(this_type const & r) = delete;

	template<typename ...Args>
	static holder_type create(Args &&... args)
	{
		return std::make_shared<this_type>(std::forward<Args>(args)...);
	}

	template<size_t IFORM> struct Domain;

	using std::enable_shared_from_this<this_type>::shared_from_this;

	template<size_t IFORM> Domain<IFORM> domain() const
	{
		return std::move(
				Domain<IFORM>(
						std::dynamic_pointer_cast<const this_type>(
								shared_from_this())));
	}

	this_type & operator=(this_type const &) = delete;

//	template<typename TDict>
//	bool load(TDict const & dict)
//	{
//		VERBOSE << "Load Manifold" << std::endl;
//
//		if (!(topology_type::load(dict["Topology"])
//				&& geometry_type::load(dict["Geometry"])))
//		{
//			RUNTIME_ERROR("ERROR: Load Manifold failed!");
//
//			return false;
//		}
//		return true;
//	}

	using geometry_type::load;
	using geometry_type::update;
	using geometry_type::sync;

};

template<typename TG, //
		template<typename > class Policy1, //
		template<typename > class Policy2>
template<size_t IFORM>
struct Manifold<TG, Policy1, Policy2>::Domain
{
public:
	typedef Manifold<TG, Policy1, Policy2> manifold_type;

	static constexpr size_t ndims = manifold_type::ndims; // number of dimensions of domain D

	static constexpr size_t iform = IFORM; // type of form, VERTEX, EDGE, FACE,VOLUME

	typedef Domain<iform> this_type;

	typedef typename manifold_type::topology_type topology_type;

	typedef typename manifold_type::coordinates_type coordinates_type;

	typedef typename manifold_type::id_type id_type;

	typedef size_t difference_type; // Type for difference of two iterators

	template<typename TV> using container_type=std::shared_ptr<TV>;

	size_t b_ = 0, e_ = 1;
public:

	std::shared_ptr<const manifold_type> manifold_;

public:
	Domain() :
			manifold_(nullptr)
	{
	}
	template<typename ...Args>
	Domain(std::shared_ptr<const manifold_type> g, Args && ... args) :
			manifold_(g->shared_from_this())
	{
	}
	// Copy constructor.
	Domain(const this_type& rhs) :
			manifold_(rhs.manifold_->shared_from_this())
	{
	}
	// Split d into two sub-domains.
//	Domain(this_type& d ) :
//			manifold_(d.manifold_->shared_from_this()), range_(d.range_,
//					split_tag())
//	{
//	}

	virtual ~Domain() = default; // Destructor.

	bool is_valid() const
	{
		return manifold_ != nullptr;
	}

	void swap(this_type & that)
	{
		sp_swap(manifold_, that.manifold_);
//		sp_swap(range_, that.range_);
	}

	template<typename ...Args>
	size_t hash(Args && ...args) const
	{
		return 0;
	}

	size_t max_hash() const
	{
		return 0;
	}

	size_t const *begin() const
	{
		return &b_;
	}

	size_t const *end() const
	{
		return &e_;
	}

//	auto begin() const
//	DECL_RET_TYPE((range_.begin()))
//
//	auto end() const
//	DECL_RET_TYPE((range_.end()))

//	auto rbegin() const
//	DECL_RET_TYPE((range_.rbegin()))
//
//	auto rend() const
//	DECL_RET_TYPE((range_.rend()))

	std::shared_ptr<const manifold_type> manifold() const
	{
		return manifold_;
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
		return DataSet();
//		DataSet(
//				{	data_, prop , make_datatype<TV>(),
//					dataspace()});
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

//	this_type operator &(this_type const & D1) const // \f$D_0 \cap \D_1\f$
//	{
//		return *this;
//	}
//	this_type operator |(this_type const & D1) const // \f$D_0 \cup \D_1\f$
//	{
//		return *this;
//	}
//	bool operator==(this_type const&)
//	{
//		return true;
//	}
//	bool is_same(this_type const&);
//
//	this_type const & parent() const; // Parent domain
//
//	std::tuple<coordinates_type, coordinates_type> boundbox() const // boundbox on _this_ coordinates system
//	{
//		return manifold_->geometry_type::boundbox<iform>(range_);
//	}
//	std::tuple<nTuple<Real, 3>, nTuple<Real, 3>> cartesian_boundbox() const // boundbox on   _Cartesian_ coordinates system
//	{
//		return manifold_->geometry_type::cartesian_boundbox<iform>(range_);
//	}
//	DataSpace dataspace() const
//		{
//			return manifold_->template dataspace<IFORM>();
//		}

	template<typename ...Args>
	auto coordinates(Args && ...args) const
	DECL_RET_TYPE((manifold_->coordinates(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto calculate(Args && ...args) const
	DECL_RET_TYPE((manifold_->calculate(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto sample(Args && ... args)
	DECL_RET_TYPE((manifold_->sample(
							std::integral_constant<size_t, iform>(),
							std::forward<Args>(args)...)))

	template<typename ...Args>
	auto gather(Args && ...args) const
	DECL_RET_TYPE((manifold_->gather(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto scatter(Args && ...args) const
	DECL_RET_TYPE((manifold_->scatter(std::forward<Args>(args)...)))

	template<typename ...Args>
	auto sample(Args && ...args)const
	DECL_RET_TYPE((manifold_->template sample<iform>(std::forward<Args>(args)...)))

	template<typename TOP, typename ...Args>
	void foreach(TOP const & op, Args && ... args)const
	{
//		for(auto s:range_)
//		{
//			op( get_value(std::forward<Args>(args),s)...);
//		}
	}
	template<typename TL,typename TFun>
	void pull_back(TL & l, TFun const &fun)const
	{
//		for(auto s:range_)
//		{
//			//FIXME geometry coordinates convert
//			get_value(l,hash(s)) = sample( s,fun( coordinates(s) ));
//		}
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
