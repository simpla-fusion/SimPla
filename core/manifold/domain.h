/*
 * domain.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef DOMAIN_H_
#define DOMAIN_H_

#include <memory>
#include <type_traits>

#include "../data_interface/data_space.h"
#include "../utilities/sp_type_traits.h"
#include "../design_pattern/expression_template.h"
namespace simpla
{

template<typename TG, size_t IFORM = 0>
class Domain
{

public:
	typedef TG manifold_type;

	static constexpr size_t ndims = manifold_type::ndims; // number of dimensions of domain D

	static constexpr size_t iform = IFORM; // type of form, VERTEX, EDGE, FACE,VOLUME

	typedef Domain<manifold_type, iform> this_type;

	typedef Domain<manifold_type, iform> domain_type;

	typedef typename manifold_type::topology_type topology_type;

	typedef typename manifold_type::coordinates_type coordinates_type;

	typedef typename manifold_type::id_type id_type;

	typedef size_t difference_type; // Type for difference of two iterators

	typedef typename manifold_type::range_type range_type;

	typedef typename range_type::iterator iterator;

public:
	std::shared_ptr<const manifold_type> manifold_;
	range_type range_;

public:
	Domain() :
			manifold_(nullptr)
	{
	}
	Domain(std::shared_ptr<const manifold_type> g) :
			manifold_(g->shared_from_this()), range_(
					manifold_->template select<iform>())/*, parent_(*this)*/
	{
	}
	// Copy constructor.
	Domain(const this_type& rhs) :
			manifold_(rhs.manifold_->shared_from_this()), range_(rhs.range_)/*, parent_(rhs.parent_) */
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
		sp_swap(range_, that.range_);
	}

	template<typename ...Args>
	auto hash(Args && ...args) const
	DECL_RET_TYPE((range_.hash(std::forward<Args>(args)...)))

	auto max_hash() const
	DECL_RET_TYPE((range_.max_hash()))

	auto begin() const
	DECL_RET_TYPE((range_.begin()))

	auto end() const
	DECL_RET_TYPE((range_.end()))

//	auto rbegin() const
//	DECL_RET_TYPE((range_.rbegin()))
//
//	auto rend() const
//	DECL_RET_TYPE((range_.rend()))

	std::shared_ptr<const manifold_type> manifold() const
	{
		return manifold_;
	}

	void manifold(std::shared_ptr<manifold_type> const &m)
	{
		manifold_ = m->shared_from_this();
	}

	// True if domain can be partitioned into two sub-domains.
	bool is_divisible() const
	{
		return range_.is_divisible();
	}

	size_t size() const
	{
		return manifold_->template dataspace<iform>().size();
	}

	template<typename TV> using container_type=std::shared_ptr<TV>;
	template<typename TV>
	std::shared_ptr<TV> allocate()const
	{
		return sp_make_shared_array<TV>(size());
	}

	template<typename TV>
	DataSet dataset(container_type<TV> const& data_,Properties const& prop)const
	{
		return
		DataSet(
				{	data_, prop , make_datatype<TV>(),
					dataspace()});
	}
	template<typename TV>
	auto index_value (container_type<TV> & v,id_type s)const
	DECL_RET_TYPE(v.get()[s])

	template<typename TV>
	auto index_value (TV* v,id_type s)const
	DECL_RET_TYPE(v[s])

	template<typename TV>
	auto index_value (TV const & v,id_type s)const
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

	DataSpace dataspace() const
	{
		return manifold_->template dataspace<iform>( );
	}

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

	template<typename TOP,typename T1,typename ...Args>
	void foreach(TOP const & op, Args && ... args)const
	{
		for(auto s:range_)
		{
			op( get_value(std::forward<Args>(args),s)...);
		}
	}
	template<typename TL,typename TFun>
	void pull_back(TL & l, TFun const &fun)const
	{
		for(auto s:range_)
		{
			//FIXME geometry coordinates convert
			get_value(l,hash(s)) = sample( s,fun( coordinates(s) ));
		}
	}

}
;

template<size_t IFORM, typename TM>
Domain<TM, IFORM> make_domain(std::shared_ptr<TM> const & m)
{
	return std::move(Domain<TM, IFORM>(m));
}
template<size_t IFORM, typename TM>
Domain<TM, IFORM> make_domain()
{
	auto m = std::make_shared<TM>();
	return std::move(Domain<TM, IFORM>(m));
}
template<size_t IFORM, typename TM, typename ...Args>
Domain<TM, IFORM> make_domain(Args && ...args)
{
	auto m = std::make_shared<TM>(std::forward<Args>(args)...);
	return std::move(Domain<TM, IFORM>(m));
}

template<typename TM, size_t IFORM>
struct SubDomain: public Domain<TM, IFORM>, public std::vector<
		typename TM::index_type>
{
	typedef Domain<TM, IFORM> root_domain_type;
	typedef typename root_domain_type::index_type index_type;
	typedef SubDomain<TM, IFORM> this_type;
	typedef std::vector<index_type> storage_type;

	SubDomain(root_domain_type const & r) :
			root_domain_type(r)
	{
	}
	SubDomain(this_type const & r) :
			root_domain_type(r), storage_type(r)
	{
	}

	~SubDomain()
	{
	}

	auto begin() const
	DECL_RET_TYPE((root_domain_type::begin()))
	auto end() const
	DECL_RET_TYPE((root_domain_type::end()))
};

template<typename T> struct domain_traits
{
	typedef std::nullptr_t manifold_type;
	static constexpr size_t iform = VERTEX;
};

template<typename TM, size_t IFORM>
struct domain_traits<Domain<TM, IFORM> >
{
	typedef TM manifold_type;
	static constexpr size_t iform = IFORM;
	typedef typename manifold_type::coordinates_type coordinates_type;
	typedef typename manifold_type::index_type index_type;
};

template<typename TM, size_t IFORM>
struct domain_traits<SubDomain<TM, IFORM> >
{
	typedef TM manifold_type;
	static constexpr size_t iform = IFORM;
	typedef typename manifold_type::coordinates_type coordinates_type;
	typedef typename manifold_type::index_type index_type;

};
}
// namespace simpla

#endif /* DOMAIN_H_ */
