/*
 * domain.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef DOMAIN_H_
#define DOMAIN_H_
#include "../utilities/sp_type_traits.h"

namespace simpla
{
class split_tag;
template<typename ...> class _Field;

template<typename TG, unsigned int IFORM>
class Domain
{

public:

//	static constexpr unsigned int ndims = geometry_type::ndims; // number of dimensions of domain D

	static constexpr unsigned int iform = IFORM; // type of form, VERTEX, EDGE, FACE,VOLUME

	typedef TG manifold_type;

	typedef Domain<manifold_type, iform> this_type;
	typedef Domain<manifold_type, iform> domain_type;

	typedef typename manifold_type::topology_type topology_type;

	typedef typename manifold_type::coordinates_type coordinates_type;

	typedef typename manifold_type::index_type index_type;

	typedef size_t difference_type; // Type for difference of two iterators

	typedef typename topology_type::iterator iterator;
	typedef typename topology_type::range_type range_type;

public:
	manifold_type const& manifold_;
private:
	Domain<TG, IFORM> const & parent_;
	range_type range_;
public:
	Domain(manifold_type const & g) :
			manifold_(g), parent_(*this), range_(g.select(iform))
	{
		;
	}
	// Copy constructor.
	Domain(const this_type& rhs) :
			manifold_(rhs.manifold_), parent_(rhs.parent_), range_(rhs.range_)
	{
	}
	Domain(this_type& d, split_tag); // Split d into two sub-domains.

	~Domain() = default; // Destructor.

	void swap(this_type& rhs);

	manifold_type const & manifold() const
	{
		return manifold_;
	}

	bool empty() const; // True if domain is empty.

	bool is_divisible() const; // True if domain can be partitioned into two subdomains.

	iterator const & begin() const // First item in domain.
	{
		return std::get<0>(range_);
	}

	iterator const &end() const // One past last item in domain.
	{
		return std::get<1>(range_);
	}

	this_type operator &(this_type const & D1) const // \f$D_0 \cap \D_1\f$
	{
		return *this;
	}
	this_type operator |(this_type const & D1) const // \f$D_0 \cup \D_1\f$
	{
		return *this;
	}
	bool operator==(this_type const&)
	{
		return true;
	}
	bool is_same(this_type const&);

	this_type const & parent() const; // Parent domain

	std::tuple<coordinates_type, coordinates_type> boundbox() const // boundbox on _this_ coordinates system
	{
		return manifold_.geometry_type::boundbox<iform>(range_);
	}
	std::tuple<nTuple<3, Real>, nTuple<3, Real>> cartesian_boundbox() const // boundbox on   _Cartesian_ coordinates system
	{
		return manifold_.geometry_type::cartesian_boundbox<iform>(range_);
	}
	size_t hash(index_type s) const // get relative  position of grid point  in the memory
	{
		return 0; // manifold_.topology_type::hash<iform>(range_, s);
	}
	size_t max_hash() const	 // get max number of grid points in memory
	{
		return 0;	 //manifold_.topology_type::max_hash();
	}

public:
	template<typename TL, typename TR>
	void assign(TL & lhs, TR const & rhs) const
	{
		parallel_for(get_domain(lhs) & get_domain(rhs),

		[& ](this_type const &r)
		{
			for(auto const & s:r)
			{
				lhs[s]= manifold_.get_value( rhs,s );
			}
		});
	}

}
;

template<typename TG, unsigned int IFORM, typename ...Args>
auto calculate(Domain<TG, IFORM> const & d, Args && ... args)
DECL_RET_TYPE((d.manifold().calculate(std::forward<Args>(args)...)))

template<typename TG, unsigned int IFORM, typename ...Args>
auto gather(Domain<TG, IFORM> const & d, Args && ... args)
DECL_RET_TYPE((d.manifold().gather(std::forward<Args>(args)...)))

template<typename TG, unsigned int IFORM, typename ...Args>
auto scatter(Domain<TG, IFORM> const & d, Args && ... args)
DECL_RET_TYPE((d.manifold().scatter(std::forward<Args>(args)...)))
}
// namespace simpla

#endif /* DOMAIN_H_ */
