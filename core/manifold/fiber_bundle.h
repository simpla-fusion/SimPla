/**
 * @file fiber_bundle.h
 * @author salmon
 * @date 2015-10-18.
 */

#ifndef SIMPLA_FIBER_BUNDLE_H
#define SIMPLA_FIBER_BUNDLE_H

namespace simpla
{
template<typename ...> struct Domain;

template<typename ...> class FiberBundle;

template<typename P, typename TBase>
class FiberBundle
{
	typedef TBase base_manifold;
	typedef P point_type;
	typedef Vec3 vector_type;
	typedef Real scalar_type;
private:

	typedef FiberBundle<point_type, TBase> this_type;
	typedef TBase::range_type range_type;


	base_manifold const &m_mesh_;


public:

	typedef P point_type;

	FiberBundle(base_manifold const &b) : m_mesh_(b)
	{
	}

	FiberBundle(this_type const &other) : m_mesh_(other.m_mesh_)
	{
	}


	~FiberBundle()
	{
	}


	template<typename ...Args>
	inline typename base_manifold::point_type project(point_type const &p, Args &&...args) const
	{
		return p.x;
	}

	template<typename TV, typename ...Args>
	inline point_type lift(typename base_manifold::point_type const &x, TV const &v, Real f,
			Args &&...args) const
	{
		point_type res{x, v, f};

		return std::move(res);
	}

	template<typename ...Args>
	inline vector_type push_forward(point_type const &p, Args &&...args) const
	{
		vector_type res;
		res = p.v * p.f;
		return std::move(res);
	}

	template<typename TJ, typename ...Args>
	inline vector_type push_forward(point_type const &p, TJ *J, Args &&...args) const
	{

		return m_mesh_.scatter(J, project(p), push_forward(p, std::forward<Args>(args)...));

	}

	template<typename TF, typename ...Args>
	inline auto pull_back(point_type const &p, TF const &f, Args &&...args) const
	DECL_RET_TYPE((f(project(p, std::forward<Args>(args)...))))


	template<typename ...Args>
	inline void move(point_type *p0, Real dt, Args &&...args) const
	{
		p0->x += p0->v * dt;

	}
};

/**
 *
 *  `FiberBundle<P,M>` represents a fiber bundle \f$ \pi:P\to M\f$
 */


template<typename TBase, typename P>
struct Manifold<FiberBundle<P, TBase> >
		: public FiberBundle<P, TBase>,
				public Distributed<UnorderedSet<typename FiberBundle<P, TBase>::point_type>,
						typename TBase::range_type>
{
	typedef Distributed<UnorderedSet<typename P::point_type>, typename TBase::range_type> container_type;

	typedef FiberBundle<P, TBase> bundle_type;
private:
	typedef TBase base_manifold;
	typedef Manifold<base_manifold, FiberBundle<P, TBase> > this_type;
public:
	template<typename ...Args>
	Manifold(Args &&...args) :
			bundle_type(std::forward<Args>(args))
	{
	}

	Manifold(this_type const &other) :
			bundle_type(other), container_type(other)
	{
	}

	~Manifold()
	{
	}

	void swap(this_type const &other)
	{
		bundle_type::swap(other);
		container_type::swap(other);
	}

	template<typename TDict, typename ...Others>
	void load(TDict const &dict, Others &&...others)
	{
		bundle_type::load(dict, std::forward<Others>(others)...);
		container_type::load(dict);
	}

	template<typename OS>
	OS &print(OS &os) const
	{
		bundle_type::print(os);
		container_type::print(os);
		return os;
	}


};


}//namespace simpla

#endif //SIMPLA_FIBER_BUNDLE_H
