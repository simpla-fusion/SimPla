/**
 * @file  kinetic_particle.h
 *
 * \date    2014年9月1日  下午2:25:26 
 * \author salmon
 */

#ifndef CORE_PARTICLE_KINETIC_PARTICLE_H_
#define CORE_PARTICLE_KINETIC_PARTICLE_H_
#include <memory>
#include "../utilities/utilities.h"
#include "../gtl/containers/sp_sorted_set.h"
#include "particle.h"
namespace simpla
{
namespace _impl
{
template<typename TDomain, typename TPoint_s>
struct particle_hasher
{
	typedef TDomain domain_type;
	typedef typename domain_type::mesh_type mesh_type;
	typedef typename mesh_type::id_type id_type;

	typedef TPoint_s value_type;

	static constexpr size_t ndims = domain_type::ndims;
	static constexpr size_t iform = domain_type::iform;
	static constexpr size_t cell_id = (iform == VOLUME) ? 7 : 0;

	mesh_type const * m_mesh_;

	particle_hasher()
			: m_mesh_(nullptr)
	{
	}
	particle_hasher(domain_type const & d)
			: m_mesh_(&d.mesh())
	{
	}
	~particle_hasher()
	{
	}

	constexpr id_type operator()(value_type const & p) const
	{
		return std::get<0>(m_mesh_->coordinates_global_to_local(p.x, cell_id));
	}

//	template<typename ...Args>
//	constexpr auto operator()(Args &&... args) const
//	DECL_RET_TYPE((m_mesh_->hash(std::forward<Args>(args)...)))
}
;
template<typename TDomain, typename TPoint_s> constexpr size_t particle_hasher<
		TDomain, TPoint_s>::iform;
template<typename TDomain, typename TPoint_s> constexpr size_t particle_hasher<
		TDomain, TPoint_s>::ndims;
template<typename TDomain, typename TPoint_s> constexpr size_t particle_hasher<
		TDomain, TPoint_s>::cell_id;

}  // namespace _impl

/**
 * @ingroup particle
 */
/**
 * @brief Sorted Particle / Kinetic Particle
 *  -  KineticParticle is a container of untracable particle  .
 *  -  KineticParticle is sorted;
 *  -  KineticParticle is an unordered container;
 */

template<typename Engine, typename TDomain, typename ...Others>
std::shared_ptr<
		Particle<TDomain, Engine,
				sp_sorted_set<typename Engine::Point_s,
						_impl::particle_hasher<TDomain, typename Engine::Point_s> > > > make_kinetic_particle(
		TDomain const & domain, Others && ...others)
{
	typedef Particle<TDomain, Engine,
			sp_sorted_set<typename Engine::Point_s,
					_impl::particle_hasher<TDomain, typename Engine::Point_s> > > particle_type;
	auto res = std::make_shared<particle_type>(domain,
			std::forward<Others>(others)...);

	res->hash_function(
			_impl::particle_hasher<TDomain, typename Engine::Point_s>(domain));

	return res;
}

} // namespace simpla

//template<typename Engine, typename TDomain>
//struct KineticParticle: public SpObject,
//						public Engine,
//						public enable_create_from_this<
//								KineticParticle<Engine, TDomain>>
//{
//
//public:
//
//	typedef Engine engine_type;
//
//	typedef KineticParticle<engine_type, Others...> this_type;
//
//	typedef typename engine_type::Point_s Point_s;
//
//	template<typename ...Args>
//	KineticParticle(domain_type const & pdomain, Args && ...);
//
//	KineticParticle(this_type const&);
//
//	~KineticParticle();
//
//	using engine_type::properties;
//
//	std::ostream& print(std::ostream & os) const
//	{
//		engine_type::print(os);
//		return os;
//	}
//
//	template<typename TDict, typename ...Others>
//	void load(TDict const & dict, Others && ...others);
//
//	bool update();
//
//	void sync();
//
//	DataSet dataset() const;
//
//	static std::string get_type_as_string_staic()
//	{
//		return engine_type::get_type_as_string();
//	}
//	std::string get_type_as_string() const
//	{
//		return get_type_as_string_staic();
//	}
//
//	//! @name   @ref splittable
//	//! @{
//
//	KineticParticle(this_type&, split);
//
//	bool empty() const;
//
//	bool is_divisible() const;
//
//	//! @}
//	/**
//	 *
//	 * @param args arguments
//	 *
//	 * - Semantics
//	 @code
//	 for( Point_s & point: all particle)
//	 {
//	 engine_type::next_time_step(& point,std::forward<Args>(args)... );
//	 }
//	 @endcode
//	 *
//	 */
//	template<typename ...Args>
//	void next_time_step(Args && ...args);
//
//	/**
//	 *  push_back and emplace will invalid data in the cache
//	 * @param args
//	 */
//
//	template<typename TFun, typename ...Args>
//	void foreach(TFun const & fun, Args && ...);
//
//	template<typename ...Args>
//	void push_back(Args && ...args)
//	{
//		if (cache_is_valid())
//		{
//			download_cache();
//		}
//
//		data_.push_back(std::forward<Args>(args)...);
//		cache_is_valid_ = false;
//	}
//
//	template<typename ...Args>
//	void emplac_back(Args && ...args)
//	{
//		if (cache_is_valid())
//		{
//			download_cache();
//		}
//
//		data_.emplac_back(std::forward<Args>(args)...);
//
//		cache_is_valid_ = false;
//	}
//
//	template<typename TIterator>
//	void insert(TIterator const & b, TIterator const &e)
//	{
//		if (!cache_is_valid())
//		{
//			upload_cache();
//		}
//
//		upload_to_cache(b, e);
//
//	}
//
//	void upload_cache();
//
//	void download_cache();
//
//	void sort();
//
//private:
//
//	domain_type domain_;
//
//	std::vector<Point_s> data_;
//
//};
//
//template<typename Engine, typename TDomain>
//template<typename ... Args>
//KineticParticle<Engine, TDomain>::KineticParticle(domain_type const & pdomain,
//		Args && ...args) :
//		engine_type(std::forward<Args>(args)...), domain_(pdomain)
//{
//
//}
//
//template<typename Engine, typename TDomain>
//KineticParticle<Engine, TDomain>::~KineticParticle()
//{
//}
//template<typename Engine, typename TDomain>
//template<typename TDict, typename ...Others>
//void KineticParticle<Engine, TDomain>::load(TDict const & dict,
//		Others && ...others)
//{
//	engine_type::load(dict, std::forward<Others>(others)...);
//}
//
//template<typename Engine, typename TDomain>
//bool KineticParticle<Engine, TDomain>::update()
//{
//
//	hash_fun_ = [& ](Point_s const & p)->id_type
//	{
//		return std::get<0>(domain_.manifold()->coordinates_global_to_local(
//						std::get<0>(engine_type::pull_back(p))));
//	};
//
//	return true;
//}
//
//template<typename Engine, typename TDomain>
//void KineticParticle<Engine, TDomain>::sync()
//{
//}
//
//template<typename Engine, typename TDomain>
//template<typename TFun, typename ...Args>
//void KineticParticle<Engine, TDomain>::foreach(TFun const & fun,
//		Args && ...args)
//{
//	if (!is_cached(std::forward<Args>(args)...))
//	{
//		foreach(fun, Cache<Args>(hint,args)...);
//	}
//	else
//	{
//		parallel_foreach(data_,
//
//		[&](std::pair<id_type,point_index_s*> & item)
//		{
//			point_index_s p = item.second;
//			id_type hint = item.first;
//			while (p != nullptr)
//			{
//				fun(p->data, std::forward<Args>(args)...);
//				p=p->next;
//			}
//
//		}
//
//		);
//	}
//}

#endif /* CORE_PARTICLE_KINETIC_PARTICLE_H_ */
