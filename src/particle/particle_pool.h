/*
 * particle_pool.h
 *
 *  Created on: 2014年4月29日
 *      Author: salmon
 */

#ifndef PARTICLE_POOL_H_
#define PARTICLE_POOL_H_
#include "../utilities/log.h"
#include "../io/data_stream.h"
#include "../utilities/memory_pool.h"
#include "../utilities/singleton_holder.h"
#include "../utilities/type_utilites.h"

#include "../parallel/parallel.h"

#include "save_particle.h"
#ifndef NO_STD_CXX
//need  libstdc++
#include <ext/mt_allocator.h>
template<typename T> using FixedSmallSizeAlloc=__gnu_cxx::__mt_alloc<T>;
#endif

namespace simpla
{

//*******************************************************************************************************

template<typename TM, typename TParticle>
class ParticlePool
{
	std::mutex write_lock_;

public:
	static constexpr int IForm = VERTEX;

	typedef TM mesh_type;
	typedef TParticle particle_type;
	typedef ParticlePool<mesh_type, particle_type> this_type;
	typedef particle_type value_type;

	typedef typename mesh_type::iterator mesh_iterator;
	typedef typename mesh_type::scalar_type scalar_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	//container

	typedef std::list<value_type, FixedSmallSizeAlloc<value_type> > cell_type;

	typedef std::map<mesh_iterator, cell_type> container_type;

	typedef typename container_type::iterator iterator;

	typedef typename container_type::const_iterator const_iterator;

	typedef typename cell_type::allocator_type allocator_type;

public:
	mesh_type const & mesh;
	//***************************************************************************************************
	// Constructor
	template<typename TDict, typename ...Args> ParticlePool(mesh_type const & pmesh, TDict const & dict,
			Args const & ...others);

	// Destructor
	~ParticlePool();

	std::string Save(std::string const & path) const;

	void Clear(mesh_iterator s);

	void Add(mesh_iterator s, cell_type &&);

	void Add(mesh_iterator s, std::function<void(particle_type*)> const & generator);

	void Remove(mesh_iterator s, std::function<bool(particle_type const&)> const & filter);

	void Merge(container_type * other);

	void Modify(mesh_iterator s, std::function<void(particle_type*)> const & foo);

	void Traversal(mesh_iterator s, std::function<void(particle_type*)> const & op);

	void UpdateGhosts(MPI_Comm comm = MPI_COMM_NULL);

	//***************************************************************************************************

	allocator_type GetAllocator()
	{
		return allocator_;
	}
	inline void Insert(mesh_iterator s, particle_type p)
	{
		data_[s].emplace_back(p);
	}
	cell_type & operator[](mesh_iterator s)
	{
		return data_.at(s);
	}
	cell_type const & operator[](mesh_iterator s) const
	{
		return data_.at(s);
	}
	cell_type &at(mesh_iterator s)
	{
		return data_.at(s);
	}
	cell_type const & at(mesh_iterator s) const
	{
		return data_.at(s);
	}

//	iterator begin()
//	{
//		return data_.begin();
//	}
//
//	iterator end()
//	{
//		return data_.end();
//	}
//
//	const_iterator begin() const
//	{
//		return data_.begin();
//	}
//
//	const_iterator end() const
//	{
//		return data_.end();
//	}
//***************************************************************************************************

	void Sort();

	bool IsSorted() const
	{
		return isSorted_;
	}

	void NeedSort()
	{
		isSorted_ = false;
	}

	size_t size() const
	{
		size_t res = 0;

		for (auto const & v : data_)
		{
			res += v.second.size();
		}
		return res;
	}

	container_type const & GetTree() const
	{
		return data_;
	}

	void WriteLock()
	{
		write_lock_.lock();
	}
	void WriteUnLock()
	{
		write_lock_.unlock();
	}

private:

	bool isSorted_;

	allocator_type allocator_;
	container_type data_;

	/**
	 *  resort particles in cell 's', and move out boundary particles to 'dest' container
	 * @param
	 */
	template<typename TSrc, typename TDest> void Sort_(TSrc *, TDest *dest);

};

/***
 * TODO:  We need a  thread-safe and  high performance allocator for
 *    std::map<mesh_iterator,std::list<allocator> > !!
 */
template<typename TM, typename TParticle>
template<typename TDict, typename ...Others>
ParticlePool<TM, TParticle>::ParticlePool(mesh_type const & pmesh, TDict const & dict, Others const & ...others) :
		mesh(pmesh), isSorted_(false), allocator_()
{
}

template<typename TM, typename TParticle>
ParticlePool<TM, TParticle>::~ParticlePool()
{
}

//*************************************************************************************************

template<typename TM, typename TParticle>
std::string ParticlePool<TM, TParticle>::Save(std::string const & name) const
{
	return simpla::Save(name, *this);
}

//*************************************************************************************************
template<typename TM, typename TParticle>
template<typename TSrc, typename TDest>
void ParticlePool<TM, TParticle>::Sort_(TSrc * p_src, TDest *p_dest)
{

	auto & src = *p_src;
	auto & dest = *p_dest;

	auto pt = src.begin();

	while (pt != src.end())
	{
		auto p = pt;
		++pt;

		mesh_iterator id_dest = mesh.CoordinatesGlobalToLocalDual(&(p->x));

		p->x = mesh.CoordinatesLocalToGlobal(id_dest, p->x);

		dest[id_dest].splice(dest[id_dest].begin(), src, p);

	}

}

template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Sort()
{

	if (IsSorted())
		return;

	VERBOSE << "Particle sorting is enabled!";

	ParallelDo(

	[this](int t_num,int t_id)
	{
		container_type dest;
		for(auto s:this->mesh.GetRange(IForm).Split(t_num,t_id))
		{
			this->Sort_(&(this->at(s)), &dest);
		}
		Merge(&dest);
	}

	);

	isSorted_ = true;

	UpdateGhosts();

}
template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::UpdateGhosts(MPI_Comm comm)
{
}
template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Clear(mesh_iterator s)
{
	data_.erase(s);
}
template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Add(mesh_iterator s, cell_type && other)
{
	data_[s].slice(this->at(s).begin(), other);
}

template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Add(mesh_iterator s, std::function<void(particle_type*)> const & gen)
{
	particle_type p;
	gen(&p);
	data_[s].push_back(p);

}
template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Merge(container_type * dest)

{
	this->write_lock_.lock();
	for (auto & v : *dest)
	{
		auto & c = this->at(v.first);
		c.splice(c.begin(), v.second);
	}
	this->write_lock_.unlock();

}
template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Remove(mesh_iterator s, std::function<bool(particle_type const&)> const & filter)
{
	auto & cell = this->at(s);

	auto pt = cell.begin();

	while (pt != cell.end())
	{
		if (filter(*pt))
		{
			pt = cell.erase(pt);
		}
		else
		{
			++pt;
		}

	}
}
template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Modify(mesh_iterator s, std::function<void(particle_type *)> const & op)
{

	for (auto & p : this->at(s))
	{
		op(&p);
	}
}

template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Traversal(mesh_iterator s, std::function<void(particle_type*)> const & op)
{

	for (auto const & p : this->at(s))
	{
		op(p);
	}

}
}  // namespace simpla

#endif /* PARTICLE_POOL_H_ */
