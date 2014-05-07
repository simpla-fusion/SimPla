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
#include "../utilities/parallel.h"

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

	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::scalar_type scalar_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	//container

	typedef std::list<value_type, FixedSmallSizeAlloc<value_type> > cell_type;

	typedef std::vector<cell_type> container_type;

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

	std::string Dump(std::string const & path) const;

	void Clear(index_type s);

	void Add(index_type s, cell_type &&);

	void Add(index_type s, std::function<void(particle_type*)> const & generator);

	void Remove(index_type s, std::function<bool(particle_type const&)> const & filter);

	void Modify(index_type s, std::function<void(particle_type*)> const & foo);

	void Traversal(index_type s, std::function<void(particle_type*)> const & op);

	//***************************************************************************************************

	allocator_type GetAllocator()
	{
		return allocator_;
	}
	inline void Insert(index_type s, particle_type p)
	{
		this->at(s).emplace_back(p);
	}
	cell_type & operator[](index_type s)
	{
		return data_[mesh.Hash(s)];
	}
	cell_type const & operator[](index_type s) const
	{
		return data_[mesh.Hash(s)];
	}
	cell_type &at(index_type s)
	{
		return data_.at(mesh.Hash(s));
	}
	cell_type const & at(index_type s) const
	{
		return data_.at(mesh.Hash(s));
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
			res += v.size();
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
	template<typename TDest> void Sort(index_type id_src, TDest *dest);

};

template<typename TM, typename TParticle>
template<typename TDict, typename ...Others>
ParticlePool<TM, TParticle>::ParticlePool(mesh_type const & pmesh, TDict const & dict, Others const & ...others)
		: mesh(pmesh), isSorted_(false), allocator_(),	//
		data_(mesh.GetNumOfElements(IForm), cell_type(allocator_))
{
}

template<typename TM, typename TParticle>
ParticlePool<TM, TParticle>::~ParticlePool()
{
}

//*************************************************************************************************

template<typename TM, typename TParticle>
std::string ParticlePool<TM, TParticle>::Dump(std::string const & name) const
{
	return simpla::Dump(*this, name);
}

#define DISABLE_MULTI_THREAD

//*************************************************************************************************
template<typename TM, typename TParticle>
template<typename TDest>
void ParticlePool<TM, TParticle>::Sort(index_type id_src, TDest *dest)
{

	auto & src = this->at(id_src);

	auto pt = src.begin();

	while (pt != src.end())
	{
		auto p = pt;
		++pt;

		index_type id_dest = mesh.CoordinatesGlobalToLocalDual(&(p->x));

		p->x = mesh.CoordinatesLocalToGlobal(id_dest, p->x);

		if (id_dest != id_src)
		{
			(*dest)[id_dest].splice((*dest)[id_dest].begin(), src, p);
		}

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
		std::map<index_type,cell_type> dest;
		for(auto s:this->mesh.GetRange(IForm).Split(t_num,t_id))
		{
			this->Sort(s, &dest);
		}

		write_lock_.lock();
		for(auto & v :dest)
		{
			auto & c = this->at(v.first);

			c.splice(c.begin(), v.second);
		}
		write_lock_.unlock();
	}

	);

	isSorted_ = true;
}

template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Clear(index_type s)
{
	this->at(s).clear();
}
template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Add(index_type s, cell_type && other)
{
	this->at(s).slice(this->at(s).begin(), other);
}

template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Add(index_type s, std::function<void(particle_type*)> const & gen)
{
	particle_type p;
	gen(&p);
	this->at(s).push_back(p);

}

template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Remove(index_type s, std::function<bool(particle_type const&)> const & filter)
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
void ParticlePool<TM, TParticle>::Modify(index_type s, std::function<void(particle_type *)> const & op)
{

	for (auto & p : this->at(s))
	{
		op(&p);
	}
}

template<typename TM, typename TParticle>
void ParticlePool<TM, TParticle>::Traversal(index_type s, std::function<void(particle_type*)> const & op)
{

	for (auto const & p : this->at(s))
	{

		op(p);
	}

}
}  // namespace simpla

#endif /* PARTICLE_POOL_H_ */
