/*
 * particle_trajectory.h
 *
 *  Created on: 2014年1月13日
 *      Author: salmon
 */

#ifndef PARTICLE_TRAJECTORY_H_
#define PARTICLE_TRAJECTORY_H_

#include "../fetl/fetl.h"
#include "particle.h"

namespace simpla
{

template<typename Engine>
class ParticleTrajcetory: public Engine, public ParticleBase<typename Engine::mesh_type>
{

public:
	typedef Engine engine_type;

	typedef ParticleBase<typename Engine::mesh_type> base_type;

	typedef Particle<engine_type> this_type;

	typedef typename engine_type::mesh_type mesh_type;

	typedef typename engine_type::Point_s particle_type;

	typedef particle_type value_type;

	DEFINE_FIELDS(mesh_type)

//container
	typedef std::vector<particle_type> cell_type;

	typedef std::vector<cell_type> container_type;

	typedef Point_s * iterator;

public:
	mesh_type const &mesh;

private:

	std::shared_ptr<Point_s> data_;

	iterator top_;

	size_t num_;

	size_t cache_depth_;

public:
	ParticleTrajcetory(mesh_type const & m);

	virtual ~ParticleTrajcetory();

	virtual std::string GetTypeAsString() const
	{
		return engine_type::GetTypeAsString();
	}
	value_type & operator[](size_t s)
	{
		return top_->at(s);
	}
	value_type const & operator[](size_t s) const
	{
		return top_->at(s);
	}

	template<typename ...Args> inline void Insert(size_t s, Args const & ...args)
	{
		top_->emplace_back(engine_type::Trans(std::forward<Args const &>(args)...));
	}

	template<typename ...Args> void NextTimeStep(Real dt, Args const &... args);

	void DumpData(std::string const &path) const override;

	void Deserialize(LuaObject const &cfg) override;

	template<typename ...Args>
	void LoadParticle(LuaObject const &cfg, Args const & ...args);

	std::ostream & Serialize(std::ostream & os) const override;

	void Update();

	void SetCacheDepth(size_t d)
	{
		cache_depth_ = d;
	}
	size_t GetCacheDepth() const
	{
		return cache_depth_;
	}

	void SetNumberOfParticle(size_t d)
	{
		num_ = d;
	}
	size_t GetNumberOfParticle() const
	{
		return num_;
	}

}
;
template<typename Engine>
ParticleTrajcetory<Engine>::ParticleTrajcetory(mesh_type const &m)
		: mesh(m), top_(nullptr), num_(0), cache_depth_(0)
{

}

template<typename Engine>
ParticleTrajcetory<Engine>::~ParticleTrajcetory()
{

}

template<class Engine>
void ParticleTrajcetory<Engine>::Update()
{
	size_t s = num_ * cache_depth_;

	data_ = std::shared_ptr<Point_s>(new value_type[num_ * cache_depth_]);

	top_ = data_.get();
}

template<class Engine>
void ParticleTrajcetory<Engine>::Deserialize(LuaObject const &cfg)
{

	engine_type::Deserialize(cfg);

	cache_depth_ = cfg["CacheDepth"].as<size_t>(100);

	num_ = cfg["NumberOfParticle"].as<size_t>(100);

	Update();

}
template<class Engine>
template<typename ...Args>
void ParticleTrajcetory<Engine>::LoadParticle(LuaObject const &cfg, Args const & ...args)
{

	nTuple<6, Real> z;
	nTuple<3, Real> x, v;

	LuaObject lua = cfg["data"];

	for (size_t s = 0; s < num_; ++s)
	{
		if (lua.is_table())
		{
			lua[s].as(&z);
		}
		else if (lua.is_function())
		{
			lua(s).as(&z);
		}

		x[0] = z[0];
		x[1] = z[1];
		x[2] = z[2];
		v[0] = z[3];
		v[1] = z[4];
		v[2] = z[5];

		engine_type::Trans(x, v, &(top_[s]), std::forward<Args const &>(args)...);
	}

}

template<typename Engine>
template<typename ...Args>
void ParticleTrajcetory<Engine>::NextTimeStep(Real dt, Args const &... args)
{

	auto next = top_ + num_;

	if (next == data_.get() + num_ * cache_depth_)
	{
		next = data_.get();
	}

	std::copy(top_, top_ + num_, next);

	top_ = next;

	for (value_type * p = top_, pe = top_ + num_; p < pe; ++p)
	{
		engine_type::NextTimeStep(p, dt, std::forward<Args const&>(args)...);
	}

}
template<class Engine>
void ParticleTrajcetory<Engine>::DumpData(std::string const &path) const
{
	if (top_ + num_ == data_.get() + num_ * cache_depth_)
	{
		size_t dims[2] = { cache_depth_, num_ };
		LOGGER << Data(v, base_type::GetName(), 2, dims,true);
	}
}
}
// namespace simpla

#endif /* PARTICLE_TRAJECTORY_H_ */
