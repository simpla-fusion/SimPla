/*
 * particle_collection.h
 *
 *  Created on: 2014年1月13日
 *      Author: salmon
 */

#ifndef PARTICLE_COLLECTION_H_
#define PARTICLE_COLLECTION_H_

#include <initializer_list>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <valarray>
#include <memory>

#include "particle.h"

#include "../utilities/log.h"

//*******************************************************************************************************
namespace simpla
{
template<typename TM>
class ParticleCollection: public std::map<std::string, std::shared_ptr<ParticleBase<TM> > >
{
public:
	typedef TM mesh_type;

	typedef ParticleBase<mesh_type> particle_type;

	typedef std::map<std::string, std::shared_ptr<particle_type> > base_type;

	typedef std::function<std::shared_ptr<particle_type>(mesh_type const &)> create_fun;

	typedef ParticleCollection<mesh_type> this_type;

private:
	std::map<std::string, create_fun> factory_;
public:

	mesh_type const & mesh;

	template<typename U>
	friend std::ostream & operator<<(std::ostream & os, ParticleCollection<U> const &self);

	ParticleCollection(mesh_type const & pmesh)
			: mesh(pmesh)
	{
	}
	~ParticleCollection()
	{
	}

	void RegisterFactory(std::string const &engine_name, create_fun const &fun)
	{
		factory_.emplace(engine_name, fun);
	}
	template<typename TEngine>
	void RegisterFactory(std::string engine_name = "")
	{
		if (engine_name == "")
			engine_name = TEngine::TypeName();

		RegisterFactory(engine_name, create_fun(&CreateParticle<TEngine>));
	}

	void Deserialize(LuaObject const &cfg);

	std::ostream & Serialize(std::ostream & os) const;

	template<typename PT>
	inline void Serialize(PT &vm) const
	{
		WARNING << "UNIMPLEMENT!!";
	}

	void Sort();

	void DumpData(std::string const & path = "") const;

	template<typename ... Args> void NextTimeStep(Args const & ... args);

	template<typename TJ, typename ... Args> void Collect(TJ *J, Args const & ... args) const;
};

template<typename TM>
void ParticleCollection<TM>::Deserialize(LuaObject const &cfg)
{
	if (cfg.empty())
		return;

	Logger logger(LOG_LOG);

	logger << "Load Particles " << endl << flush << indent;

	for (auto const &p : cfg)
	{
		std::string key;

		if (!p.first.is_number())
		{
			key = p.first.template as<std::string>();
		}
		else
		{
			p.second.GetValue("Name", &key);
		}

		std::string engine = p.second.at("Engine").template as<std::string>();

		auto it = factory_.find(engine);

		if (it != factory_.end())
		{
			auto t = it->second(mesh);

			t->Deserialize(p.second);

			this->emplace(key, t);

			t->SetName(key);

		}
		else
		{
			WARNING << "I do not know how to create \"" << key << "\" particle! [engine=" << engine << "]";

			return;
		}

	}

	logger << DONE;

}
template<typename TM>

std::ostream & ParticleCollection<TM>::Serialize(std::ostream & os) const
{

	os << "Particles={ \n";

	ContainerOutPut3(os, this->begin(), this->end(),

	[](std::ostream & oos, decltype(this->begin()) const &it)->std::ostream &
	{
		oos<<"\t";

		it->second->Serialize(oos);

		oos<<"\n";

		return oos;
	});

	os << "} \n";

	return os;
}

template<typename TM>
template<typename ... Args>
void ParticleCollection<TM>::NextTimeStep(Args const & ... args)
{
	for (auto & p : *this)
	{
		LOG_CMD2(("Move Particle [" + p.first + ":" + p.second->GetTypeAsString() + "]"),
		        (p.second->NextTimeStep(args...)));
	}
}

template<typename TM>
template<typename TJ, typename ... Args>
void ParticleCollection<TM>::Collect(TJ *J, Args const & ... args) const
{
	for (auto & p : *this)
	{
		LOG_CMD2(("Collect particle [" + p.first + ":" + p.second->GetTypeAsString()

		+ "] to Form<" + ToString(TJ::IForm) + ","

		+ (is_ntuple<typename TJ::value_type>::value ? "Vector" : "Scalar") + ">!]"

		), (p.second->Collect(J, args...)));
	}
}
template<typename TM>
void ParticleCollection<TM>::Sort()
{
	for (auto & p : *this)
	{
		p.second->Sort();
	}
}

template<typename TM>
void ParticleCollection<TM>::DumpData(std::string const & path) const
{
	for (auto const &p : *this)
	{
		p.second->DumpData(path);
	}
}
template<typename TM>
std::ostream & operator<<(std::ostream & os, ParticleCollection<TM> const &self)
{
	return self.Serialize(os);
}
} //namespace simpla
#endif /* PARTICLE_COLLECTION_H_ */
