/**
 * @file particle_generator.h
 *
 * @date 2015-2-12
 * @author salmon
 */

#ifndef CORE_PARTICLE_PARTICLE_GENERATOR_H_
#define CORE_PARTICLE_PARTICLE_GENERATOR_H_
#include "../gtl/type_traits.h"
namespace simpla
{

template<typename Engine, typename XRandomEngine, typename VRandomEngine,
		typename DistFunction = Real>
struct ParticleGenerator
{

	typedef Engine engine_type;

	typedef XRandomEngine x_dist_engine;
	typedef VRandomEngine v_dist_engine;
	typedef DistFunction distribute_function;
	typedef ParticleGenerator<engine_type, x_dist_engine, v_dist_engine,
			distribute_function> this_type;

	typedef this_type generator_type;

	typedef typename engine_type::Point_s value_type;

	typedef nTuple<Real, 3> vector_type;

	x_dist_engine m_x_dist_;

	v_dist_engine m_v_dist_;

	distribute_function m_dist_function_;

	engine_type const & m_engine_;

	ParticleGenerator(engine_type const & engine, x_dist_engine const& x_dist,
			v_dist_engine const& v_dist,
			distribute_function const & dist_function = distribute_function())
			: m_engine_(engine), m_x_dist_(x_dist), m_v_dist_(v_dist), m_dist_function_(
					dist_function)
	{
	}
	~ParticleGenerator()
	{
	}

	template<typename TRNDGen>
	value_type operator()(TRNDGen & rnd_gen)
	{
		return std::move(
				gen_(rnd_gen,
						std::integral_constant<bool,
								is_callable<distribute_function, vector_type,
										vector_type>::value>()));
	}
private:

	template<typename TRNDGen>
	value_type gen_(TRNDGen & rnd_gen, std::integral_constant<bool, true>)

	{
		auto x = m_x_dist_(rnd_gen);
		auto v = m_v_dist_(rnd_gen);
		return std::move(
				m_engine_.push_forward(x, v,
						static_cast<Real>(m_dist_function_(x, v))));
	}
	template<typename TRNDGen>
	value_type gen_(TRNDGen & rnd_gen, std::integral_constant<bool, false>)
	{
		return std::move(
				m_engine_.push_forward(m_x_dist_(rnd_gen), m_v_dist_(rnd_gen),
						m_dist_function_));

	}
public:
	template<typename TRNDGen>
	struct input_iterator: public std::iterator<std::input_iterator_tag,
			value_type>
	{
		typedef TRNDGen rnd_generator;

		generator_type & m_dist_;

		rnd_generator & m_rnd_gen_;

		size_t m_count_;
		value_type m_z_;

		input_iterator(generator_type & gen, rnd_generator & rnd_gen_,
				size_t count = 0)
				: m_dist_(gen), m_count_(count), m_rnd_gen_(rnd_gen_)
		{
		}
		input_iterator(input_iterator const& other)
				: m_dist_(other.m_dist_), m_count_(other.m_count_), m_rnd_gen_(
						other.m_rnd_gen_)
		{
		}

		input_iterator(input_iterator && other)
				: m_dist_(other.m_dist_), m_count_(other.m_count_), m_rnd_gen_(
						other.m_rnd_gen_)
		{
		}

		~input_iterator()
		{
		}

		value_type const & operator*() const
		{
			return m_z_;
		}

		value_type const * operator->() const
		{
			return &m_z_;
		}

		input_iterator operator++()
		{
			m_z_ = m_dist_(m_rnd_gen_);
			++m_count_;
			return *this;
		}

		bool operator==(input_iterator const & other) const
		{
			return m_count_ == other.m_count_;
		}

		bool operator!=(input_iterator const & other) const
		{
			return (m_count_ != other.m_count_);
		}
	};

	template<typename TRNDGen>
	input_iterator<TRNDGen> begin(TRNDGen & rnd_gen, size_t pos = 0)
	{
		return std::move(input_iterator<TRNDGen>(*this, rnd_gen, pos));
	}
	template<typename TRNDGen>
	input_iterator<TRNDGen> end(TRNDGen & rnd_gen, size_t pos = 0)
	{
		return std::move(input_iterator<TRNDGen>(*this, rnd_gen, pos));
	}
};

template<typename EngineType, typename XGen, typename VGen>
ParticleGenerator<EngineType, XGen, VGen> make_particle_generator(
		EngineType const & eng, XGen const& xgen, VGen const& vgen)
{
	return std::move(ParticleGenerator<EngineType, XGen, VGen>(eng, xgen, vgen));
}

template<typename EngineType, typename XGen, typename VGen, typename DistFun>
ParticleGenerator<EngineType, XGen, VGen, DistFun> make_particle_generator(
		EngineType const & eng, XGen const& xgen, VGen const& vgen,
		DistFun const & distfun)
{
	return std::move(
			ParticleGenerator<EngineType, XGen, VGen>(eng, xgen, vgen, distfun));
}

}  // namespace simpla

#endif /* CORE_PARTICLE_PARTICLE_GENERATOR_H_ */
