/*
 * model.h
 *
 *  created on: 2013-12-15
 *      Author: salmon
 */

#ifndef MODEL_H_
#define MODEL_H_

#include <algorithm>
#include <bitset>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../utilities/primitives.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"
#include "../utilities/utilities.h"
#include "../utilities/sp_iterator_filter.h"
#include "../utilities/sp_range_filter.h"
#include "../utilities/sp_type_traits.h"

#include "../numeric/pointinpolygon.h"
#include "../numeric/geometric_algorithm.h"

namespace simpla
{

/**
 *  \defgroup  Model Model
 *   \brief Geometry modeling
 */

/**
 *  \ingroup Model
 *  \brief Model
 */
template<typename TM>
class Model: public TM
{
private:
	bool is_ready_ = false;
public:
	static constexpr size_t MAX_NUM_OF_MEIDA_TYPE = std::numeric_limits<
			unsigned long>::digits;
	typedef TM manifold_type;
	static constexpr size_t ndims = manifold_type::ndims;
	typedef std::bitset<MAX_NUM_OF_MEIDA_TYPE> material_type;
	typedef typename manifold_type::iterator iterator;
	typedef typename manifold_type::coordinates_type coordinates_type;
	typedef typename manifold_type::compact_index_type compact_index_type;

	const material_type null_material;

	std::map<compact_index_type, material_type> material_;
	std::map<std::string, material_type> registered_material_;

	size_t max_material_;
public:

	enum
	{
		NONE = 0, VACUUM = 1, PLASMA, CORE, BOUNDARY, LIMITER,
		// @NOTE: add materials for different physical area or media
		CUSTOM = 20
	};

	Model() :
			manifold_type(), null_material(0), max_material_(CUSTOM + 1)
	{
		registered_material_.emplace("NONE", null_material);

		registered_material_.emplace("Vacuum", material_type(1 << VACUUM));
		registered_material_.emplace("Plasma", material_type(1 << PLASMA));
		registered_material_.emplace("Core", material_type(1 << CORE));
		registered_material_.emplace("Boundary", material_type(1 << BOUNDARY));
		registered_material_.emplace("Limiter", material_type(1 << LIMITER));

	}
	~Model()
	{
	}

	bool empty() const
	{
		return material_.empty();
	}

	bool is_ready() const
	{
		return is_ready_ && manifold_type::is_ready();
	}
	template<typename TDict>
	bool load(TDict const & dict)
	{
		manifold_type::load(dict["Mesh"]);

		manifold_type::update();

		if (dict["Material"].is_table())
		{
			for (auto const & item : dict["Material"])
			{
				modify(item.second);
			}
		}

		return true;
	}
	std::string save(std::string const & path) const
	{
		return manifold_type::save(path);
	}
	template<typename OS>
	OS & print(OS &os) const
	{
		os << std::endl

		<< " Type= \"" << manifold_type::get_type_as_string() << "\","
				<< std::endl;

		manifold_type::print(os);

		return os;
	}

	void update()
	{
		manifold_type::update();
		is_ready_ = manifold_type::is_ready();
	}

	operator bool() const
	{
		return !material_.empty() && is_ready();
	}

	material_type RegisterMaterial(std::string const & name)
	{
		material_type res;
		if (registered_material_.find(name) != registered_material_.end())
		{
			res = registered_material_[name];
		}
		else if (max_material_ < MAX_NUM_OF_MEIDA_TYPE)
		{
			res.set(max_material_);
			++max_material_;
		}
		else
		{
			RUNTIME_ERROR("Too much media Type");
		}
		return res;
	}

	size_t get_num_of_material_type() const
	{
		return max_material_;
	}

	material_type get_material(material_type const & m) const
	{
		return m;
	}

	material_type get_material(size_t material_pos) const
	{
		material_type res;
		res.set(material_pos);
		return std::move(res);
	}
	material_type get_material(std::string const &name) const
	{

		if (name == "" || name == "NONE")
		{
			return null_material;
		}
		material_type res;

		try
		{
			res = registered_material_.at(name);

		} catch (...)
		{
			RUNTIME_ERROR("Unknown material name : " + name);
		}
		return std::move(res);
	}

	material_type get(compact_index_type s) const;

	material_type operator[](compact_index_type s) const
	{
		return get(s);
	}

	void clear()
	{
		material_.clear();
	}

	typedef std::function<bool(compact_index_type const &)> pred_fun_type;

	template<typename TDict>
	void modify(TDict const& dict)
	{

		std::function<material_type(material_type const &)> fun;

		auto range = select_by_config(VERTEX, dict["Select"]);

		auto material_name = dict["Value"].template as<std::string>("");
		auto material = get_material(material_name);

		std::string op = dict["Op"].template as<std::string>("");

		if (op == "Set")
		{
			Set(range, material);
		}
		else if (op == "Unset")
		{
			Unset(range, material);
		}
		else if (op == "Erase")
		{
			Erase(range);
		}
		LOGGER << op << " material [" << material_name << "]" << DONE;

	}

	template<typename TR>
	void modify(TR const & r,
			std::function<material_type(material_type const &)> const &fun)
	{
		for (auto s : r)
		{
			material_[s] = fun(material_[s]);
			if (material_[s] == null_material)
				material_.erase(s);
		}
	}

	template<typename TR>
	void Erase(TR const & r)
	{
		for (auto s : r)
		{
			material_.erase(s);
		}
	}

	template<typename TR, typename M>
	void Set(TR const & r, M const& material)
	{
		auto t = get_material(material);
		modify(r, [=](material_type const & m)->material_type
		{	return m|t;});
	}

	template<typename TR, typename M>
	void Unset(TR const & r, M const& material)
	{
		auto t = get_material(material);
		modify(r, [=](material_type const & m)->material_type
		{	return m&(~t);});
	}

	template<typename TR, typename TDict>
	FilterRange<TR> select_by_config(TR const& range, TDict const& dict) const;

	template<typename TR>
	FilterRange<TR> SelectByFunction(TR const& range,
			std::function<bool(coordinates_type)> fun) const;

	template<typename TR, typename ...Args>
	FilterRange<TR> SelectByMaterial(TR const& range, Args &&...args) const;

	template<typename TR, typename T1, typename T2>
	FilterRange<TR> SelectInterface(TR const& range, T1 in, T2 out) const;

	template<typename TR>
	FilterRange<TR> SelectByRectangle(TR const& range, coordinates_type v0,
			coordinates_type v1) const;

	template<typename TR>
	FilterRange<TR> SelectByPolylines(TR const& range,
			PointInPolygon checkPointsInPolygen) const;

	template<typename TR>
	FilterRange<TR> SelectByPoints(TR const& range,
			std::vector<coordinates_type>const & points) const;
	template<typename TR>
	FilterRange<TR> SelectByNGP(TR const& range,
			coordinates_type const & points) const;

	template<typename TR, typename T1, typename T2>
	FilterRange<TR> SelectOnSurface(TR const& range, T1 in, T2 out) const;
	template<typename TR, typename T1, typename T2>
	FilterRange<TR> SelectCrossSurface(TR const& range, T1 in, T2 out) const;

}
;
template<typename TM>
std::ostream & operator<<(std::ostream & os, Model<TM> const & model)
{
	return model.print(os);
}

template<typename TM>
typename Model<TM>::material_type Model<TM>::get(compact_index_type s) const
{

	material_type res = null_material;

	if (this->manifold_type::IForm(s) == VERTEX)
	{
		auto it = material_.find(s);
		if (it != material_.end())
		{
			res = it->second;
		}
	}
	else
	{
		compact_index_type neighbours[manifold_type::MAX_NUM_NEIGHBOUR_ELEMENT];

		int num = this->manifold_type::get_vertices(s, neighbours);

		for (int i = 0; i < num; ++i)
		{
			res |= this->get(neighbours[i]);
		}
	}
	return std::move(res);
}

template<typename TM>
template<typename TR, typename TDict>
FilterRange<TR> Model<TM>::select_by_config(TR const& range,
		TDict const& dict) const
{
	if (!dict)
	{
		pred_fun_type pred = [=]( compact_index_type const & s )->bool
		{
			return true;
		};

		return std::move(FilterRange<TR>(range, std::move(pred)));
	}
	else if (dict.is_function())
	{
		pred_fun_type pred =
				[=]( compact_index_type const & s )->bool
				{
					return (dict( this->manifold_type::get_coordinates( s)).template as<bool>());
				};

		return std::move(FilterRange<TR>(range, std::move(pred)));
	}
	else if (dict["Material"])
	{
		return std::move(
				SelectByMaterial(range,
						dict["Material"].template as<std::string>()));
	}
	else if (dict["Type"])
	{
		auto type = dict["Type"].template as<std::string>("");

		if (type == "NGP")
		{
			return std::move(
					SelectByNGP(range,
							dict["Points"].template as<coordinates_type>()));

		}
		else if (type == "Boundary")
		{
			return std::move(
					SelectInterface(range,
							dict["In"].template as<std::string>("NONE"),
							"NONE"));

		}
		else if (type == "Interface")
		{
			return std::move(
					SelectInterface(range,
							dict["In"].template as<std::string>("NONE"),
							dict["Out"].template as<std::string>("NONE")));
		}
		else if (type == "Range" && dict["Points"].is_table())
		{
			std::vector<coordinates_type> points;

			dict["Points"].as(&points);

			if (points.size() == 2) //select points in a rectangle with diagonal  (x0,y0,z0)~(x1,y1,z1）,
			{
				return std::move(SelectByRectangle(range, points[0], points[1]));
			}
			else if (points.size() > 2) //select points in polylines
			{
				return std::move(
						SelectByPolylines(range,
								PointInPolygon(points,
										dict["Z-Axis"].template as<int>(2))));
			}
			else
			{
				PARSER_ERROR(
						"Number of points  [" + ToString(points.size()) + "]<2");
			}

		}

	}

	else
	{
		PARSER_ERROR("Unknown 'Select' options");
	}
	return FilterRange<TR>();
}
template<typename TM> template<typename TR>
FilterRange<TR> Model<TM>::SelectByPoints(TR const& range,
		std::vector<coordinates_type>const & points) const
{
	if (points.size() == 1)
	{
		return std::move(SelectByNGP(range, points[0]));
	}
	else if (points.size() == 2)
	{
		return std::move(SelectByRectangle(range, points[0], points[1]));
	}
	else
	{
		PointInPolygon poly(points);
		return std::move(SelectByPolylines(range, poly));
	}
}

template<typename TM> template<typename TR>
FilterRange<TR> Model<TM>::SelectByNGP(TR const& range,
		coordinates_type const & x) const
{
	compact_index_type dest;

	std::tie(dest, std::ignore) = manifold_type::coordinates_global_to_local(x);

	if (manifold_type::in_local_range(dest))
	{

		pred_fun_type pred =
				[dest]( compact_index_type const & s )->bool
				{
					return manifold_type::get_cell_index(s)==manifold_type::get_cell_index(dest);
				};

		return std::move(FilterRange<TR>(range, std::move(pred)));
	}
	else
	{
		pred_fun_type pred = []( compact_index_type const & )->bool
		{
			return false;
		};

		return std::move(FilterRange<TR>(range, std::move(pred)));
	}

}

template<typename TM> template<typename TR>
FilterRange<TR> Model<TM>::SelectByFunction(TR const& range,
		std::function<bool(coordinates_type)> fun) const
{
	pred_fun_type pred = [fun,this]( compact_index_type const & s )->bool
	{
		return fun( this->manifold_type::get_coordinates( s));
	};

	return std::move(FilterRange<TR>(range, std::move(pred)));
}

template<typename TM>
template<typename TR, typename T1, typename T2>
FilterRange<TR> Model<TM>::SelectInterface(TR const& range, T1 pin,
		T2 pout) const
{
	/** \note
	 * Good
	 *  +----------#----------+
	 *  |          #          |
	 *  |    A     #-> B   C  |
	 *  |          #          |
	 *  +----------#----------+
	 *
	 *  +--------------------+
	 *  |         ^          |
	 *  |       B |     C    |
	 *  |     ########       |
	 *  |     #      #       |
	 *  |     #  A   #       |
	 *  |     #      #       |
	 *  |     ########       |
	 *  +--------------------+
	 *
	 *             +----------+
	 *             |      C   |
	 *  +----------######     |
	 *  |          | A  #     |
	 *  |    A     | &  #  B  |
	 *  |          | B  #->   |
	 *  +----------######     |
	 *             |          |
	 *             +----------+
	 *
	 *     	       +----------+
	 *       C     |          |
	 *  +----------#----+     |
	 *  |          # A  |     |
	 *  |    B   <-# &  |  A  |
	 *  |          # B  |     |
	 *  +----------#----+     |
	 *             |          |
	 *             +----------+
	 *
	 *
	 * 	 Bad
	 *
	 *  +--------------------+
	 *  |                    |
	 *  |        A           |
	 *  |     ########       |
	 *  |     #      #       |
	 *  |     #->B C #       |
	 *  |     #      #       |
	 *  |     ########       |
	 *  +--------------------+
	 *
	 * 	            +----------+
	 *              |          |
	 *   +-------+  |          |
	 *   |       |  |          |
	 *   |   B   |  |    A     |
	 *   |       |  |          |
	 *   +-------+  |          |
	 *              |          |
	 *              +----------+
	 */
	material_type in = get_material(pin);
	material_type out = get_material(pout);

	if (in == out)
		out = null_material;

	pred_fun_type pred =

			[=]( compact_index_type const & s )->bool
			{

				material_type res;

				auto iform = this->manifold_type::IForm(s);

				auto self=this->get(s);

				if (( self & in).none() && ( (self & out).any() || (out == null_material) ))
				{
					compact_index_type neighbours[manifold_type::MAX_NUM_NEIGHBOUR_ELEMENT];

					int num=0;
					switch(iform)
					{	case VERTEX:
						num= this->manifold_type::get_adjacent_cells(std::integral_constant<size_t ,VERTEX>(), std::integral_constant<size_t ,VOLUME>(), s, neighbours);
						break;
						case EDGE:
						num= this->manifold_type::get_adjacent_cells(std::integral_constant<size_t ,EDGE>(), std::integral_constant<size_t ,VOLUME>(), s, neighbours);
						break;
						case FACE:
						num= this->manifold_type::get_adjacent_cells(std::integral_constant<size_t ,FACE>(), std::integral_constant<size_t ,VOLUME>(), s, neighbours);
						break;
						case VOLUME:
						num= this->manifold_type::get_adjacent_cells(std::integral_constant<size_t ,VOLUME>(), std::integral_constant<size_t ,VOLUME>(), s, neighbours);
						break;
					}

					for (int i = 0; i < num; ++i)
					{
						res |=this->get(neighbours[i]);
					}
				}

				return (res & in).any();
			};

	return std::move(FilterRange<TR>(range, std::move(pred)));

}

template<typename TM> template<typename TR, typename ...Args>
FilterRange<TR> Model<TM>::SelectByMaterial(TR const& range,
		Args && ... args) const
{
	auto material = get_material(std::forward<Args>(args)...);

	if (material != null_material)
	{
		pred_fun_type pred = [=]( compact_index_type const & s )->bool
		{
			return (this->get(s) & material).any();
		};
		return std::move(FilterRange<TR>(range, std::move(pred)));
	}

	else
	{
		pred_fun_type pred = [=]( compact_index_type const & s )->bool
		{
			return (this->get(s) == null_material);
		};
		return std::move(FilterRange<TR>(range, std::move(pred)));
	}

}

template<typename TM> template<typename TR>
FilterRange<TR> Model<TM>::SelectByRectangle(TR const& range,
		coordinates_type v0, coordinates_type v1) const
{
	pred_fun_type pred =
			[v0,v1,this]( compact_index_type const & s )->bool
			{

				auto x = this->manifold_type::get_coordinates(s);
				return ((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0) && (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
						&& (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0));
			};
	return std::move(FilterRange<TR>(range, std::move(pred)));
}

template<typename TM> template<typename TR>
FilterRange<TR> Model<TM>::SelectByPolylines(TR const& range,
		PointInPolygon checkPointsInPolygen) const
{

	return FilterRange<TR>(range,
			[=](compact_index_type s )->bool
			{	return (checkPointsInPolygen(this->manifold_type::get_coordinates(s) ));});
}

}
// namespace simpla

#endif /* MODEL_H_ */
