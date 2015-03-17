/**
 * @file model.h
 *
 *  created on: 2013-12-15
 *      Author: salmon
 */

#ifndef MODEL_H_
#define MODEL_H_

#include <stddef.h>
#include <algorithm>
#include <bitset>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../numeric/pointinpolygon.h"
#include "../utilities/utilities.h"

namespace simpla
{

/**
 *  @defgroup  Model Model
 *  @brief Geometry modeling
 */

/**
 *  @ingroup Model
 *   @brief Model
 */

class Model
{

public:
	static constexpr size_t MAX_NUM_OF_MEIDA_TYPE = std::numeric_limits<
			unsigned long>::digits;

	static constexpr size_t ndims = 3;

	typedef std::bitset<MAX_NUM_OF_MEIDA_TYPE> material_type;

	typedef nTuple<Real, ndims> coordinates_type;

	typedef size_t id_type;

	const material_type null_material;

	std::map<id_type, material_type> m_data_;

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
			null_material(0), max_material_(CUSTOM + 1)
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
		return m_data_.empty();
	}

	material_type register_material(std::string const & name)
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

	material_type get(id_type s) const
	{
		auto it = m_data_.find(s);

		if (it != m_data_.end())
		{
			return *it;
		}
		else
		{
			return null_material;
		}

	}

	material_type operator[](id_type s) const
	{
		return get(s);
	}

	void clear()
	{
		m_data_.clear();
	}

	template<typename TR>
	void erase(TR const & r)
	{
		for (auto s : r)
		{
			m_data_.erase(s);
		}
	}

	template<typename TR, typename M>
	void set(TR const & r, M const& material)
	{
		auto tag = get_material(material);
		for (auto s : r)
		{
			m_data_[s] |= tag;
		}
	}
	template<typename TR, typename M>
	void unset(TR const & r, M const& material)
	{
		auto tag = get_material(material);
		for (auto s : r)
		{
			m_data_[s] &= ~tag;
		}
	}

	template<typename TR, typename TDict>
	std::set<id_type> select_by_config(TR const& range,
			TDict const& dict) const;

}
;

//typename Model::material_type Model::get(id_type s) const
//{
//
//	material_type res = null_material;
//
//	if (this->geometry_type::IForm(s) == VERTEX)
//	{
//		auto it = m_data_.find(s);
//		if (it != m_data_.end())
//		{
//			res = it->second;
//		}
//	}
//	else
//	{
//		id_type neighbours[geometry_type::MAX_NUM_NEIGHBOUR_ELEMENT];
//
//		int num = this->geometry_type::get_vertices(s, neighbours);
//
//		for (int i = 0; i < num; ++i)
//		{
//			res |= this->get(neighbours[i]);
//		}
//	}
//	return std::move(res);
//}

//template<typename TM>
//template<typename TR, typename TDict>
//FilterRange<TR> Model<TM>::select_by_config(TR const& range,
//		TDict const& dict) const
//{
//	if (!dict)
//	{
//		pred_fun_type pred = [=]( id_type const & s )->bool
//		{
//			return true;
//		};
//
//		return std::move(FilterRange < TR > (range, std::move(pred)));
//	}
//	else if (dict.is_function())
//	{
//		pred_fun_type pred =
//				[=]( id_type const & s )->bool
//				{
//					return (dict( this->geometry_type::get_coordinates( s)).template as<bool>());
//				};
//
//		return std::move(FilterRange < TR > (range, std::move(pred)));
//	}
//	else if (dict["Material"])
//	{
//		return std::move(
//				SelectByMaterial(range,
//						dict["Material"].template as<std::string>()));
//	}
//	else if (dict["Type"])
//	{
//		auto type = dict["Type"].template as < std::string > ("");
//
//		if (type == "NGP")
//		{
//			return std::move(
//					SelectByNGP(range,
//							dict["Points"].template as<coordinates_type>()));
//
//		}
//		else if (type == "Boundary")
//		{
//			return std::move(
//					SelectInterface(range,
//							dict["In"].template as < std::string > ("NONE"),
//							"NONE"));
//
//		}
//		else if (type == "Interface")
//		{
//			return std::move(
//					SelectInterface(range,
//							dict["In"].template as < std::string > ("NONE"),
//							dict["Out"].template as < std::string > ("NONE")));
//		}
//		else if (type == "Range" && dict["Points"].is_table())
//		{
//			std::vector < coordinates_type > points;
//
//			dict["Points"].as(&points);
//
//			if (points.size() == 2) //select points in a rectangle with diagonal  (x0,y0,z0)~(x1,y1,z1）,
//			{
//				return std::move(SelectByRectangle(range, points[0], points[1]));
//			}
//			else if (points.size() > 2) //select points in polylines
//			{
//				return std::move(
//						SelectByPolylines(range,
//								PointInPolygon(points,
//										dict["Z-Axis"].template as<int>(2))));
//			}
//			else
//			{
//				PARSER_ERROR(
//						"Number of points  [" + value_to_string(points.size())
//								+ "]<2");
//			}
//
//		}
//
//	}
//
//	else
//	{
//		PARSER_ERROR("Unknown 'Select' options");
//	}
//	return FilterRange < TR > (range, [=](id_type const &)
//	{	return true;});
//}
//
//template<typename TM>
//template<typename TR, typename T1, typename T2>
//FilterRange<TR> Model<TM>::SelectInterface(TR const& range, T1 pin,
//		T2 pout) const
//{
//	/** \note
//	 * Good
//	 *  +----------#----------+
//	 *  |          #          |
//	 *  |    A     #-> B   C  |
//	 *  |          #          |
//	 *  +----------#----------+
//	 *
//	 *  +--------------------+
//	 *  |         ^          |
//	 *  |       B |     C    |
//	 *  |     ########       |
//	 *  |     #      #       |
//	 *  |     #  A   #       |
//	 *  |     #      #       |
//	 *  |     ########       |
//	 *  +--------------------+
//	 *
//	 *             +----------+
//	 *             |      C   |
//	 *  +----------######     |
//	 *  |          | A  #     |
//	 *  |    A     | &  #  B  |
//	 *  |          | B  #->   |
//	 *  +----------######     |
//	 *             |          |
//	 *             +----------+
//	 *
//	 *     	       +----------+
//	 *       C     |          |
//	 *  +----------#----+     |
//	 *  |          # A  |     |
//	 *  |    B   <-# &  |  A  |
//	 *  |          # B  |     |
//	 *  +----------#----+     |
//	 *             |          |
//	 *             +----------+
//	 *
//	 *
//	 * 	 Bad
//	 *
//	 *  +--------------------+
//	 *  |                    |
//	 *  |        A           |
//	 *  |     ########       |
//	 *  |     #      #       |
//	 *  |     #->B C #       |
//	 *  |     #      #       |
//	 *  |     ########       |
//	 *  +--------------------+
//	 *
//	 * 	            +----------+
//	 *              |          |
//	 *   +-------+  |          |
//	 *   |       |  |          |
//	 *   |   B   |  |    A     |
//	 *   |       |  |          |
//	 *   +-------+  |          |
//	 *              |          |
//	 *              +----------+
//	 */
//	material_type in = get_material(pin);
//	material_type out = get_material(pout);
//
//	if (in == out)
//		out = null_material;
//
//	pred_fun_type pred =
//
//			[=]( id_type const & s )->bool
//			{
//
//				material_type res;
//
//				auto iform = this->geometry_type::IForm(s);
//
//				auto self=this->get(s);
//
//				if (( self & in).none() && ( (self & out).any() || (out == null_material) ))
//				{
//					id_type neighbours[geometry_type::MAX_NUM_NEIGHBOUR_ELEMENT];
//
//					int num=0;
//					switch(iform)
//					{	case VERTEX:
//						num= this->geometry_type::get_adjacent_cells(std::integral_constant<size_t ,VERTEX>(), std::integral_constant<size_t ,VOLUME>(), s, neighbours);
//						break;
//						case EDGE:
//						num= this->geometry_type::get_adjacent_cells(std::integral_constant<size_t ,EDGE>(), std::integral_constant<size_t ,VOLUME>(), s, neighbours);
//						break;
//						case FACE:
//						num= this->geometry_type::get_adjacent_cells(std::integral_constant<size_t ,FACE>(), std::integral_constant<size_t ,VOLUME>(), s, neighbours);
//						break;
//						case VOLUME:
//						num= this->geometry_type::get_adjacent_cells(std::integral_constant<size_t ,VOLUME>(), std::integral_constant<size_t ,VOLUME>(), s, neighbours);
//						break;
//					}
//
//					for (int i = 0; i < num; ++i)
//					{
//						res |=this->get(neighbours[i]);
//					}
//				}
//
//				return (res & in).any();
//			};
//
//	return std::move(FilterRange < TR > (range, (pred)));
//
//}
//
//template<typename TM> template<typename TR, typename ...Args>
//FilterRange<TR> Model<TM>::SelectByMaterial(TR const& range,
//		Args && ... args) const
//{
//	auto material = get_material(std::forward<Args>(args)...);
//
//	if (material != null_material)
//	{
//		pred_fun_type pred = [=]( id_type const & s )->bool
//		{
//			return (this->get(s) & material).any();
//		};
//		return std::move(FilterRange < TR > (range, std::move(pred)));
//	}
//
//	else
//	{
//		pred_fun_type pred = [=]( id_type const & s )->bool
//		{
//			return (this->get(s) == null_material);
//		};
//		return std::move(FilterRange < TR > (range, std::move(pred)));
//	}
//
//}

}
// namespace simpla

#endif /* MODEL_H_ */
