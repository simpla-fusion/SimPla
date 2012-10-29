/*
 * context_impl.h
 *
 *  Created on: 2012-10-17
 *      Author: salmon
 */

#ifndef CONTEXT_IMPL_H_
#define CONTEXT_IMPL_H_
#include "context.h"
#include "fetl/fetl.h"
#include "modules/em/maxwell.h"
#include "modules/em/pml.h"
#include "modules/em/rf_src.h"
#include "modules/fluid/cold_fluid.h"
#include "io/write_xdmf.h"
#include "io/read_hdf5.h"
#include <boost/foreach.hpp>
namespace simpla
{

template<typename TG>
inline std::string Context<TG>::Summary() const
{
	std::ostringstream os;

	os

	<< PHYS_CONSTANTS.Summary()

	<< SINGLELINE << std::endl

	<< std::setw(20) << "dt : " << dt << std::endl

	<< grid.Summary() << std::endl

	<< DOUBLELINE << std::endl

	;

//	<< "Objects List" << std::endl
//
//	<< SINGLELINE << std::endl
//
//
//	for (typename std::map<std::string, TR1::shared_ptr<Object> >::const_iterator it =
//			objects.begin(); it != objects.end(); ++it)
//	{
//		os << it->first << std::endl;
//	}

	return os.str();

}

template<typename TG>
void Context<TG>::RegisterFactory()
{
//#define REG_OBJ_FACTORY(_NAME_,_TYPE_)  \
//    objFactory[_NAME_] =  &BaseContext::CreateObject<_TYPE_> ;			\
//	objFactory[std::string("R")+_NAME_] =  &BaseContext::CreateObject<R##_TYPE_> ;		\
//	objFactory[std::string("C")+_NAME_ ] =  &BaseContext::CreateObject<C##_TYPE_> ;		\
//
//	REG_OBJ_FACTORY("ZeroForm", ZeroForm);
//	REG_OBJ_FACTORY("OneForm", OneForm);
//	REG_OBJ_FACTORY("TwoForm", TwoForm);
//
//	REG_OBJ_FACTORY("VecZeroForm", VecZeroForm);
//	REG_OBJ_FACTORY("VecOneForm", VecOneForm);
//	REG_OBJ_FACTORY("VecTwoForm", VecTwoForm);
//
//#undef REG_OBJ_FACTORY

}

template<typename TG>
Context<TG>::Context(ptree const & pt) :
		BaseContext(pt), grid(pt.get_child("Grid"))
{
	LOG << "Register Fields";

	RegisterFactory();

	BOOST_FOREACH(const typename ptree::value_type &v, pt.get_child("Fields"))
	{
		if (v.first != "Field")
		{
			continue;
		}
		try
		{
			std::string type = v.second.template get<std::string>(
					"<xmlattr>.type");
			std::string id = v.second.template get<std::string>("<xmlattr>.id");

			if (type == "ZeroForm")
			{
				*GetObject<ZeroForm>(id) = v.second.get("value", 0.0d);

			}
			else if (type == "OneForm")
			{
				nTuple<THREE, Real> dv =
				{ 0, 0, 0 };
				*GetObject<OneForm>(id) = v.second.get("value", dv,
						pt_trans<nTuple<THREE, Real>, std::string>());

			}
			else if (type == "TwoForm")
			{
				nTuple<THREE, Real> dv =
				{ 0, 0, 0 };
				*GetObject<TwoForm>(id) = v.second.get("value", dv,
						pt_trans<nTuple<THREE, Real>, std::string>());
			}
			else if (type == "ThreeForm")
			{
				*GetObject<ThreeForm>(id) = v.second.get("value", 0.0d);
			}
			else if (type == "VecZeroForm")
			{
				nTuple<THREE, Real> dv =
				{ 0, 0, 0 };
				*GetObject<VecZeroForm>(id) = v.second.get("value", dv,
						pt_trans<nTuple<THREE, Real>, std::string>());
			}
			else if (type == "VecOneForm")
			{
				nTuple<THREE, nTuple<THREE, Real> > dv =
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };

				*GetObject<VecOneForm>(id) = v.second.get("value", dv,
						pt_trans<nTuple<THREE, nTuple<THREE, Real> >,
								std::string>());

			}
			else if (type == "TwoForm")
			{
				nTuple<THREE, nTuple<THREE, Real> > dv =
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };

				*GetObject<VecTwoForm>(id) = v.second.get("value", dv,
						pt_trans<nTuple<THREE, nTuple<THREE, Real> >,
								std::string>());

			}
			else if (type == "VecThreeForm")
			{
				nTuple<THREE, Real> dv =
				{ 0, 0, 0 };
				*GetObject<VecThreeForm>(id) = v.second.get("value", dv,
						pt_trans<nTuple<THREE, Real>, std::string>());
			}
			else if (type == "CZeroForm")
			{
				*GetObject<CZeroForm>(id) = v.second.get("value", 0.0d);
			}
			else if (type == "COneForm")
			{
				nTuple<THREE, Real> dv =
				{ 0, 0, 0 };
				*GetObject<COneForm>(id) = v.second.get("value", dv,
						pt_trans<nTuple<THREE, Real>, std::string>());

			}
			else if (type == "CTwoForm")
			{
				nTuple<THREE, Real> dv =
				{ 0, 0, 0 };
				*GetObject<CTwoForm>(id) = v.second.get("value", dv,
						pt_trans<nTuple<THREE, Real>, std::string>());
			}
			else if (type == "CThreeForm")
			{
				*GetObject<CThreeForm>(id) = v.second.get("value", 0.0d);
			}
			else if (type == "CVecZeroForm")
			{
				nTuple<THREE, Real> dv =
				{ 0, 0, 0 };
				*GetObject<CVecZeroForm>(id) = v.second.get("value", dv,
						pt_trans<nTuple<THREE, Real>, std::string>());
			}
			else if (type == "CVecOneForm")
			{
				nTuple<THREE, nTuple<THREE, Real> > dv =
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };

				*GetObject<CVecOneForm>(id) = v.second.get("value", dv,
						pt_trans<nTuple<THREE, nTuple<THREE, Real> >,
								std::string>());

			}
			else if (type == "CTwoForm")
			{
				nTuple<THREE, nTuple<THREE, Real> > dv =
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };

				*GetObject<CVecTwoForm>(id) = v.second.get("value", dv,
						pt_trans<nTuple<THREE, nTuple<THREE, Real> >,
								std::string>());

			}
			else if (type == "CVecThreeForm")
			{
				nTuple<THREE, Real> dv =
				{ 0, 0, 0 };
				*GetObject<CVecThreeForm>(id) = v.second.get("value", dv,
						pt_trans<nTuple<THREE, Real>, std::string>());
			}

			boost::optional<std::string> url =
					v.second.get_optional<std::string>("url");
			if (!!url)
			{
				boost::optional<TR1::shared_ptr<Object> > obj = FindObject(id);

				if (!!obj)
				{
					io::ReadData(*url, *obj);
				}
			}

		} catch (...)
		{
			WARNING << "Register field error!";
		}
	}

	BOOST_FOREACH(const typename ptree::value_type &v, pt.get_child("Modules"))
	{
		if (v.first != "Module")
		{
			continue;
		}
		boost::optional<std::string> type = v.second.template get_optional<
				std::string>("<xmlattr>.type");
		if (!type)
		{
			continue;
		}
		else if (*type == "RFSrc")
		{
			modules.push_back(
					TR1::bind(&em::RFSrc<TG>::Eval,
							new em::template RFSrc<TG>(*this, v.second)));
		}
		else if (*type == "Maxwell")
		{
			modules.push_back(
					TR1::bind(&em::Maxwell<TG>::Eval,
							new em::template Maxwell<TG>(*this, v.second)));
		}
		else if (*type == "PML")
		{
			modules.push_back(
					TR1::bind(&em::PML<TG>::Eval,
							new em::template PML<TG>(*this, v.second)));
		}
		else if (*type == "ColdFluid")
		{
			modules.push_back(
					TR1::bind(&em::ColdFluid<TG>::Eval,
							new em::template ColdFluid<TG>(*this, v.second)));
		}

	}

	std::string type = pt.template get("OutPut.<xmlattr>.type", "XDMF");

	if (type == "XDMF")
	{
		modules.push_back(
				TR1::bind(&io::WriteXDMF<TG>::Eval,
						new io::template WriteXDMF<TG>(*this,
								pt.get_child("OutPut"))));

	}

}
template<typename TG>
Context<TG>::~Context()
{
}
}  // namespace simpla

#endif /* CONTEXT_IMPL_H_ */
