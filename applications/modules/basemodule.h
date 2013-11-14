/*
 * BaseModule.h
 *
 *  Created on: 2012-11-2
 *      Author: salmon
 */

#ifndef BASEMODULE_H_
#define BASEMODULE_H_

#include "fetl/fetl.h"
#include "engine/basecontext.h"
#include "engine/context.h"
#include "utilities/properties.h"

namespace simpla
{

class BaseModule
{

public:
	typedef BaseModule ThisType;

	typedef TR1::shared_ptr<BaseModule> Holder;

	BaseModule(BaseContext & d, ptree const &pt)
	{
		BOOST_FOREACH(const typename ptree::value_type &v, pt)
		{
			if (v.first == "DataSet")
			{
				std::string o_name = v.second.get_value<std::string>();

				boost::optional<TR1::shared_ptr<Object> > obj = d.FindObject(
						o_name);
				if (!!obj)
				{
					dataset_[v.second.get<std::string>("<xmlattr>.Name")] =
							*obj;
				}
				else
				{
					ERROR << "Undefined object [" << o_name << "]";
				}
			}

		}
		LOG << "Create module Maxwell";
	}

	virtual ~BaseModule()
	{
	}

	virtual void Eval()=0;
protected:
	std::map<std::string, TR1::shared_ptr<Object> > dataset_;
};

}  // namespace simpla

#endif /* BASEMODULE_H_ */
