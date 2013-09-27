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

	typedef std::shared_ptr<BaseModule> Holder;

	BaseModule(BaseContext * d, PTree const &pt)
	{
		BOOST_FOREACH(const typename PTree::value_type &v, pt)
		{
			if (v.first == "DataSet")
			{
				std::string o_name = v.second.get_value<std::string>();

				boost::optional<std::shared_ptr<Object> > obj =
						d->objects->Find(o_name);
				if (!!obj)
				{
					std::string p_name = v.second.get("<xmlattr>.Name",
							"default");
					dataset_[p_name] = *obj;
				}
				else
				{
					ERROR << "Undefined object [" << o_name << "]";
				}
			}

		}
	}

	virtual ~BaseModule()
	{
	}

	virtual void Eval()=0;
protected:
	CompoundObject dataset_;
};

}  // namespace simpla

#endif /* BASEMODULE_H_ */
