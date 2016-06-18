/*
 * spIO.cpp
 *
 *  Created on: 2016年6月18日
 *      Author: salmon
 */

#include <memory>
#include "../src/io/IO.h"
#include "../src/parallel/DistributedObject.h"

#include "../src/data_model/DataSet.h"
#include "../src/data_model/DataType.h"
#include "../src/data_model/DataSpace.h"
#include "spSimPlaWrap.h"

#include "sp_def.h"
using namespace simpla;

void hdf5_write_field(char const url[], void *d, int ndims, size_type const * dims, ptrdiff_t const*offset,
		size_type const *count, int flag)
{
	simpla::data_model::DataSet dset;
	data_model::DataSpace d_space(ndims, &count[0]);
	data_model::DataSpace m_space(ndims, &dims[0]);
	m_space.select_hyperslab(&offset[0], nullptr, &count[0], nullptr);
	dset.memory_space = m_space;
	dset.data_space = d_space;
	dset.data_type = data_model::DataType::create<Real>();
	dset.data = std::shared_ptr<void>(reinterpret_cast<void *>(d), tags::do_nothing()),

	simpla::io::write(url, dset, flag);
}

struct spDistributedObject_s
{
	simpla::parallel::DistributedObject self;
};
void spCreateDistributeObject(spDistributedObject**obj)
{
	*obj = new spDistributedObject_s;

}
void spCreateDestroyObject(spDistributedObject**obj)
{
	delete *obj;
	*obj = 0x0;
}
void spDestroyObjectNonblockingSync(spDistributedObject* obj)
{
	obj->self.sync();
}
void spDestroyObjectWait(spDistributedObject* obj)
{
	obj->self.wait();
}
