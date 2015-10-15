/**
 * @file distributed_array.h
 *
 *  created on: 2014-5-30
 *      Author: salmon
 */

#ifndef DISTRIBUTED_ARRAY_H_
#define DISTRIBUTED_ARRAY_H_


#include <memory>
#include "../gtl/dataset/dataset.h"

namespace simpla
{

/* @brief  DistributedArray is used to manage the parallel
 * communication while using the n-dimensional regular array.
 *
 * @note
 *  - DistributedArray is continue in each dimension
 *
 * inspired by :
 *  - DMDA in PETSc http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/index.html
 *
 **/

struct DistributedArray : public DataSet
{

	DistributedArray(DataType const &d_type, DataSpace const &d_space);

	DistributedArray(const DistributedArray &);

	virtual ~DistributedArray();


	void swap(DistributedArray &);

	virtual std::string get_type_as_string() const = 0;

//	virtual bool is_valid() const;

	virtual void deploy();

	virtual void sync();

	virtual void async();

	virtual void wait();

	virtual bool is_ready() const;


private:

	struct pimpl_s;
	std::unique_ptr<pimpl_s> pimpl_;


};
}//namespace simpla

#endif /* DISTRIBUTED_ARRAY_H_ */
