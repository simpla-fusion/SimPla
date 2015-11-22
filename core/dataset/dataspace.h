/**
 * @file dataspace.h
 *
 *  Created on: 2014-11-10
 *  @author: salmon
 */

#ifndef CORE_DATASET_DATASPACE_H_
#define CORE_DATASET_DATASPACE_H_

#include <cstdbool>
#include <memory>

#include "../gtl/ntuple.h"
#include "../gtl/primitives.h"
#include "../gtl/properties.h"
#include "../gtl/utilities/log.h"

namespace simpla
{

struct DataSet;

/**
 * @ingroup data_interface
 * @brief  Define the size and  shape of data set in memory/file
 *  Ref. http://www.hdfgroup.org/HDF5/doc/UG/UG_frame12Dataspaces.html
 */
class DataSpace
{
public:
    typedef size_t index_type;
    typedef nTuple <index_type, MAX_NDIMS_OF_ARRAY> index_tuple;

    struct data_shape_s
    {
        int ndims = 3;

        index_tuple dimensions;
        index_tuple offset;
        index_tuple count;
        index_tuple stride;
        index_tuple block;
    };


    // Creates a null dataspace
    DataSpace();


    DataSpace(int rank, size_t const *dims);

    // Copy constructor: makes a copy of the original DataSpace object.
    DataSpace(const DataSpace &other);

//	DataSpace(DataSpace&& other);
    // Destructor: properly terminates access to this dataspace.
    ~DataSpace();

    void swap(DataSpace &);

    // Assignment operator
    DataSpace &operator=(const DataSpace &rhs)
    {
        DataSpace(rhs).swap(*this);
        return *this;
    }

    static DataSpace create_simple(int rank, const index_type *dims);

    DataSpace &set_local_shape(index_type const *local_dimensions,
                               index_type const *local_offset);

    DataSpace &select_hyperslab(index_type const *offset,
                                index_type const *stride, index_type const *count,
                                index_type const *block = nullptr);

    bool is_valid() const;

    bool is_distributed() const;

    bool is_simple() const
    {
        /// TODO support  complex selection of data space
        /// @ref http://www.hdfgroup.org/HDF5/doc/UG/UG_frame12Dataspaces.html
        return is_valid() && (!is_distributed());
    }

    /**
     * @return <ndims,dimensions,start,count,stride,block>
     */
    data_shape_s shape() const;

    data_shape_s local_shape() const;

    data_shape_s global_shape() const;

    size_t local_memory_size() const;

    size_t global_size() const;

    size_t size() const { return global_size(); }

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> pimpl_;

};

/**
 * @ingroup data_interface
 * create dataspace
 * @param args
 * @return
 */
template<typename ... Args>
DataSpace make_dataspace(Args &&... args)
{
    return DataSpace(std::forward<Args>(args)...);
}

/**@}  */

}  // namespace simpla

#endif /* CORE_DATASET_DATASPACE_H_ */
