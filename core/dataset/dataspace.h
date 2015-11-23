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

    typedef size_t index_type;
    typedef nTuple <index_type, MAX_NDIMS_OF_ARRAY> index_tuple;

public:

    typedef std::tuple<
            int // ndims
            , index_tuple // dimensions
            , index_tuple // start
            , index_tuple // stride
            , index_tuple // count
            , index_tuple // block
    > data_shape_s;


    // Creates a null dataspace
    DataSpace();


    DataSpace(int rank, size_t const *dims);

    // Copy constructor: makes a copy of the original DataSpace object.
    DataSpace(const DataSpace &other);


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


    DataSpace &select_hyperslab(index_type const *start,
                                index_type const *_stride,
                                index_type const *count,
                                index_type const *_block);

    bool is_valid() const;


    bool is_simple() const;

    /**
     * @return <ndims,dimensions,start,count,stride,block>
     */
    data_shape_s const &shape() const;

    size_t size() const;

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
