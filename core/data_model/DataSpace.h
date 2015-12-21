/**
 * @file data_space.h
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
#include "../gtl/Properties.h"
#include "../gtl/utilities/log.h"
#include "../base/Object.h"

namespace simpla { namespace data_model
{

struct DataSet;

/**
 * @ingroup data_interface
 * @brief  Define the size and  shape of data set in memory/file
 *  Ref. http://www.hdfgroup.org/HDF5/doc/UG/UG_frame12Dataspaces.html
 */
class DataSpace : public base::Object
{
public:

    SP_OBJECT_HEAD(DataSpace, base::Object);


    typedef size_t index_type;
    typedef nTuple <index_type, MAX_NDIMS_OF_ARRAY> index_tuple;


    typedef std::tuple<
            int // ndims
            , index_tuple // dimensions
            , index_tuple // start
            , index_tuple // stride
            , index_tuple // count
            , index_tuple // block
    > data_shape_s;

    // Creates a null data_space
    DataSpace();

    DataSpace(int rank, size_t const *dims);


    // Copy constructor: makes a copy of the original data_space object.
    DataSpace(const DataSpace &other);


    // Destructor: properly terminates access to this data_space.
    ~DataSpace();

    void swap(DataSpace &);

    // Assignment operator
    DataSpace &operator=(const DataSpace &rhs)
    {
        DataSpace(rhs).swap(*this);
        return *this;
    }

    virtual std::ostream &print(std::ostream &os, int indent) const;


    static DataSpace create_simple(int rank, const index_type *dims);


    // TODO complete support H5Sselect_hyperslab:H5S_seloper_t
    DataSpace &select_hyperslab(index_type const *start,
                                index_type const *_stride,
                                index_type const *count,
                                index_type const *_block);

    bool is_valid() const;


    bool is_simple() const;

    std::vector<size_t> const &selected_points() const;

    std::vector<size_t> &selected_points();

    void select_point(const size_t *idx);

    void select_point(size_t pos);

    void select_points(size_t num, const size_t *b);


    /**
     * @return <ndims,dimensions,start,count,stride,block>
     */
    data_shape_s const &shape() const;

    size_t size() const;

    size_t num_of_elements() const;

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};


}} //namespace simpla { namespace data_model


#endif /* CORE_DATASET_DATASPACE_H_ */
