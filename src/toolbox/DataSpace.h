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

#include "nTuple.h"
#include "../sp_def.h"
#include "Properties.h"
#include "Log.h"
#include "../base/Object.h"

namespace simpla { namespace toolbox
{


/**
 * @ingroup data_interface
 * @brief  Define the size and  shape of m_data set in memory/file
 *  Ref. http://www.hdfgroup.org/HDF5/doc/UG/UG_frame12Dataspaces.html
 */
class DataSpace
{
public:

    typedef nTuple<size_type, MAX_NDIMS_OF_ARRAY> index_tuple;

    typedef std::tuple<
            int // m_ndims_
            , index_tuple // dimensions
            , index_tuple // start
            , index_tuple // stride
            , index_tuple // count
            , index_tuple // block
    > data_shape_s;

    // Creates a null data_space
    DataSpace();

    DataSpace(int rank, size_type const *dims);


    // Copy constructor: makes a copy of the original data_space object.
    DataSpace(const DataSpace &other);

    DataSpace(DataSpace &&other);

    // Destructor: properly terminates access to this data_space.
    ~DataSpace();

    void swap(DataSpace &);

    // Assignment operator
    DataSpace &operator=(const DataSpace &rhs)
    {
        DataSpace(rhs).swap(*this);
        return *this;
    }

    bool is_scalar() const { return std::get<0>(shape()) == 0; }

    std::ostream &print(std::ostream &os, int indent) const;

    static DataSpace create_simple(int rank, const size_type *dims = nullptr);

    static std::tuple<DataSpace, DataSpace> create_simple_unordered(size_type size);

//    static std::tuple<DataSpace, DataSpace> create(size_type rank,
//                                                   index_type const *topology_dims = nullptr,
//                                                   index_type const *start = nullptr,
//                                                   index_type const *_stride = nullptr,
//                                                   index_type const *count = nullptr,
//                                                   index_type const *_block = nullptr
//    );


    // TODO complete support H5Sselect_hyperslab:H5S_seloper_t
    DataSpace &select_hyperslab(const size_type *start,
                                size_type const *_stride,
                                size_type const *count,
                                size_type const *_block);

    void clear_selected();

    bool is_valid() const;

    bool is_simple() const;

    bool is_full() const;

    std::vector<size_type> const &selected_points() const;

    std::vector<size_type> &selected_points();

    void select_point(const size_type *idx);

    void select_point(size_type pos);

    void select_points(size_type num, const size_type *b);


    /**
     * @return <ndims,dimensions,start,count,stride,block>
     */
    data_shape_s const &shape() const;

    data_shape_s &shape();

    size_type size() const;

    size_type num_of_elements() const;

private:
    struct pimpl_s;
    std::shared_ptr<pimpl_s> m_pimpl_;

};


}} //namespace simpla { namespace data_model


#endif /* CORE_DATASET_DATASPACE_H_ */
