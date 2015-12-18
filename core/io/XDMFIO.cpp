/**
 * @file xdmf_io.cpp
 * @author salmon
 * @date 2015-12-02.
 */
#include "XDMFIO.h"
#include "../data_model/DataType.h"
#include "../gtl/utilities/log.h"

namespace simpla { namespace io
{


XdmfInt32 convert_datatype_sp_to_xdmf(DataType const &d_type)
{
    typedef XdmfInt32 xdmf_id_t;

    xdmf_id_t res = XDMF_UNKNOWN_TYPE;

    if (!d_type.is_valid()) THROW_EXCEPTION_RUNTIME_ERROR("illegal data type");

    if (!d_type.is_compound())
    {

        if (d_type.template is_same<int>())
        {
            res = XDMF_INT32_TYPE;
        }
        else if (d_type.template is_same<long>())
        {
            res = XDMF_INT64_TYPE;
        }
        else if (d_type.template is_same<unsigned long>())
        {
            res = H5T_NATIVE_ULONG;
        }
        else if (d_type.template is_same<float>())
        {
            res = XDMF_FLOAT32_TYPE;
        }
        else if (d_type.template is_same<double>())
        {
            res = XDMF_FLOAT64_TYPE;
        }
        else if (d_type.template is_same<std::complex<double>>())
        {
            THROW_EXCEPTION_RUNTIME_ERROR("Unknown data type:" + d_type.name());
        }

        if (d_type.is_array())
        {
            UNIMPLEMENTED;
        }
    }
    else
    {
        UNIMPLEMENTED;
    }

    if (res == XDMF_UNKNOWN_TYPE)
    {
        WARNING << "sp.DataType convert to H5.datatype failed!" << std::endl;
        throw std::bad_cast();
    }
    return (res);
}

void InsertDataItem(XdmfDataItem *dataitem, int rank, XdmfInt64 const *dims, DataType const &dtype,
                    void *data, std::string const &HeavyDataSetName)
{
    dataitem->SetShape(rank, const_cast<XdmfInt64 *>(dims));
    dataitem->SetFormat(XDMF_FORMAT_HDF);
    dataitem->SetArrayIsMine(false);
    dataitem->SetHeavyDataSetName(HeavyDataSetName.c_str());

    XdmfArray *myArray = dataitem->GetArray(1);
    myArray->SetAllowAllocate(false);
    myArray->SetNumberType(convert_datatype_sp_to_xdmf(dtype));
    myArray->SetShape(rank, const_cast<XdmfInt64 *>(dims));
    myArray->SetDataPointer(data);
}

}}//namespace simpla{namespace io{