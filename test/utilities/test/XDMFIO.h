/*
 * XDMFIO.h
 *
 *  created on: 2013-12-10
 *      Author: salmon
 */

#ifndef XDMF_IO_H_
#define XDMF_IO_H_

#include <Xdmf.h>
#include "../DataType.h"

namespace simpla
{
namespace io
{
XdmfInt32 convert_datatype_sp_to_xdmf(DataType const &d_type);

/** @ingroup XDMF
 * @{
 */
template<typename T>
struct XdmfTypeTraits
{
    static const int value = XDMF_UNKNOWN_TYPE;
};

#define DEF_XDMF_TYPE_TRAITS(_T_, _V_) \
template<> struct XdmfTypeTraits<_T_>{ static const  int  value = _V_;};

DEF_XDMF_TYPE_TRAITS(char, XDMF_INT8_TYPE);
DEF_XDMF_TYPE_TRAITS(short, XDMF_INT16_TYPE);
DEF_XDMF_TYPE_TRAITS(int, XDMF_INT32_TYPE);
DEF_XDMF_TYPE_TRAITS(long
                             long, XDMF_INT64_TYPE);
DEF_XDMF_TYPE_TRAITS(float, XDMF_FLOAT32_TYPE);
DEF_XDMF_TYPE_TRAITS(double, XDMF_FLOAT64_TYPE);
DEF_XDMF_TYPE_TRAITS(unsigned  char, XDMF_UINT8_TYPE);
DEF_XDMF_TYPE_TRAITS(unsigned   short, XDMF_UINT16_TYPE);
DEF_XDMF_TYPE_TRAITS(unsigned   int, XDMF_UINT32_TYPE);

#undef DEF_XDMF_TYPE_TRAITS

void InsertDataItem(XdmfDataItem *dataitem, int rank, XdmfInt64 const *pdims, DataType const &dtype,
                    void *data, std::string const &HeavyDataSetName = "");

template<typename T> inline void
InsertDataItem(XdmfDataItem *dataitem, int rank, XdmfInt64 const *dims,
               T *data, std::string const &HeavyDataSetName = "")
{

    InsertDataItem(dataitem, rank, dims, traits::datatype<T>::create(), data, HeavyDataSetName);

}

template<typename T> inline void
InsertDataItem(XdmfDataItem *dataitem, size_t num, T *data,
               std::string const &HeavyDataSetName = "")
{
    XdmfInt64 s = static_cast<XdmfInt64>(num);
    InsertDataItem(dataitem, 1, &s, traits::datatype<T>::create(), data, HeavyDataSetName);
}

template<typename TI, typename TFun>
inline void
InsertDataItemWithFun(XdmfDataItem *dataitem, unsigned int rank,
                      TI *pdims, TFun const &fun,
                      std::string const &HeavyDataSetName)
{

    XdmfInt64 dims[rank];
    std::copy(pdims, pdims + rank, dims);
    dataitem->SetShape(rank, dims);
//	dataitem->SetFormat(XDMF_FORMAT_HDF);
//	dataitem->SetArrayIsMine(true);
    dataitem->SetHeavyDataSetName(HeavyDataSetName.c_str());

    XdmfArray *myArray = dataitem->GetArray(1);
//	myArray->SetAllowAllocate(true);
    myArray->SetNumberType(XdmfTypeTraits<decltype(fun(pdims))>::value);
    myArray->SetShape(rank, dims);

    TI idx[rank];
    std::fill(idx, idx + rank, 0);

    XdmfInt64 s = 0;
    while (idx[0] < static_cast<XdmfInt64>(pdims[0]))
    {

        myArray->setValue(s, fun(idx));
        ++s;

        ++(idx[rank - 1]);

        for (int j = rank - 1; j > 0; --j)
        {
            if (idx[j] >= pdims[j])
            {
                idx[j] = 0;
                ++idx[j - 1];
            }
        }
    }

}

/**
 * @}
 */

}
}//namespace simpla { namespace io


#endif /* XDMF_IO_H_ */
