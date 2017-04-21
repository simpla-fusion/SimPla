//
// Created by salmon on 17-3-8.
//
#include "DataArray.h"
#include <iomanip>
#include "DataTraits.h"
namespace simpla {
namespace data {

//DataArray::DataArray() {}
//DataArray::~DataArray() {}
std::ostream& DataArray::Serialize(std::ostream& os, int indent) const {
    size_type ie = size();
    os << "[";
    Get(0)->Serialize(os, indent + 1);
    for (size_type i = 1; i < ie; ++i) {
        os << " , ";
        Get(i)->Serialize(os, indent + 1);
        //        if (i % 5 == 0) { os << std::endl; }
    }
    os << "]";
    return os;
};

}  // namespace data {
}  // namespace simpla {