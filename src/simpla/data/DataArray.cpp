//
// Created by salmon on 17-3-8.
//
#include "DataArray.h"
#include <iomanip>
#include "DataTraits.h"
namespace simpla {
namespace data {

DataArray::DataArray() {}
DataArray::~DataArray() {}
std::ostream& DataArray::Print(std::ostream& os, int indent) const {
    size_type ie = count();

    os << "[";
    Get(0)->Print(os, indent + 1);
    for (size_type i = 1; i < ie; ++i) {
        os << " , ";
        Get(i)->Print(os, indent + 1);
//        if (i % 5 == 0) { os << std::endl; }
    }
    os << "]";
    return os;
};

}  // namespace data {
}  // namespace simpla {