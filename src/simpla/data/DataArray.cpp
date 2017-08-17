//
// Created by salmon on 17-3-8.
//
#include "DataArray.h"
#include <iomanip>
#include "DataTraits.h"
namespace simpla {
namespace data {

int DataArray::Foreach(std::function<int(std::shared_ptr<DataEntity>)> const& fun) const {
    int res = 0;
    for (size_type i = 0, ie = Count(); i < ie; ++i) { res += fun(Get(i)); }
    return res;
};

}  // namespace data {
}  // namespace simpla {