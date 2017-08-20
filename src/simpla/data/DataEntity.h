//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H
#include "simpla/SIMPLA_config.h"

#include <experimental/any>
#include <typeindex>
#include <vector>
#include "DataTraits.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/ObjectHead.h"

namespace simpla {
namespace data {
class DataLight;
class DataArray;

struct DataEntity : public std::enable_shared_from_this<DataEntity> {
    SP_OBJECT_BASE(DataEntity);

   protected:
    DataEntity() = default;

   public:
    virtual ~DataEntity() = default;
    SP_DEFAULT_CONSTRUCT(DataEntity)

    static auto New() { return std::shared_ptr<DataEntity>(new DataEntity); }

    virtual std::type_info const& value_type_info() const { return typeid(void); };
    virtual size_type value_type_size() const {return 0};
    virtual size_type rank() const { return 0; }
    virtual size_type extents(size_type* d) const { return rank(); }
    virtual size_type size() const { return 0; }

    virtual bool equal(DataEntity const& other) const { return false; }
    bool operator==(DataEntity const& other) { return equal(other); }

    /**
     *
     * @addtogroup experimental
     * @{
     */
    virtual std::experimental::any any() const override { return std::experimental::any(); }
    template <typename U>
    bool Check(U const& other) const {
        return value_type_info() == typeid(U) && rank() == 0 && std::experimental::any_cast<U>(*this) == other;
    }
    /**
     * @}
     */
};

}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
