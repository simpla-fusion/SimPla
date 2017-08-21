//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H
#include "simpla/SIMPLA_config.h"

#include <experimental/any>
#include <typeindex>
#include <vector>
#include "simpla/utilities/Log.h"
#include "simpla/utilities/ObjectHead.h"

namespace simpla {
namespace data {

struct DataEntity : public std::enable_shared_from_this<DataEntity> {
    SP_OBJECT_BASE(DataEntity);

   protected:
    DataEntity() = default;

   public:
    virtual ~DataEntity() = default;
    SP_DEFAULT_CONSTRUCT(DataEntity)

    static std::shared_ptr<DataEntity> New() { return std::shared_ptr<DataEntity>(new DataEntity); }

    virtual std::type_info const& value_type_info() const { return typeid(void); };
    virtual size_type value_type_size() const { return 0; };
    virtual size_type rank() const { return 0; }
    virtual size_type extents(size_type* d) const { return rank(); }
    virtual size_type size() const { return 0; }

    virtual std::ostream& Print(std::ostream& os, int indent = 0) const { return os; }

    virtual bool value_equal(void const* other, std::type_info const& info) const { return false; }

    virtual bool equal(DataEntity const& other) const { return false; }

    template <typename U>
    bool equal(U const& other) const {
        return value_equal(reinterpret_cast<void const*>(&other), typeid(U));
    }
    bool operator==(DataEntity const& other) { return equal(other); }

    /**
     *
     * @addtogroup experimental
     * @{
     */
    virtual std::experimental::any any() const { return std::experimental::any(); }
    template <typename U>
    U as() const {
        return std::experimental::any_cast<U>(any());
    }
    template <typename U>
    U as(U const& default_value) const {
        auto res = any();
        return res.empty() ? default_value : std::experimental::any_cast<U>(any());
    }
    template <typename U>
    bool Check(U const& other) const {
        return value_type_info() == typeid(U) && rank() == 0 && std::experimental::any_cast<U>(*this) == other;
    }
    /**
     * @}
     */
};

std::ostream& operator<<(std::ostream& os, DataEntity const&);
}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
