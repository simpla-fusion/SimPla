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
    virtual size_type value_alignof() const { return 0; };
    virtual size_type value_sizeof() const { return 0; };

    virtual size_type rank() const { return 0; }
    virtual size_type extents(size_type* d) const { return rank(); }
    virtual size_type size() const { return 0; }

    virtual bool isContinue() const { return true; }
    virtual size_type GetAlignOf() const { return size() * value_alignof(); }
    virtual size_type CopyOut(void* dst) const { return 0; }
    virtual size_type CopyIn(void const* src) { return 0; }
    virtual void* GetPointer() { return nullptr; }
    virtual void const* GetPointer() const { return nullptr; }

    virtual bool equal(DataEntity const& other) const { return false; }
    bool operator==(DataEntity const& other) { return equal(other); }

    virtual std::ostream& Print(std::ostream& os, int indent) const {
        os << "<N/A>";
        return os;
    }

    virtual std::shared_ptr<DataEntity> Prepend(std::shared_ptr<DataEntity> const& v) const {
        return v == nullptr ? const_cast<this_type*>(this)->shared_from_this()
                            : v->Append(const_cast<this_type*>(this)->shared_from_this());
    }
    virtual std::shared_ptr<DataEntity> Append(std::shared_ptr<DataEntity> const& v) const { return DataEntity::New(); }


    /**
     * @}
     */
};

std::ostream& operator<<(std::ostream& os, DataEntity const&);
}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITY_H
