//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H

#include <typeindex>
#include <vector>

#include "simpla/config.h"
#include "simpla/utils/logger.h"

namespace simpla {

struct DataEntity : public std::enable_shared_from_this<DataEntity> {
   private:
    typedef DataEntity this_type;

   public:
    virtual std::string fancy_type_name() const { return "DataEntity"; }

   protected:
    DataEntity() = default;
    DataEntity(DataEntity const&) = default;

   public:
    virtual ~DataEntity() = default;

    static std::shared_ptr<DataEntity> create() { return std::shared_ptr<DataEntity>(new DataEntity); }
    std::shared_ptr<DataEntity> Copy() const { return nullptr; }
    virtual std::type_info const& value_type_info() const { return typeid(void); };
    virtual size_type value_alignof() const { return 0; };
    virtual size_type value_sizeof() const { return 0; };

    virtual size_type rank() const { return 0; }
    virtual size_type extents(size_type* d) const { return rank(); }
    virtual size_type size() const { return 0; }

    virtual bool isContinue() const { return true; }
    virtual size_type get_align_of() const { return size() * value_alignof(); }
    virtual size_type copy_out(void* dst) const { return 0; }
    virtual size_type copy_in(void const* src) { return 0; }
    virtual void* get_pointer() { return nullptr; }
    virtual void const* get_pointer() const { return nullptr; }

    virtual bool equal(DataEntity const& other) const { return false; }
    bool operator==(DataEntity const& other) { return equal(other); }

    virtual std::string __str__(std::ostream& os, int indent) const {
        std::ostream os os << "<N/A>";
        return os.str();
    }

    virtual std::shared_ptr<DataEntity> prepend(std::shared_ptr<DataEntity> const& v) const {
        return v == nullptr ? const_cast<this_type*>(this)->shared_from_this()
                            : v->append(const_cast<this_type*>(this)->shared_from_this());
    }
    virtual std::shared_ptr<DataEntity> append(std::shared_ptr<DataEntity> const& v) const {
        return DataEntity::create();
    }

    /**
     * @}
     */
};

inline std::ostream& operator<<(std::ostream& os, DataEntity const& v) {
    v.__str__(os, 0);
    return os;
}
}  // namespace simpla
#endif  // SIMPLA_DATAENTITY_H
