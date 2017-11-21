//
// Created by salmon on 17-2-13.
//

#ifndef SIMPLA_MESHBLOCK_H
#define SIMPLA_MESHBLOCK_H
#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/nTuple.ext.h>
#include <simpla/algebra/nTuple.h>
#include <simpla/data/Serializable.h>
#include <simpla/utilities/ObjectHead.h>
#include <memory>

namespace simpla {
namespace engine {
class MeshBlock : public data::Serializable {
    SP_SERIALIZABLE_HEAD(data::Serializable, MeshBlock)
    MeshBlock();
    explicit MeshBlock(index_box_type b, int level = 0, size_type local_id = 0);
    ~MeshBlock() override;

    static std::shared_ptr<MeshBlock> New(std::shared_ptr<const simpla::data::DataEntry> const &);
    static std::shared_ptr<MeshBlock> New(index_box_type const &box, int level = 0, size_type local_id = 0);

    std::shared_ptr<simpla::data::DataEntry> Serialize() const override;
    void Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) override;

   private:
    static constexpr int MAX_LEVEL_NUMBER = 8;

   public:
    bool operator==(MeshBlock const &other) const {
        return m_level_ == other.m_level_ && m_index_box_ == other.m_index_box_;
    };

    int GetLevel() const { return m_level_; }
    id_type GetLocalID() const { return m_local_id_; }
    id_type GetGUID() const;

    index_box_type GetIndexBox() const { return m_index_box_; }

    //    static id_type hash_id(id_type id = 0, int level = 0, int owner = 0);

   private:
    static size_type m_count_;
    int m_level_ = 0;
    id_type m_local_id_ = 0;
    index_box_type m_index_box_{{0, 0, 0}, {1, 1, 1}};

};  // class MeshBlock

}  // namespace engine{
}  // namespace simpla

#endif  // SIMPLA_MESHBLOCK_H
