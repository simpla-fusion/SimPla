/**
 *  @file MeshAtlas.cpp
 *  Created by salmon on 16-5-23.
 */
#include <mpi.h>
#include <simpla/geometry/BoxUtilities.h>
#include <simpla/geometry/Chart.h>
#include <simpla/parallel/MPIComm.h>
#include <simpla/parallel/MPIUpdater.h>
#include "simpla/SIMPLA_config.h"

#include "simpla/geometry/Chart.h"

#include <simpla/parallel/MPIComm.h>
#include <simpla/parallel/MPIUpdater.h>
#include "Atlas.h"
#include "MeshBlock.h"
//#include "TransitionMap.h"
//#include "simpla/utilities/BoxUtility.h"

namespace simpla {
namespace engine {

struct Atlas::pimpl_s {
    std::map<id_type, std::shared_ptr<Patch>> m_patches_;

    //    std::map<id_type, std::shared_ptr<MeshBlock>> m_blocks_;
    std::shared_ptr<geometry::Chart> m_chart_ = nullptr;
    //    static constexpr int MAX_NUM_OF_LEVEL = 5;
    //    typedef typename std::multimap<id_type, id_type>::iterator link_iterator;
    //    typedef typename std::multimap<id_type, id_type>::const_iterator const_link_iterator;
    //    typedef std::pair<const_link_iterator, const_link_iterator> multi_links_type;
    //    std::multimap<id_type, id_type> m_adjacent_;
    //    std::multimap<id_type, id_type> m_refine_;
    //    std::multimap<id_type, id_type> m_coarsen_;
    //    std::set<std::shared_ptr<data::DataNode>> m_layers_[MAX_NUM_OF_LEVEL];
    //    nTuple<int, 3> m_refine_ratio_[MAX_NUM_OF_LEVEL] = {{2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}};
    bool m_has_bounding_box_ = false;
    box_type m_box_{{0, 0, 0}, {1, 1, 1}};
    index_box_type m_global_index_box_{{0, 0, 0}, {1, 1, 1}};
    index_box_type m_index_box_{{0, 0, 0}, {1, 1, 1}};
    index_tuple m_ghost_width_{3, 3, 3};

    //    box_type m_local_box_{{0, 0, 0}, {1, 1, 1}};
    //    index_box_type m_index_box_{{0, 0, 0}, {1, 1, 1}};
    //    box_type m_global_box_{{0, 0, 0}, {1, 1, 1}};
    //    index_box_type m_global_index_box_{{0, 0, 0}, {1, 1, 1}};
};

Atlas::Atlas() : m_pimpl_(new pimpl_s) {
    SPObject::SetName("Atlas");
    SetMaxLevel(2);
    SetPeriodicDimensions(nTuple<int, 3>{0, 0, 0});

    SetMaxLevel(1);
    SetSmallestPatchDimensions(nTuple<int, 3>{4, 4, 4});
    SetLargestPatchDimensions(nTuple<int, 3>{128, 128, 128});
    SetRefineRatio(nTuple<int, 3>{2, 2, 2});
};
Atlas::~Atlas() { delete m_pimpl_; }

std::shared_ptr<data::DataNode> Atlas::Serialize() const {
    ASSERT(m_pimpl_->m_chart_ != nullptr);

    auto tdb = base_type::Serialize();

    tdb->Set("Chart", m_pimpl_->m_chart_->Serialize());
    auto patches = tdb->CreateNode("Patches", data::DataNode::DN_TABLE);
    for (auto const &item : m_pimpl_->m_patches_) { patches->Set(item.first, item.second->Serialize()); }
    return tdb;
};
void Atlas::Deserialize(std::shared_ptr<data::DataNode> const &tdb) {
    base_type::Deserialize(tdb);
    if (m_pimpl_->m_chart_ != nullptr) {
        m_pimpl_->m_chart_->Deserialize(tdb->Get("Chart"));
    } else {
        m_pimpl_->m_chart_ = geometry::Chart::New(tdb->Get("Chart"));
    }

    auto blocks = tdb->Get("Patches");
    blocks->Foreach([&](std::string const &key, std::shared_ptr<data::DataNode> const &patch) {
        auto res = m_pimpl_->m_patches_.emplace(std::stoi(key), Patch::New(patch));
    });

    Click();
};
std::shared_ptr<geometry::Chart> Atlas::GetChart() const { return m_pimpl_->m_chart_; }
void Atlas::SetChart(std::shared_ptr<geometry::Chart> const &c) { m_pimpl_->m_chart_ = c; }
void Atlas::DoSetUp() {
    ASSERT(m_pimpl_->m_chart_ != nullptr);  //{ m_pimpl_->m_chart_ = geometry::csCartesian::New(); }
    m_pimpl_->m_chart_->SetUp();
    point_type lo{0, 0, 0}, hi{0, 0, 0};
    std::tie(lo, hi) = m_pimpl_->m_box_;
    std::get<0>(m_pimpl_->m_index_box_) = ((lo - GetChart()->GetOrigin()) / GetChart()->GetScale());
    std::get<1>(m_pimpl_->m_index_box_) = ((hi - GetChart()->GetOrigin()) / GetChart()->GetScale());
    //    m_pimpl_->m_index_box_ = m_pimpl_->m_global_index_box_;
    //    m_pimpl_->m_global_index_box_ = m_pimpl_->m_global_index_box_;
    m_pimpl_->m_global_index_box_ = m_pimpl_->m_index_box_;
#ifdef MPI_FOUND
    {
        int mpi_ndims = 0;
        nTuple<int, 3> mpi_dims = {1, 1, 1};
        nTuple<int, 3> mpi_period = {1, 1, 1};
        nTuple<int, 3> mpi_coord = {0, 0, 0};

        GLOBAL_COMM.topology(&mpi_ndims, &mpi_dims[0], &mpi_period[0], &mpi_coord[0]);

        for (int i = 0; i < mpi_ndims; ++i) {
            std::get<0>(m_pimpl_->m_index_box_)[i] =
                std::get<0>(m_pimpl_->m_global_index_box_)[i] +
                (std::get<1>(m_pimpl_->m_global_index_box_)[i] - std::get<0>(m_pimpl_->m_global_index_box_)[i]) *
                    mpi_coord[i] / mpi_dims[i];
            std::get<1>(m_pimpl_->m_index_box_)[i] =
                std::get<0>(m_pimpl_->m_global_index_box_)[i] +
                (std::get<1>(m_pimpl_->m_global_index_box_)[i] - std::get<0>(m_pimpl_->m_global_index_box_)[i]) *
                    (mpi_coord[i] + 1) / mpi_dims[i];
        }
    }
#endif
    //    NewPatch(MeshBlock::New(m_pimpl_->m_index_box_, 0, 0));
};

void Atlas::DoUpdate() {
    if (m_pimpl_->m_chart_ != nullptr) { m_pimpl_->m_chart_->Update(); }
}

void Atlas::DoTearDown() {
    if (m_pimpl_->m_chart_ != nullptr) {
        m_pimpl_->m_chart_->TearDown();
        m_pimpl_->m_chart_.reset();
    }
};
std::shared_ptr<Patch> Atlas::NewPatch(box_type const &box, int level) {
    std::shared_ptr<Patch> res = nullptr;
    auto b =
        geometry::Overlap(m_pimpl_->m_index_box_,
                          std::make_tuple(std::get<1>(m_pimpl_->m_chart_->invert_local_coordinates(std::get<0>(box))),
                                          std::get<1>(m_pimpl_->m_chart_->invert_local_coordinates(std::get<1>(box)))));

    if (!geometry::isIllCondition(b)) { res = SetPatch(Patch::New(MeshBlock::New(b, level))); }
    return res;
}

std::shared_ptr<Patch> Atlas::SetPatch(std::shared_ptr<Patch> const &p) {
    if (p == nullptr) { return nullptr; }
    auto res = m_pimpl_->m_patches_.emplace(p->GetGUID(), p);
    if (!res.second) { res.first->second->Push(p); }
    return res.first->second;
}
std::shared_ptr<Patch> Atlas::GetPatch(id_type gid) const {
    auto it = m_pimpl_->m_patches_.find(gid);
    return it == m_pimpl_->m_patches_.end() ? nullptr : it->second;
}
size_type Atlas::DeletePatch(id_type id) { return m_pimpl_->m_patches_.erase(id); }

int Atlas::Foreach(std::function<void(std::shared_ptr<Patch> const &)> const &fun) {
    int count = 0;
    for (auto &item : m_pimpl_->m_patches_) {
        fun(item.second);
        ++count;
    }
    return count;
};
bool Atlas::hasBoundingBox() const { return m_pimpl_->m_has_bounding_box_; }

void Atlas::SetBoundingBox(box_type const &b) {
    m_pimpl_->m_box_ = b;
    m_pimpl_->m_has_bounding_box_ = true;
}
box_type Atlas::GetBoundingBox() const { return m_pimpl_->m_box_; }
index_box_type Atlas::GetGlobalIndexBox() const { return m_pimpl_->m_global_index_box_; };
index_box_type Atlas::GetBoundingIndexBox(int iform, int direction) const {
    index_tuple lo, hi;
    std::tie(lo, hi) = m_pimpl_->m_index_box_;
    switch (iform) {
        case NODE:
            hi += 1;
            break;
        case EDGE:
            hi[(direction + 1) % 3] += 1;
            hi[(direction + 2) % 3] += 1;

            break;
        case FACE:
            hi[(direction) % 3] += 1;
            break;
        case CELL:
        default:
            break;
    }
    return std::make_tuple(lo, hi);
}
index_box_type Atlas::GetBoundingHaloIndexBox(int tag, int direction) const {
    index_box_type res = GetBoundingIndexBox(tag, direction);
    std::get<0>(res) -= GetHaloWidth();
    std::get<1>(res) += GetHaloWidth();
    return res;
}

index_tuple Atlas::GetHaloWidth() const { return m_pimpl_->m_ghost_width_; }
void Atlas::SyncGlobal(std::string const &key, std::type_info const &t_info, int num_of_sub, int level) {
    std::shared_ptr<parallel::MPIUpdater> updater = nullptr;

    if (t_info == typeid(double)) {
        updater = parallel::MPIUpdater::New<double>();
    } else if (t_info == typeid(int)) {
        updater = parallel::MPIUpdater::New<int>();
    } else if (t_info == typeid(long)) {
        updater = parallel::MPIUpdater::New<long>();
    } else if (t_info == typeid(unsigned long)) {
        updater = parallel::MPIUpdater::New<unsigned long>();
    } else {
        UNIMPLEMENTED;
    }
    auto idx_box = GetBoundingIndexBox();
    auto halo_box = GetBoundingHaloIndexBox();

    for (int dir = 0; dir < 3; ++dir) {
        updater->SetIndexBox(idx_box);
        updater->SetHaloIndexBox(halo_box);
        updater->SetDirection(dir);
        updater->SetUp();
        for (int d = 0; d < num_of_sub; ++d) {
            updater->Clear();
            for (auto &item : m_pimpl_->m_patches_) {
                if (auto patch = item.second->GetDataBlock(key)) {
                    if (auto blk = patch->Get("_DATA_"))
                        if (auto data = std::dynamic_pointer_cast<ArrayBase>(blk->GetEntity(d))) {
                            updater->Push(*data);
                        };
                }
            }
            updater->SendRecv();
            for (auto &item : m_pimpl_->m_patches_) {
                if (auto patch = item.second->GetDataBlock(key)) {
                    if (auto blk = patch->Get("_DATA_"))
                        if (auto data = std::dynamic_pointer_cast<ArrayBase>(blk->GetEntity(d))) {
                            updater->Pop(*data);
                        }
                };
            }
        }
        updater->TearDown();
    }
}

void Atlas::SyncLocal(int level) {
    for (auto ia = m_pimpl_->m_patches_.begin(), ie = m_pimpl_->m_patches_.end(); ia != ie; ++ia) {
        auto ib = ia;
        ++ib;
        for (; ib != ie; ++ib) {
            auto box_a = ia->second->GetIndexBox();
            auto box_b = ib->second->GetIndexBox();
            index_box_type a_box;
            if (!geometry::isOverlapped(box_a, box_b)) { continue; }
            VERBOSE << box_a << box_b;

            for (auto const &item : ia->second->GetAllDataBlocks()) {
                auto attr_a = item.second->Get("_DATA_");
                auto attr_b = ib->second->GetDataBlock(item.first)->Get("_DATA_");
                for (int d = 0; d < attr_a->size(); ++d) {
                    auto array_a = std::dynamic_pointer_cast<ArrayBase>(attr_a->GetEntity(d));
                    auto array_b = std::dynamic_pointer_cast<ArrayBase>(attr_b->GetEntity(d));
                    //                    array_b->CopyIn(*array_a->GetSelectionP(box_a));
                    //                    array_a->CopyIn(*array_b->GetSelectionP(box_b));
                }
            };
        };
    };
}

// int Atlas::GetNumOfLevel() const { return m_pimpl_->(); }
// int Atlas::GetMaxLevel() const { return m_pimpl_->m_max_level_; }
// void Atlas::SetMaxLevel(int l) {
//    m_pimpl_->m_max_level_ = l;
//    Click();
//}
// void Atlas::SetRefineRatio(nTuple<int, 3> const &v, int level) {
//    m_pimpl_->m_refine_ratio_[level] = v;
//    m_pimpl_->m_level_ = std::max(level, m_pimpl_->m_level_);
//    Click();
//}
// nTuple<int, 3> Atlas::GetRefineRatio(int l) const { return m_pimpl_->m_refine_ratio_[l]; };
// void Atlas::SetLargestPatchDimensions(nTuple<int, 3> const &d) { m_pimpl_->m_largest_dimensions_ = d; };
// nTuple<int, 3> Atlas::GetLargestPatchDimensions() const { return m_pimpl_->m_largest_dimensions_; };
// void Atlas::SetSmallestPatchDimensions(nTuple<int, 3> const &d) { m_pimpl_->m_smallest_dimensions_ = d; };
// nTuple<int, 3> Atlas::GetSmallestPatchDimensions() const { return m_pimpl_->m_smallest_dimensions_; };
//
// void Atlas::SetPeriodicDimensions(nTuple<int, 3> const &b) { m_pimpl_->m_periodic_dimensions_ = b; }
// nTuple<int, 3> const &Atlas::GetPeriodicDimensions() const { return m_pimpl_->m_periodic_dimensions_; }
//
// void Atlas::SetCoarsestIndexBox(index_box_type const &b) { m_pimpl_->m_global_index_box_ = b; }
// index_box_type const &Atlas::GetCoarsestIndexBox() const { return m_pimpl_->m_global_index_box_; }

//        size_tuple Atlas::GetDimensions() const {
//            size_tuple d;
//            d = std::get<1>(m_pimpl_->m_global_index_box_) - std::get<0>(m_pimpl_->m_global_index_box_);
//            return d;
//        }
// size_type Atlas::size(int level) const { return m_database_->m_layer_[level].size(); }
// void Atlas::max_level(int ml) { m_database_->m_max_level_ = ml; }
// int Atlas::max_level() const { return m_database_->m_max_level_; }
// bool Atlas::has(id_type id) const { return m_database_->m_nodes_.find(id) != m_database_->m_nodes_.end(); };
//
// RectMesh *Atlas::find(id_type id) {
//    auto it = m_database_->m_nodes_.find(id);
//    if (it != m_database_->m_nodes_.end()) {
//        return it->m_node_.get();
//    } else {
//        return nullptr;
//    }
//}
//
// RectMesh const *Atlas::find(id_type id) const {
//    auto it = m_database_->m_nodes_.find(id);
//    if (it != m_database_->m_nodes_.end()) {
//        return it->m_node_.get();
//    } else {
//        return nullptr;
//    }
//}
//
// RectMesh *Atlas::at(id_type id) { return (m_database_->m_nodes_.at(id)).get(); };
//
// RectMesh const *Atlas::at(id_type id) const { return (m_database_->m_nodes_.at(id)).get(); };
//
// RectMesh const *Atlas::Connect(std::shared_ptr<RectMesh> const &p_m, RectMesh const *hint) {
////    m_pimpl_->m_nodes_.emplace(std::make_pair(p_m->id(), p_m));
//    return p_m.get();
//};
//
// std::ostream &Atlas::Print(std::ostream &os, int indent) const {
////    os << std::setw(indent) << "*" << name() << std::endl;
////    for (auto const &item : m_pimpl_->m_nodes_) {
////        os << "|" << std::setw(indent + 5 + item.second->level()) << std::setfill('-') << "> " << std::setfill('
///')
////           << std::setw(10) << std::left << item.first << std::right << " = {";
////        item.second->Print(os, indent + 1);
////        os << "}," << std::endl;
////    }
//    return os;
//};
//
// void Atlas::Load(const data::DataNode &) { UNIMPLEMENTED; }
//
// void Atlas::Save(data::DataNode *) const { UNIMPLEMENTED; }
//
// void Atlas::MPISync(id_type id)
//{
//    unlink(id);
//
//    auto it = m_nodes_.find(id);
//
//    if (it == m_nodes_.end())
//    {
//        return;
//    } else if (id != it->m_node_->id())
//    {
//        m_nodes_.Finalizie(it);
//        return;
//    }
//    RectMesh const &m = *(it->m_node_);
//    assert(m.level() < MAX_NUM_OF_LEVEL);
//    m_max_level_ = std::max(m_max_level_, m.level());
//    m_layer_[m.level()].Connect(id);
//
//    for (auto const &item:m_nodes_) { if (item.first != id) { link(id, item.first); }}
//}
//
// void Atlas::link(id_type i0, id_type i1)
//{
//    assert(has(i0) && find(i1));
//    RectMesh const &m0 = *at(i0);
//    RectMesh const &m1 = *at(i1);
//    int l0 = m0.level();
//    int l1 = m1.level();
//    box_type b0 = m0.box();
//    box_type b1 = m1.box();
//    vector_type dx = m0.dx();
//    vector_type L;//= m0.period_length();
//    switch (l0 - l1)
//    {
//        case 0:
//            if (utilities::check_adjoining(b0, b1, dx, L))
//            {
//                m_adjacent_.Connect(i0, i1);
//                m_adjacent_.Connect(i1, i0);
//            }
//            break;
//
//        case -1:
//            if (utilities::check_overlapping(b0, b1))
//            {
//                m_refine_.Connect(i0, i1);
//                m_coarsen_.Connect(i1, i0);
//            }
//            break;
//        case 1:
//            if (utilities::check_overlapping(b0, b1))
//            {
//                m_coarsen_.Connect(i0, i1);
//                m_refine_.Connect(i1, i0);
//            }
//            break;
//        default:
//            break;
//    }
//
//    return l0 - l1;
//}
//
// void Atlas::unlink(id_type id)
//{
//    m_adjacent_.Finalizie(id);
//    m_refine_.Finalizie(id);
//    m_coarsen_.Finalizie(id);
//    for (int i = 0; i < MAX_NUM_OF_LEVEL; ++i) { m_layer_[i].Finalizie(id); }
//}
//
//
// void Atlas::update_all()
//{
//    for (auto ib = m_nodes_.begin(), ie = m_nodes_.end(); ib != ie; ++ib)
//        for (auto it = ib; it != ie; ++it) { link(ib->first, it->first); }
//    m_max_level_ = 0;
//    for (unsigned int i = 0; i < MAX_NUM_OF_LEVEL; ++i)
//    {
//        if (m_layer_[i].size() == 0) { break; }
//        m_max_level_ = i;
//    }
//
//}
//
//
// std::shared_ptr<TransitionMapBase>
// Atlas::add_adjacency(MeshBlockId first, MeshBlockId m_node_)
//{
//    return add_adjacency(get_block(first), get_block(m_node_));
//}
//
// std::shared_ptr<TransitionMapBase>
// Atlas::add_adjacency(std::shared_ptr<const RectMesh> first, std::shared_ptr<const RectMesh> m_node_)
//{
//    return add_adjacency(std::make_shared<TransitionMapBase>(*first, *m_node_));
//}
//
// std::tuple<std::shared_ptr<TransitionMapBase>, std::shared_ptr<TransitionMapBase>>
// Atlas::add_connection(std::shared_ptr<const RectMesh> first, std::shared_ptr<const RectMesh> m_node_)
//{
//    return std::make_tuple(add_adjacency(first, m_node_), add_adjacency(m_node_, first));
//}
}  // namespace engine
}  // namespace simpla{namespace mesh_as{