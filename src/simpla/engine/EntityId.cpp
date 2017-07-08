//
// Created by salmon on 17-4-19.
//
#include "EntityId.h"
namespace simpla {

/**
 * Solve problem: Undefined reference to static constexpr char[]
 * http://stackoverflow.com/questions/22172789/passing-a-static-constexpr-variable-by-universal-reference
 */
//
constexpr int EntityIdCoder::ndims;
constexpr int EntityIdCoder::MESH_RESOLUTION;
constexpr Real EntityIdCoder::_R;
constexpr EntityId EntityIdCoder::_DK;
constexpr EntityId EntityIdCoder::_DJ;
constexpr EntityId EntityIdCoder::_DI;
constexpr EntityId EntityIdCoder::_DA;
constexpr int EntityIdCoder::m_id_to_index_[];
constexpr int EntityIdCoder::m_id_to_iform_[];
constexpr int EntityIdCoder::m_id_to_num_of_ele_in_cell_[];
constexpr int EntityIdCoder::m_adjacent_cell_num_[4][8];
constexpr int EntityIdCoder::m_iform_to_num_of_ele_in_cell_[];
constexpr EntityId EntityIdCoder::m_num_to_id_[];
constexpr EntityId EntityIdCoder::m_id_to_shift_[];
constexpr int EntityIdCoder::m_id_to_sub_index_[];
constexpr int EntityIdCoder::m_sub_index_to_id_[4][3];
constexpr EntityId EntityIdCoder::m_adjacent_cell_matrix_[4 /* to GetIFORM*/][NUM_OF_NODE_ID /* node id*/]
                                                         [MAX_NUM_OF_ADJACENT_CELL /*id shift*/];
constexpr Real EntityIdCoder::m_id_to_coordinates_shift_[8][3];
}