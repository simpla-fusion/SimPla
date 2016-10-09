



SET(HYPRE_FOUND TRUE)
SET(HYPRE_INCLUDE_DIRS /usr)

SET(HYPRE_LIBRARIES
        HYPRE_FEI_fgmres
        HYPRE_DistributedMatrixPilutSolver
        HYPRE_sstruct_mv
        HYPRE_seq_mv
        HYPRE_krylov
        HYPRE_multivector
        HYPRE_ParaSails
        HYPRE_MatrixMatrix
        HYPRE_parcsr_mv
        HYPRE_utilities
        HYPRE_FEI
        HYPRE
        HYPRE_parcsr_block_mv
        HYPRE_struct_ls
        HYPRE_Euclid
        HYPRE_parcsr_ls
        HYPRE_DistributedMatrix
        HYPRE_mli
        HYPRE_sstruct_ls
        HYPRE_struct_mv
        HYPRE_IJ_mv
        )