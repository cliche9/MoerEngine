/**
 * Software Rasterizer OIT - Push Constant Parameter Structs
 *
 * Each pipeline stage uses a dedicated push constant struct.
 *
 * CPP:  #include "shaderheaders/shared/raster/software_rasterizer_oit/SoftRasterParam.h"
 * HLSL: #include "shared/raster/software_rasterizer_oit/SoftRasterParam.h"
 */
#pragma once

#ifdef __cplusplus
#include "misc/Traits.h"
namespace Moer::Render {
#else
namespace Moer {
#endif

/// Setup pass (triangle transform + precomputed raster equations)
struct SoftRasterSetupParam {
    float4x4 world2clip;

    // Bindless scene buffer handles
    uint instance_buf_hdl;
    uint primitive_buf_hdl;
    uint index_buf_hdl;
    uint position_buf_hdl;
    uint draw_cmd_buf_hdl;

    // Screen / tile dimensions
    uint screen_width;
    uint screen_height;

    uint tile_count_x;
    uint tile_count_y;
    uint num_draw_cmds;
    uint total_triangles;
};

/// Fine rasterization (Count / Write)
struct SoftRasterTileParam {
    uint screen_width;
    uint screen_height;
    uint tile_count_x;
    uint tile_count_y;

    // Scene geometry buffer handles (forward-shading path only)
    uint instance_buf_hdl;
    uint primitive_buf_hdl;
    uint index_buf_hdl;
    uint position_buf_hdl;
    uint packed_normal_buf_hdl;
    uint packed_tangent_buf_hdl;
    uint texcoord0_buf_hdl;
    uint material_buf_hdl;

    // Lighting buffers (forward-shading path only)
    uint light_buf_hdl;
    uint global_param_handle;

    // Extra ambient
    uint  enable_extra_ambient;
    uint  _pad0;
    float3 extra_ambient_color;
    float  extra_ambient_intensity;
};

/// Per-tile contiguous list allocation from global tile-triangle pool
struct SoftRasterTileAllocParam {
    uint total_tiles;
    uint max_tile_entries;
};

/// Bin passes (counter + dispatch) over transformed triangles
struct SoftRasterTileWriteParam {
    uint tile_count_x;
    uint tile_count_y;
    uint total_triangles;
};

/// Split tiles into low/high density bins (Lucid-style)
struct SoftRasterBinCategorizeParam {
    uint total_tiles;
    uint high_tri_threshold;
};

/// Fragment space allocation
struct SoftRasterAllocParam {
    uint total_pixels;
    uint max_fragments;
};

/// Per-pixel sort + active pixel mapping
struct SoftRasterSortParam {
    uint screen_width;
    uint total_pixels;
};

#ifdef __cplusplus
}
#else
}
#endif
