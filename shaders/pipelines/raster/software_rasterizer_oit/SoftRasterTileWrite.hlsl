/**
 * Soft Raster OIT — Tile List Write (Compute)
 *
 * Second binning pass. Re-evaluates tile overlap from the transformed
 * triangle buffer and writes triangle IDs into per-tile contiguous ranges.
 */

#include "shared/raster/software_rasterizer_oit/SoftRasterOITData.h"
#include "shared/raster/software_rasterizer_oit/SoftRasterParam.h"
#include "pipelines/raster/software_rasterizer_oit/utils/SoftRasterOverlapUtils.hlsli"
#include "pipelines/raster/software_rasterizer_oit/utils/SoftRasterTileUtils.hlsli"

// UAV bindings (set 0)
[[vk::binding(0, 0)]] RWBuffer<uint4> triangle_buf;
[[vk::binding(1, 0)]] RWBuffer<uint>  tile_count_buf;
[[vk::binding(2, 0)]] RWBuffer<uint>  tile_offset_buf;
[[vk::binding(3, 0)]] RWBuffer<uint>  tile_write_cursor_buf;
[[vk::binding(4, 0)]] RWBuffer<uint>  tile_tri_buf;
[[vk::binding(5, 0)]] RWBuffer<uint>  debug_stats_buf;

[[vk::push_constant]] ConstantBuffer<Moer::SoftRasterTileWriteParam> param;

[numthreads(SOFT_RASTER_SETUP_WG_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint tri_id = dtid.x;
    if (tri_id >= param.total_triangles) return;

    uint tb = tri_id * SOFT_RASTER_TRI_STRIDE;
    uint4 d0 = triangle_buf[tb + 0];
    uint4 d1 = triangle_buf[tb + 1];
    uint4 d2 = triangle_buf[tb + 2];

    // Skip invalid triangles marked by setup pass.
    if ((d2.w & SOFT_RASTER_TRI_FLAG_VALID) == 0u) return;

    float2 scr0 = float2(asfloat(d0.x), asfloat(d0.y));
    float2 scr1 = float2(asfloat(d0.z), asfloat(d0.w));
    float2 scr2 = float2(asfloat(d1.x), asfloat(d1.y));

    int2 tile_min, tile_max;
    if (!ComputeClampedTileBounds(scr0, scr1, scr2, param.tile_count_x, param.tile_count_y, tile_min, tile_max)) {
        return;
    }

    for (int ty = tile_min.y; ty <= tile_max.y; ++ty) {
        for (int tx = tile_min.x; tx <= tile_max.x; ++tx) {
            float2 tile_min_px = float2((float)(tx * SOFT_RASTER_TILE_SIZE), (float)(ty * SOFT_RASTER_TILE_SIZE)) - 0.5;
            float2 tile_max_px = tile_min_px + float2((float)SOFT_RASTER_TILE_SIZE + 1.0, (float)SOFT_RASTER_TILE_SIZE + 1.0);
            if (!TriangleIntersectsRect(scr0, scr1, scr2, tile_min_px, tile_max_px)) continue;

            uint tile_id = (uint)ty * param.tile_count_x + (uint)tx;
            uint tile_count = tile_count_buf[tile_id];
            if (tile_count == 0u) continue;

            uint slot;
            InterlockedAdd(tile_write_cursor_buf[tile_id], 1u, slot);
            if (slot < tile_count) {
                uint dst = tile_offset_buf[tile_id] + slot;
                tile_tri_buf[dst] = tri_id;
            } else {
                InterlockedAdd(debug_stats_buf[SOFT_RASTER_STAT_TILE_TRI_OVERFLOW], 1u);
            }
        }
    }
}
