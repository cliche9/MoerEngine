/**
 * Soft Raster OIT — Bin Counter (Lucid-style)
 *
 * Counts triangle coverage per tile from precomputed setup data.
 */

#include "shared/raster/software_rasterizer_oit/SoftRasterOITData.h"
#include "shared/raster/software_rasterizer_oit/SoftRasterParam.h"

#include "pipelines/raster/software_rasterizer_oit/utils/SoftRasterOverlapUtils.hlsli"
#include "pipelines/raster/software_rasterizer_oit/utils/SoftRasterTileUtils.hlsli"

[[vk::binding(0, 0)]] RWBuffer<uint4> triangle_buf;
[[vk::binding(1, 0)]] RWBuffer<uint>  tile_count_buf;
[[vk::binding(2, 0)]] RWBuffer<uint>  debug_stats_buf;

[[vk::push_constant]] ConstantBuffer<Moer::SoftRasterTileWriteParam> param;

[numthreads(SOFT_RASTER_SETUP_WG_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint tri_id = dtid.x;
    if (tri_id >= param.total_triangles) return;

    uint tb = tri_id * SOFT_RASTER_TRI_STRIDE;
    uint4 d0 = triangle_buf[tb + 0];
    uint4 d1 = triangle_buf[tb + 1];
    uint4 d2 = triangle_buf[tb + 2];
    if ((d2.w & SOFT_RASTER_TRI_FLAG_VALID) == 0u) return;

    float2 scr0 = float2(asfloat(d0.x), asfloat(d0.y));
    float2 scr1 = float2(asfloat(d0.z), asfloat(d0.w));
    float2 scr2 = float2(asfloat(d1.x), asfloat(d1.y));

    int2 tile_min, tile_max;
    if (!ComputeClampedTileBounds(scr0, scr1, scr2, param.tile_count_x, param.tile_count_y, tile_min, tile_max)) {
        return;
    }

    [loop]
    for (int ty = tile_min.y; ty <= tile_max.y; ++ty) {
        [loop]
        for (int tx = tile_min.x; tx <= tile_max.x; ++tx) {
            float2 tile_min_px = float2((float)(tx * SOFT_RASTER_TILE_SIZE), (float)(ty * SOFT_RASTER_TILE_SIZE)) - 0.5;
            float2 tile_max_px = tile_min_px + float2((float)SOFT_RASTER_TILE_SIZE + 1.0, (float)SOFT_RASTER_TILE_SIZE + 1.0);
            if (!TriangleIntersectsRect(scr0, scr1, scr2, tile_min_px, tile_max_px)) continue;

            uint tile_id = (uint)ty * param.tile_count_x + (uint)tx;
            uint slot;
            InterlockedAdd(tile_count_buf[tile_id], 1u, slot);
            InterlockedMax(debug_stats_buf[SOFT_RASTER_STAT_MAX_TILE_TRI_COUNT], slot + 1u);
        }
    }
}
