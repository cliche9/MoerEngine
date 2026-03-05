/**
 * Soft Raster OIT — Tile List Allocation (Compute)
 *
 * Converts per-tile triangle counts into contiguous offsets in a global
 * tile-triangle list buffer. If the global list capacity is exceeded,
 * counts are clamped and overflow is accumulated into debug stats.
 */

#include "shared/raster/software_rasterizer_oit/SoftRasterOITData.h"
#include "shared/raster/software_rasterizer_oit/SoftRasterParam.h"

// UAV bindings (set 0)
[[vk::binding(0, 0)]] RWBuffer<uint> tile_count_buf;
[[vk::binding(1, 0)]] RWBuffer<uint> tile_offset_buf;
[[vk::binding(2, 0)]] RWBuffer<uint> tile_write_cursor_buf;
[[vk::binding(3, 0)]] RWBuffer<uint> debug_stats_buf;

[[vk::push_constant]] ConstantBuffer<Moer::SoftRasterTileAllocParam> param;

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x != 0) return;

    uint running  = 0u;
    uint overflow = 0u;

    for (uint tile_id = 0u; tile_id < param.total_tiles; ++tile_id) {
        uint count = tile_count_buf[tile_id];
        uint alloc_count = count;

        if (running >= param.max_tile_entries) {
            overflow += count;
            alloc_count = 0u;
        } else {
            uint remaining = param.max_tile_entries - running;
            if (alloc_count > remaining) {
                overflow += (alloc_count - remaining);
                alloc_count = remaining;
            }
        }

        tile_offset_buf[tile_id]       = running;
        tile_count_buf[tile_id]        = alloc_count;
        tile_write_cursor_buf[tile_id] = 0u;

        running += alloc_count;
    }

    debug_stats_buf[SOFT_RASTER_STAT_TOTAL_TILE_ENTRIES] = running;

    if (overflow > 0u) {
        InterlockedAdd(debug_stats_buf[SOFT_RASTER_STAT_TILE_TRI_OVERFLOW], overflow);
    }
}
