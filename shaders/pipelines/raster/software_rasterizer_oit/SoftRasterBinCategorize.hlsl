/**
 * Soft Raster OIT — Bin Categorizer (Lucid-style low/high split)
 */

#include "shared/raster/software_rasterizer_oit/SoftRasterOITData.h"
#include "shared/raster/software_rasterizer_oit/SoftRasterParam.h"

[[vk::binding(0, 0)]] RWBuffer<uint> tile_count_buf;
[[vk::binding(1, 0)]] RWBuffer<uint> low_bin_list_buf;
[[vk::binding(2, 0)]] RWBuffer<uint> high_bin_list_buf;
[[vk::binding(3, 0)]] RWBuffer<uint> low_bin_count_buf;
[[vk::binding(4, 0)]] RWBuffer<uint> high_bin_count_buf;

[[vk::push_constant]] ConstantBuffer<Moer::SoftRasterBinCategorizeParam> param;

[numthreads(SOFT_RASTER_ALLOC_WG_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint tile_id = dtid.x;
    if (tile_id >= param.total_tiles) return;

    uint tri_count = tile_count_buf[tile_id];
    if (tri_count == 0u) return;

    if (tri_count < param.high_tri_threshold) {
        uint slot;
        InterlockedAdd(low_bin_count_buf[0], 1u, slot);
        low_bin_list_buf[slot] = tile_id;
    } else {
        uint slot;
        InterlockedAdd(high_bin_count_buf[0], 1u, slot);
        high_bin_list_buf[slot] = tile_id;
    }
}

