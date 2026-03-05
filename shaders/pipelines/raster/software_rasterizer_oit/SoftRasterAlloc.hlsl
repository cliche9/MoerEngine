/**
 * Soft Raster OIT — Fragment Space Allocation  (Compute)
 *
 * For every pixel with fragment count > 0, atomically allocates a contiguous
 * block of (count + 1) entries in the fragment buffer (the extra slot is for
 * the end marker written by the Sort pass).
 */

#include "shared/raster/software_rasterizer_oit/SoftRasterOITData.h"
#include "shared/raster/software_rasterizer_oit/SoftRasterParam.h"

// UAV bindings (set 0)
[[vk::binding(0, 0)]] RWBuffer<uint> pixel_frag_count_buf;
[[vk::binding(1, 0)]] RWBuffer<uint> pixel_frag_offset_buf;
[[vk::binding(2, 0)]] RWBuffer<uint> alloc_counter_buf;
[[vk::binding(3, 0)]] RWBuffer<uint> debug_stats_buf;

[[vk::push_constant]] ConstantBuffer<Moer::SoftRasterAllocParam> param;

[numthreads(SOFT_RASTER_ALLOC_WG_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint pixel_idx = dtid.x;
    if (pixel_idx >= param.total_pixels) return;

    uint count = pixel_frag_count_buf[pixel_idx];
    if (count == 0) {
        pixel_frag_offset_buf[pixel_idx] = 0;
        return;
    }

    // Clamp to the per-pixel maximum before allocation
    if (count > (uint)SOFT_RASTER_MAX_FRAGS_PER_PIXEL) {
        InterlockedAdd(debug_stats_buf[SOFT_RASTER_STAT_PIXEL_CLAMPED], 1u);
    }
    count = min(count, (uint)SOFT_RASTER_MAX_FRAGS_PER_PIXEL);
    pixel_frag_count_buf[pixel_idx] = count; // write clamped value back

    // Allocate (count + 1) slots: count fragments + 1 end marker
    uint offset;
    InterlockedAdd(alloc_counter_buf[0], count + 1u, offset);

    if (offset + count + 1u > param.max_fragments) {
        // Pool overflow — mark with sentinel so Write/Sort stages skip this pixel
        InterlockedAdd(debug_stats_buf[SOFT_RASTER_STAT_POOL_OVERFLOW], 1u);
        pixel_frag_count_buf[pixel_idx]  = 0;
        pixel_frag_offset_buf[pixel_idx] = SOFT_RASTER_INVALID_FRAG_OFFSET;
        return;
    }

    pixel_frag_offset_buf[pixel_idx] = offset;
}
