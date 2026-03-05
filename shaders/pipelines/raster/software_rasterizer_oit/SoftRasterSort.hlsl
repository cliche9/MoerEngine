/**
 * Soft Raster OIT — Per-Pixel Fragment Sort + Active Pixel Mapping  (Compute)
 *
 * For each pixel with fragments:
 *   1. Load fragments from fragment_buf into local arrays
 *   2. Insertion sort by depth (front-to-back, reversed-Z: larger = closer)
 *   3. Write sorted fragments back, followed by an end marker
 *   4. Append pixel to the active_pixel_buf mapping
 */

#include "shared/raster/software_rasterizer_oit/SoftRasterOITData.h"
#include "shared/raster/software_rasterizer_oit/SoftRasterParam.h"

// UAV bindings (set 0)
[[vk::binding(0, 0)]] RWBuffer<uint2> fragment_buf;
[[vk::binding(1, 0)]] RWBuffer<float4> fragment_shade_buf;
[[vk::binding(2, 0)]] RWBuffer<uint>  pixel_frag_count_buf;
[[vk::binding(3, 0)]] RWBuffer<uint>  pixel_frag_offset_buf;
[[vk::binding(4, 0)]] RWBuffer<uint2> active_pixel_buf;
[[vk::binding(5, 0)]] RWBuffer<uint>  active_pixel_count_buf;

[[vk::push_constant]] ConstantBuffer<Moer::SoftRasterSortParam> param;

[numthreads(SOFT_RASTER_SORT_WG_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint pixel_idx = dtid.x;
    if (pixel_idx >= param.total_pixels) return;

    uint count = pixel_frag_count_buf[pixel_idx];
    if (count == 0) return;

    uint offset = pixel_frag_offset_buf[pixel_idx];
    count = min(count, (uint)SOFT_RASTER_MAX_FRAGS_PER_PIXEL);

    // ---- Load fragments into local arrays ----
    uint local_depth[SOFT_RASTER_MAX_FRAGS_PER_PIXEL];
    uint local_vis  [SOFT_RASTER_MAX_FRAGS_PER_PIXEL];
#if !SOFT_RASTER_USE_VISIBILITY_BUFFER
    float4 local_shade[SOFT_RASTER_MAX_FRAGS_PER_PIXEL];
#endif

    for (uint i = 0; i < count; i++) {
        uint2 frag    = fragment_buf[offset + i];
        local_depth[i] = frag.x;
        local_vis[i]   = frag.y;
#if !SOFT_RASTER_USE_VISIBILITY_BUFFER
        local_shade[i] = fragment_shade_buf[offset + i];
#endif
    }

    // ---- Insertion sort descending by depth (reversed-Z: larger depth = closer) ----
    //      Tie-break by packed visibility id to keep ordering deterministic.
    for (uint i = 1; i < count; i++) {
        uint kd = local_depth[i];
        uint kv = local_vis[i];
#if !SOFT_RASTER_USE_VISIBILITY_BUFFER
        float4 ks = local_shade[i];
#endif
        uint j  = i;
        while (j > 0 &&
               (local_depth[j - 1] < kd || (local_depth[j - 1] == kd && local_vis[j - 1] > kv))) {
            local_depth[j] = local_depth[j - 1];
            local_vis[j]   = local_vis[j - 1];
#if !SOFT_RASTER_USE_VISIBILITY_BUFFER
            local_shade[j] = local_shade[j - 1];
#endif
            j--;
        }
        local_depth[j] = kd;
        local_vis[j]   = kv;
#if !SOFT_RASTER_USE_VISIBILITY_BUFFER
        local_shade[j] = ks;
#endif
    }

    // ---- Write sorted fragments + end marker ----
    for (uint i = 0; i < count; i++) {
        fragment_buf[offset + i] = uint2(local_depth[i], local_vis[i]);
#if !SOFT_RASTER_USE_VISIBILITY_BUFFER
        fragment_shade_buf[offset + i] = local_shade[i];
#endif
    }
    fragment_buf[offset + count] = uint2(SOFT_RASTER_FRAG_END_MARKER, 0);

    // ---- Append to active pixel mapping ----
    uint slot;
    InterlockedAdd(active_pixel_count_buf[0], 1u, slot);

    uint px = pixel_idx % param.screen_width;
    uint py = pixel_idx / param.screen_width;
    active_pixel_buf[slot] = uint2((py << 16) | px, offset);
}
