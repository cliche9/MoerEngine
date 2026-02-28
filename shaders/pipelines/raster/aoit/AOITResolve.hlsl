/**
 * AOIT Resolve Pass - Compute Shader
 *
 * For each pixel, walks the per-pixel linked list built by AOITCollect,
 * collects up to AOIT_MAX_SORT_COUNT fragments, sorts them front-to-back
 * by depth, then alpha-composites over the opaque scene color.
 */

#include "core/common/Bindless.hlsl"
BINDLESS_BINDINGS(3, 2, 4, 5)

#include "shared/raster/aoit/AOITData.h"

// UAV bindings (set 0)
[[vk::binding(0, 0)]] RWTexture2D<uint>      head_pointer_tex;
[[vk::binding(1, 0)]] RWBuffer<uint>         counter_buf;
[[vk::binding(2, 0)]] RWBuffer<uint4>        fragment_pool_buf;
[[vk::binding(3, 0)]] RWTexture2D<float4>    output_image;

// ---- Unpack helpers ----
float2 UnpackRG(uint packed) {
    return float2(f16tof32(packed & 0xFFFF), f16tof32(packed >> 16));
}

float2 UnpackBA(uint packed) {
    return float2(f16tof32(packed & 0xFFFF), f16tof32(packed >> 16));
}

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint2 pixel = dtid.xy;

    uint width = 0, height = 0;
    output_image.GetDimensions(width, height);

    if (pixel.x >= width || pixel.y >= height) {
        return;
    }

    // ---- Collect fragments from linked list ----
    float  local_depth[AOIT_MAX_SORT_COUNT];
    float3 local_color[AOIT_MAX_SORT_COUNT];
    float  local_alpha[AOIT_MAX_SORT_COUNT];
    uint   local_count = 0;

    uint node_index = head_pointer_tex[pixel];

    [loop]
    while (node_index != AOIT_INVALID_POINTER && local_count < AOIT_MAX_SORT_COUNT) {
        uint4 frag = fragment_pool_buf[node_index];

        float  depth = asfloat(frag.y);
        float2 rg    = UnpackRG(frag.z);
        float2 ba    = UnpackBA(frag.w);

        local_depth[local_count] = depth;
        local_color[local_count] = float3(rg.x, rg.y, ba.x);
        local_alpha[local_count] = ba.y;
        local_count++;

        node_index = frag.x; // next pointer
    }

    // No transparent fragments on this pixel -> keep opaque color
    if (local_count == 0) {
        return;
    }

    // ---- Insertion sort by depth (front-to-back for reversed-Z: larger depth = closer) ----
    for (uint i = 1; i < local_count; i++) {
        float  key_depth = local_depth[i];
        float3 key_color = local_color[i];
        float  key_alpha = local_alpha[i];

        uint j = i;
        while (j > 0 && local_depth[j - 1] < key_depth) {
            local_depth[j] = local_depth[j - 1];
            local_color[j] = local_color[j - 1];
            local_alpha[j] = local_alpha[j - 1];
            j--;
        }
        local_depth[j] = key_depth;
        local_color[j] = key_color;
        local_alpha[j] = key_alpha;
    }

    // ---- Alpha composite front-to-back over opaque background ----
    float4 opaque_color = output_image[pixel];

    float3 accum_color = float3(0.0, 0.0, 0.0);
    float  accum_transmittance = 1.0;

    for (uint k = 0; k < local_count; k++) {
        float a = local_alpha[k];
        accum_color += local_color[k] * a * accum_transmittance;
        accum_transmittance *= (1.0 - a);
    }

    // Blend accumulated transparent color over the opaque background
    float3 final_color = accum_color + opaque_color.rgb * accum_transmittance;

    output_image[pixel] = float4(final_color, opaque_color.a);
}
