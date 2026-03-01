/**
 * AOIT Collect Pass - Pixel Shader (Visibility Buffer)
 *
 * Rasterizes transparent geometry using the hardware raster pipeline.
 * Instead of computing full forward shading, only stores minimal visibility
 * data into a per-pixel linked list as a 64-bit vis_depth value.
 * Shading is deferred to the AOITResolve compute pass.
 *
 * Data structures:
 *   head_pointer_tex  : RWTexture2D<uint>  - per-pixel linked list head (R32_UINT)
 *   counter_buf       : RWBuffer<uint>     - [0] = atomic counter, [1] = pool capacity
 *   fragment_pool_buf : RWBuffer<uint4>    - fragment pool
 *     .x = next pointer
 *     .y = asuint(depth)     (vis_depth upper 32)
 *     .z = packed_vis_info   (vis_depth lower 32: instance_id | triangle_id | face)
 *     .w = reserved (0)
 */

#ifndef SHADOW_DEPTH_PASS
#define SHADOW_DEPTH_PASS 0
#endif

#include "core/common/Bindless.hlsl"
#include "core/common/Common.hlsl"
BINDLESS_BINDINGS(3, 2, 4, 5)

#include "shared/Geometry.h"
#include "shared/raster/ShaderParameters.h"
#include "shared/raster/aoit/AOITData.h"
#include "shared/scene/SharedSceneStruct.h"

#include "materials/Material.hlsli"
#include "pipelines/raster/deferred/geometry/GeometryPassCommon.hlsli"

// UAV bindings (set 0)
[[vk::binding(0, 0)]] RWTexture2D<uint> head_pointer_tex;
[[vk::binding(1, 0)]] RWBuffer<uint>    counter_buf;
[[vk::binding(2, 0)]] RWBuffer<uint4>   fragment_pool_buf;

// Push constant
[[vk::push_constant]] ConstantBuffer<Moer::GeometryPassBindlessParam> param;

void main(VsOutput input, uint triangle_id : SV_PrimitiveID, bool is_front_face : SV_IsFrontFace) {
    // ---- Alpha test (minimal material access) ----
    ArrayBuffer material_buf = ArrayBuffer(param.material_buf_hdl);
    Moer::GMaterial mat = material_buf.Load<Moer::GMaterial>(input.material_id);

    if (mat.alpha_mode != Moer::EAlphaMode::Blend) {
        discard;
    }

    // Sample albedo alpha for early-out on nearly-invisible fragments
    if (param.enable_alpha_test != 0) {
        int albedo_map_hdl = asint(mat.albedo_map_hdl);
        float4 albedo_tex = GetTextureData<float4>(
            albedo_map_hdl,
            input.texcoord0,
            float4(1.0, 1.0, 1.0, 1.0),
            float4(1.0, 1.0, 1.0, 1.0)
        );
        float alpha = saturate(albedo_tex.a * mat.albedo_factor.a);
        if (alpha < param.alpha_test_blend_pixel_cutoff) {
            discard;
        }
    }

    // ---- Allocate a slot in the fragment pool ----
    uint pool_capacity = counter_buf[1]; // set each frame by C++

    uint slot;
    InterlockedAdd(counter_buf[0], 1u, slot);

    if (slot >= pool_capacity) {
        // Pool overflow: fragment is lost (no forward shading fallback in visbuf mode)
        return;
    }

    // ---- Pack visibility data into 64-bit vis_depth and store ----
    float depth = input.position.z;

    // Pack instance_id, triangle_id, is_front_face into 32 bits
    uint packed_vis_info = ((input.instance_id & AOIT_INSTANCE_ID_MASK) << AOIT_INSTANCE_ID_SHIFT)
                         | ((triangle_id      & AOIT_TRIANGLE_ID_MASK) << AOIT_TRIANGLE_ID_SHIFT)
                         | (is_front_face ? 1u : 0u);

    // Atomically insert at the head of the linked list for this pixel
    uint2 screen_pos = uint2(input.position.xy);

    uint old_head;
    InterlockedExchange(head_pointer_tex[screen_pos], slot, old_head);

    // .yz together form the 64-bit vis_depth for future atomic depth testing
    fragment_pool_buf[slot] = uint4(old_head, asuint(depth), packed_vis_info, 0);
}
