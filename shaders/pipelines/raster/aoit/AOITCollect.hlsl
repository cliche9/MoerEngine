/**
 * AOIT Collect Pass - Pixel Shader
 *
 * Rasterizes transparent geometry using the hardware raster pipeline.
 * For each fragment, computes full forward shading (same as TransparentBlendPixel),
 * then atomically inserts the fragment into a per-pixel linked list.
 *
 * Data structures:
 *   head_pointer_tex  : RWTexture2D<uint>  - per-pixel linked list head (R32_UINT)
 *   counter_buf       : RWBuffer<uint>     - [0] = atomic counter, [1] = pool capacity
 *   fragment_pool_buf : RWBuffer<uint4>    - fragment pool
 *     .x = next pointer, .y = asuint(depth), .z = packed RG, .w = packed BA
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
#include "materials/Brdf.hlsli"
#include "pipelines/raster/deferred/lighting/Lighting.hlsli"
#include "pipelines/raster/deferred/geometry/GeometryPassCommon.hlsli"

// UAV bindings (set 0)
[[vk::binding(0, 0)]] RWTexture2D<uint> head_pointer_tex;
[[vk::binding(1, 0)]] RWBuffer<uint>    counter_buf;
[[vk::binding(2, 0)]] RWBuffer<uint4>   fragment_pool_buf;

// Push constant (same as TransparentBlendPass)
[[vk::push_constant]] ConstantBuffer<Moer::GeometryPassBindlessParam> param;

// ---- Pack helpers ----
uint PackRG(float r, float g) {
    return f32tof16(r) | (f32tof16(g) << 16);
}

uint PackBA(float b, float a) {
    return f32tof16(b) | (f32tof16(a) << 16);
}

float4 main(VsOutput input, bool is_front_face : SV_IsFrontFace) : SV_TARGET {
    // ---- Material fetch ----
    ArrayBuffer material_buf = ArrayBuffer(param.material_buf_hdl);
    Moer::GMaterial mat = material_buf.Load<Moer::GMaterial>(input.material_id);

    if (mat.alpha_mode != Moer::EAlphaMode::Blend) {
        discard;
    }

    // ---- Albedo ----
    int albedo_map_hdl = asint(mat.albedo_map_hdl);
    float4 albedo_tex = GetTextureData<float4>(
        albedo_map_hdl,
        input.texcoord0,
        float4(1.0, 1.0, 1.0, 1.0),
        float4(1.0, 1.0, 1.0, 1.0)
    );

    float3 albedo_color = albedo_tex.rgb * mat.albedo_factor.rgb;
    float alpha = saturate(albedo_tex.a * mat.albedo_factor.a);

    if (param.enable_alpha_test != 0 && alpha < param.alpha_test_blend_pixel_cutoff) {
        discard;
    }

    // ---- Metallic / Roughness ----
    float2 metallic_roughness = GetTextureData<float2>(
        asint(mat.metallic_roughness_map_hdl),
        input.texcoord0,
        float2(mat.metallic_factor, mat.roughness_factor),
        float2(mat.metallic_factor, mat.roughness_factor)
    );
    float metallic  = metallic_roughness.x;
    float roughness = metallic_roughness.y;

    // ---- Normal (two-sided) ----
    float3 N = GetNormalFromNormalMap(
        asint(mat.normal_map_hdl), input.texcoord0,
        normalize(input.normal), normalize(input.tangent)
    );
    N = is_front_face ? N : -N;

    // ---- Lighting ----
    ArrayBuffer global_params = ArrayBuffer(param.global_param_handle);
    Moer::LightingData lighting_data = global_params.Load<Moer::LightingData>(0);

    float3 V = normalize(lighting_data.camera_position - input.world_position.xyz);
    float NoV = saturate(dot(N, V));

    BRDFContext brdf_ctx;
    brdf_ctx.Init(
        roughness,
        albedo_color,
        metallic,
        N,
        V,
        TextureHandle(lighting_data.lut_ggx_emu_handle).Sample2D<float3>(float2(NoV, roughness)),
        TextureHandle(lighting_data.lut_ggx_eavg_handle).Sample2D<float3>(float2(0.0, roughness))
    );
    brdf_ctx.SetConfig(
        lighting_data.brdf_enable_multi_scatter,
        lighting_data.brdf_NDF_mode,
        lighting_data.brdf_G_mode,
        lighting_data.brdf_G_is_ibl
    );

    ArrayBuffer light_buf = ArrayBuffer(param.light_buf_hdl);
    LightContext light_ctx;
    light_ctx.Init(brdf_ctx, input.world_position, lighting_data.lut_ggx_emu_handle);

    for (uint i = 0; i < lighting_data.light_count; i++) {
        Moer::GLight light = light_buf.Load<Moer::GLight>(i);
        light_ctx.AccumulateLight(light, 1.0);
    }

    float3 shaded_color = light_ctx.GetResult();
    if (param.enable_extra_ambient != 0u) {
        shaded_color += param.extra_ambient_intensity * param.extra_ambient_color * brdf_ctx.albedo;
    }

    // ---- Allocate a slot in the fragment pool ----
    uint pool_capacity = counter_buf[1]; // set each frame by C++

    uint slot;
    InterlockedAdd(counter_buf[0], 1u, slot);

    if (slot >= pool_capacity) {
        // Pool overflow: graceful degradation via hardware alpha blend
        return float4(shaded_color * alpha, alpha);
    }

    // ---- Pack and store the fragment ----
    float depth = input.position.z;
    uint packed_rg = PackRG(shaded_color.r, shaded_color.g);
    uint packed_ba = PackBA(shaded_color.b, alpha);

    // Atomically insert at the head of the linked list for this pixel
    uint2 screen_pos = uint2(input.position.xy);

    uint old_head;
    InterlockedExchange(head_pointer_tex[screen_pos], slot, old_head);

    fragment_pool_buf[slot] = uint4(old_head, asuint(depth), packed_rg, packed_ba);

    return float4(0, 0, 0, 0);
}
