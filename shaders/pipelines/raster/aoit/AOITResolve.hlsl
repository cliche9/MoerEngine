/**
 * AOIT Resolve Pass - Compute Shader (Visibility Buffer)
 *
 * For each pixel, walks the per-pixel linked list built by AOITCollect,
 * collects up to AOIT_MAX_SORT_COUNT visibility fragments, reconstructs
 * geometry from triangle IDs, evaluates full PBR shading, sorts
 * front-to-back by depth, then alpha-composites over the opaque scene color.
 *
 * Fragment layout (uint4):
 *   .x = next pointer
 *   .y = asuint(depth)
 *   .z = instance_id
 *   .w = triangle_id (bits 0..30) | is_front_face (bit 31)
 */

#include "core/common/Bindless.hlsl"
#include "core/common/Common.hlsl"
BINDLESS_BINDINGS(3, 2, 4, 5)

#include "shared/raster/aoit/AOITData.h"
#include "shared/raster/aoit/AOITResolveParam.h"
#include "shared/scene/SharedSceneStruct.h"
#include "shared/utils/Packing.h"

#include "materials/Brdf.hlsli"
#include "pipelines/raster/deferred/lighting/Lighting.hlsli"

// UAV bindings (set 0)
[[vk::binding(0, 0)]] RWTexture2D<uint>      head_pointer_tex;
[[vk::binding(1, 0)]] RWBuffer<uint>         counter_buf;
[[vk::binding(2, 0)]] RWBuffer<uint4>        fragment_pool_buf;
[[vk::binding(3, 0)]] RWTexture2D<float4>    output_image;

// Push constant
[[vk::push_constant]] ConstantBuffer<Moer::AOITResolveParam> resolve_param;

// ============================================================================
// Compute-shader safe texture sampling helpers (SampleLevel instead of Sample2D)
// ============================================================================

template <typename T>
T GetTextureDataLevel(int bindless_handle, float2 uv, T default_value, T missing_value) {
    if (bindless_handle >= 0) {
        return TextureHandle(bindless_handle).SampleLevel<T>(uv, 0.0);
    } else if (bindless_handle == -1) {
        return default_value;
    } else {
        return missing_value;
    }
}

float3 GetNormalFromNormalMapLevel(int normal_map, float2 uv, float3 normal, float3 tangent) {
    if (normal_map >= 0) {
        float3 normal_in_tbn = normalize((TextureHandle(normal_map).SampleLevel<float3>(uv, 0.0) * 2.0) - 1.0);
        float3 bitangent = cross(normal, tangent);
        float3x3 tbn = float3x3(tangent, bitangent, normal);
        return normalize(mul(normal_in_tbn, tbn));
    } else {
        return normal;
    }
}

// ============================================================================
// Barycentric computation
// ============================================================================

float3 ComputeBarycentrics(float3 world_pos, float3 v0, float3 v1, float3 v2) {
    float3 e0 = v1 - v0;
    float3 e1 = v2 - v0;
    float3 e2 = world_pos - v0;

    float d00 = dot(e0, e0);
    float d01 = dot(e0, e1);
    float d11 = dot(e1, e1);
    float d20 = dot(e2, e0);
    float d21 = dot(e2, e1);

    float inv_denom = 1.0 / (d00 * d11 - d01 * d01);
    float b1 = (d11 * d20 - d01 * d21) * inv_denom;
    float b2 = (d00 * d21 - d01 * d20) * inv_denom;
    float b0 = 1.0 - b1 - b2;
    return float3(b0, b1, b2);
}

// ============================================================================
// Reconstruct world position from screen pixel + depth + inverse VP matrix
// ============================================================================

float3 ReconstructWorldPos(uint2 pixel, float depth, uint width, uint height) {
    float2 screen_uv = (float2(pixel) + 0.5) / float2(width, height);
    float2 ndc = screen_uv * 2.0 - 1.0;
    ndc.y = -ndc.y; // Vulkan NDC has Y pointing down

    float4 clip = float4(ndc, depth, 1.0);
    float4 world_h = mul(resolve_param.clip2world, clip);
    return world_h.xyz / world_h.w;
}

// ============================================================================
// Data types for per-fragment storage during sort
// ============================================================================

struct FragmentVisData {
    float depth;
    uint  instance_id;
    uint  triangle_id;
    bool  is_front_face;
};

// ============================================================================
// Shade one fragment: reconstruct geometry from visibility, evaluate full PBR
// Returns (color.rgb, alpha)
// ============================================================================

float4 ShadeFragment(FragmentVisData frag, uint2 pixel, uint width, uint height) {
    // ---- Reconstruct world position from depth ----
    float3 world_pos = ReconstructWorldPos(pixel, frag.depth, width, height);

    // ---- Look up instance and primitive ----
    ArrayBuffer instance_buf  = ArrayBuffer(resolve_param.instance_buf_hdl);
    ArrayBuffer primitive_buf = ArrayBuffer(resolve_param.primitive_buf_hdl);
    ArrayBuffer index_buf     = ArrayBuffer(resolve_param.index_buf_hdl);

    Moer::GInstance  inst = instance_buf.Load<Moer::GInstance>(frag.instance_id);
    Moer::GPrimitive prim = primitive_buf.Load<Moer::GPrimitive>(inst.primitive_id);

    // ---- Fetch triangle vertex indices ----
    uint idx0 = index_buf.Load<uint>(prim.index_start_idx + frag.triangle_id * 3 + 0);
    uint idx1 = index_buf.Load<uint>(prim.index_start_idx + frag.triangle_id * 3 + 1);
    uint idx2 = index_buf.Load<uint>(prim.index_start_idx + frag.triangle_id * 3 + 2);

    // ---- Fetch vertex positions and transform to world space ----
    ArrayBuffer position_buf = ArrayBuffer(resolve_param.position_buf_hdl);
    float3 local_p0 = position_buf.Load<float3>(prim.position_start_idx + idx0);
    float3 local_p1 = position_buf.Load<float3>(prim.position_start_idx + idx1);
    float3 local_p2 = position_buf.Load<float3>(prim.position_start_idx + idx2);

    float4x4 model2world = inst.world_transform;
    float3 wp0 = mul(model2world, float4(local_p0, 1.0)).xyz;
    float3 wp1 = mul(model2world, float4(local_p1, 1.0)).xyz;
    float3 wp2 = mul(model2world, float4(local_p2, 1.0)).xyz;

    // ---- Compute barycentrics ----
    float3 bary = ComputeBarycentrics(world_pos, wp0, wp1, wp2);

    // ---- Interpolate texcoord ----
    float2 texcoord = float2(0, 0);
    if (prim.attribute_mask & Moer::GPrimitiveEAttributeMask::Texcoord0) {
        ArrayBuffer texcoord0_buf = ArrayBuffer(resolve_param.texcoord0_buf_hdl);
        float2 tc0 = texcoord0_buf.Load<float2>(prim.texcoord0_start_idx + idx0);
        float2 tc1 = texcoord0_buf.Load<float2>(prim.texcoord0_start_idx + idx1);
        float2 tc2 = texcoord0_buf.Load<float2>(prim.texcoord0_start_idx + idx2);
        texcoord = bary.x * tc0 + bary.y * tc1 + bary.z * tc2;
    }

    // ---- Interpolate and transform normal ----
    float3x3 model2world_3x3 = (float3x3)model2world;
    float3 normal = float3(0, 0, 1);
    if (prim.attribute_mask & Moer::GPrimitiveEAttributeMask::PackedNormal) {
        ArrayBuffer packed_normal_buf = ArrayBuffer(resolve_param.packed_normal_buf_hdl);
        float3 n0 = Moer::Unpack_Normal(packed_normal_buf.Load<uint>(prim.packed_normal_start_idx + idx0));
        float3 n1 = Moer::Unpack_Normal(packed_normal_buf.Load<uint>(prim.packed_normal_start_idx + idx1));
        float3 n2 = Moer::Unpack_Normal(packed_normal_buf.Load<uint>(prim.packed_normal_start_idx + idx2));
        normal = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
    }
    normal = normalize(mul(model2world_3x3, normal));

    // ---- Interpolate and transform tangent ----
    float3 tangent = float3(0, 0, 0);
    if (prim.attribute_mask & Moer::GPrimitiveEAttributeMask::PackedTangent) {
        ArrayBuffer packed_tangent_buf = ArrayBuffer(resolve_param.packed_tangent_buf_hdl);
        float3 t0 = Moer::Unpack_Normal(packed_tangent_buf.Load<uint>(prim.packed_tangent_start_idx + idx0));
        float3 t1 = Moer::Unpack_Normal(packed_tangent_buf.Load<uint>(prim.packed_tangent_start_idx + idx1));
        float3 t2 = Moer::Unpack_Normal(packed_tangent_buf.Load<uint>(prim.packed_tangent_start_idx + idx2));
        tangent = normalize(bary.x * t0 + bary.y * t1 + bary.z * t2);
    } else {
        tangent = cross(normal, float3(0, 0, 1));
        if (length(tangent) < 1e-2) {
            tangent = cross(normal, float3(0, 1, 0));
        }
        tangent = normalize(tangent);
    }
    tangent = normalize(mul(model2world_3x3, tangent));

    // ---- Fetch material ----
    ArrayBuffer material_buf = ArrayBuffer(resolve_param.material_buf_hdl);
    Moer::GMaterial mat = material_buf.Load<Moer::GMaterial>(prim.material_idx);

    // ---- Albedo ----
    float4 albedo_tex = GetTextureDataLevel<float4>(
        asint(mat.albedo_map_hdl), texcoord,
        float4(1.0, 1.0, 1.0, 1.0), float4(1.0, 1.0, 1.0, 1.0)
    );
    float3 albedo_color = albedo_tex.rgb * mat.albedo_factor.rgb;
    float  alpha = saturate(albedo_tex.a * mat.albedo_factor.a);

    // ---- Metallic / Roughness ----
    float2 metallic_roughness = GetTextureDataLevel<float2>(
        asint(mat.metallic_roughness_map_hdl), texcoord,
        float2(mat.metallic_factor, mat.roughness_factor),
        float2(mat.metallic_factor, mat.roughness_factor)
    );
    float metallic  = metallic_roughness.x;
    float roughness = metallic_roughness.y;

    // ---- Normal map (two-sided) ----
    float3 N = GetNormalFromNormalMapLevel(asint(mat.normal_map_hdl), texcoord, normal, tangent);
    N = frag.is_front_face ? N : -N;

    // ---- Lighting ----
    ArrayBuffer global_params = ArrayBuffer(resolve_param.global_param_handle);
    Moer::LightingData lighting_data = global_params.Load<Moer::LightingData>(0);

    float3 V   = normalize(lighting_data.camera_position - world_pos);
    float  NoV = saturate(dot(N, V));

    BRDFContext brdf_ctx;
    brdf_ctx.Init(
        roughness, albedo_color, metallic, N, V,
        TextureHandle(lighting_data.lut_ggx_emu_handle).SampleLevel<float3>(float2(NoV, roughness), 0.0),
        TextureHandle(lighting_data.lut_ggx_eavg_handle).SampleLevel<float3>(float2(0.0, roughness), 0.0)
    );
    brdf_ctx.SetConfig(
        lighting_data.brdf_enable_multi_scatter,
        lighting_data.brdf_NDF_mode,
        lighting_data.brdf_G_mode,
        lighting_data.brdf_G_is_ibl
    );

    ArrayBuffer light_buf = ArrayBuffer(resolve_param.light_buf_hdl);
    LightContext light_ctx;
    light_ctx.Init(brdf_ctx, world_pos, lighting_data.lut_ggx_emu_handle);

    for (uint i = 0; i < lighting_data.light_count; i++) {
        Moer::GLight light = light_buf.Load<Moer::GLight>(i);
        light_ctx.AccumulateLight(light, 1.0);
    }

    float3 shaded_color = light_ctx.GetResult();
    if (resolve_param.enable_extra_ambient != 0u) {
        shaded_color += resolve_param.extra_ambient_intensity * resolve_param.extra_ambient_color * brdf_ctx.albedo;
    }

    return float4(shaded_color, alpha);
}

// ============================================================================
// Main
// ============================================================================

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint2 pixel = dtid.xy;

    uint width = 0, height = 0;
    output_image.GetDimensions(width, height);

    if (pixel.x >= width || pixel.y >= height) {
        return;
    }

    // ---- Collect visibility fragments from linked list ----
    float local_depth[AOIT_MAX_SORT_COUNT];
    uint  local_vis[AOIT_MAX_SORT_COUNT];   // packed_vis_info (instance_id | triangle_id | face)
    uint  local_count = 0;

    uint node_index = head_pointer_tex[pixel];

    [loop]
    while (node_index != AOIT_INVALID_POINTER && local_count < AOIT_MAX_SORT_COUNT) {
        uint4 frag = fragment_pool_buf[node_index];

        local_depth[local_count] = asfloat(frag.y);
        local_vis[local_count]   = frag.z;
        local_count++;

        node_index = frag.x; // next pointer
    }

    // No transparent fragments on this pixel -> keep opaque color
    if (local_count == 0) {
        return;
    }

    // ---- Insertion sort by depth (front-to-back for reversed-Z: larger depth = closer) ----
    for (uint i = 1; i < local_count; i++) {
        float key_depth = local_depth[i];
        uint  key_vis   = local_vis[i];

        uint j = i;
        while (j > 0 && local_depth[j - 1] < key_depth) {
            local_depth[j] = local_depth[j - 1];
            local_vis[j]   = local_vis[j - 1];
            j--;
        }
        local_depth[j] = key_depth;
        local_vis[j]   = key_vis;
    }

    // ---- Shade, then alpha composite front-to-back over opaque background ----
    float4 opaque_color = output_image[pixel];

    float3 accum_color = float3(0.0, 0.0, 0.0);
    float  accum_transmittance = 1.0;

    for (uint k = 0; k < local_count; k++) {
        FragmentVisData vis;
        vis.depth        = local_depth[k];
        vis.instance_id  = (local_vis[k] >> AOIT_INSTANCE_ID_SHIFT) & AOIT_INSTANCE_ID_MASK;
        vis.triangle_id  = (local_vis[k] >> AOIT_TRIANGLE_ID_SHIFT) & AOIT_TRIANGLE_ID_MASK;
        vis.is_front_face = (local_vis[k] & (1u << AOIT_FRONT_FACE_BIT)) != 0;

        float4 shaded = ShadeFragment(vis, pixel, width, height);
        float  a = shaded.a;

        accum_color += shaded.rgb * a * accum_transmittance;
        accum_transmittance *= (1.0 - a);
    }

    // Blend accumulated transparent color over the opaque background
    float3 final_color = accum_color + opaque_color.rgb * accum_transmittance;

    output_image[pixel] = float4(final_color, opaque_color.a);
}
