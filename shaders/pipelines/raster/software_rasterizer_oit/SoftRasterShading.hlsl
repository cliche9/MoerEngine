/**
 * Software Rasterizer OIT — Shading Pass (Compute Shader)
 *
 * Dispatched with one thread per active pixel.  Each thread reads its pixel
 * coordinates and fragment-buffer offset from the active_pixel_buf, walks
 * the sorted fragment list until the end marker, reconstructs geometry from
 * visibility data, evaluates full PBR shading, and alpha-composites
 * front-to-back over the opaque scene colour.
 *
 * Inputs:
 *   fragment_buf           — uint2[], sorted per pixel, end-marker terminated
 *   fragment_shade_buf     — float4[], sorted per pixel (forward path only)
 *   active_pixel_buf       — uint2[] (packed_pixel_pos, frag_offset)
 *   active_pixel_count_buf — uint[1]
 *   output_image           — opaque scene colour (read-modify-write)
 *
 * Fragment uint2 layout:
 *   .x = asuint(depth)
 *   .y = packed_vis_info   (see AOITData.h)
 *
 * packed_pixel_pos = (y << 16) | x
 */

#include "core/common/Bindless.hlsl"
#include "core/common/Common.hlsl"
BINDLESS_BINDINGS(3, 2, 4, 5)

#include "shared/raster/aoit/AOITData.h"
#include "shared/raster/aoit/AOITResolveParam.h"
#include "shared/raster/software_rasterizer_oit/SoftRasterOITData.h"
#include "shared/scene/SharedSceneStruct.h"
#include "shared/utils/Packing.h"

#include "materials/Brdf.hlsli"
#include "pipelines/raster/deferred/lighting/Lighting.hlsli"

// ---- UAV bindings (set 0) ----
[[vk::binding(0, 0)]] RWBuffer<uint2>       fragment_buf;
[[vk::binding(1, 0)]] RWBuffer<float4>      fragment_shade_buf;
[[vk::binding(2, 0)]] RWBuffer<uint2>       active_pixel_buf;
[[vk::binding(3, 0)]] RWBuffer<uint>        active_pixel_count_buf;
[[vk::binding(4, 0)]] RWTexture2D<float4>   output_image;

// Push constant
[[vk::push_constant]] ConstantBuffer<Moer::AOITResolveParam> resolve_param;

// ============================================================================
// Compute-shader safe texture sampling helpers (SampleLevel instead of Sample)
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

float4x4 Inverse4x4(float4x4 m) {
    float n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
    float n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
    float n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
    float n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

    float t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 -
                n23 * n32 * n44 + n22 * n33 * n44;
    float t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 +
                n13 * n32 * n44 - n12 * n33 * n44;
    float t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 -
                n13 * n22 * n44 + n12 * n23 * n44;
    float t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 +
                n13 * n22 * n34 - n12 * n23 * n34;

    float det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
    float idet = 1.0f / det;

    float4x4 ret;
    ret[0][0] = t11 * idet;
    ret[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 +
                 n23 * n31 * n44 - n21 * n33 * n44) * idet;
    ret[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 -
                 n22 * n31 * n44 + n21 * n32 * n44) * idet;
    ret[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 +
                 n22 * n31 * n43 - n21 * n32 * n43) * idet;

    ret[1][0] = t12 * idet;
    ret[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 -
                 n13 * n31 * n44 + n11 * n33 * n44) * idet;
    ret[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 +
                 n12 * n31 * n44 - n11 * n32 * n44) * idet;
    ret[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 -
                 n12 * n31 * n43 + n11 * n32 * n43) * idet;

    ret[2][0] = t13 * idet;
    ret[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 +
                 n13 * n21 * n44 - n11 * n23 * n44) * idet;
    ret[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 -
                 n12 * n21 * n44 + n11 * n22 * n44) * idet;
    ret[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 +
                 n12 * n21 * n43 - n11 * n22 * n43) * idet;

    ret[3][0] = t14 * idet;
    ret[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 -
                 n13 * n21 * n34 + n11 * n23 * n34) * idet;
    ret[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 +
                 n12 * n21 * n34 - n11 * n22 * n34) * idet;
    ret[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 -
                 n12 * n21 * n33 + n11 * n22 * n33) * idet;
    return ret;
}

// ============================================================================
// Fragment data
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

float4 ShadeFragment(
    FragmentVisData frag,
    uint2           pixel,
    uint            width,
    uint            height,
    float4x4        world2clip
) {
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

    // ---- Stable perspective-correct barycentrics from screen space ----
    float3 bary;
    float3 world_pos;
    {
        float4 cp0 = mul(world2clip, float4(wp0, 1.0));
        float4 cp1 = mul(world2clip, float4(wp1, 1.0));
        float4 cp2 = mul(world2clip, float4(wp2, 1.0));

        bool use_fallback = (cp0.w <= 1e-8) || (cp1.w <= 1e-8) || (cp2.w <= 1e-8);
        if (!use_fallback) {
            float2 sp0 = float2((cp0.x / cp0.w * 0.5 + 0.5) * width,  (cp0.y / cp0.w * -0.5 + 0.5) * height);
            float2 sp1 = float2((cp1.x / cp1.w * 0.5 + 0.5) * width,  (cp1.y / cp1.w * -0.5 + 0.5) * height);
            float2 sp2 = float2((cp2.x / cp2.w * 0.5 + 0.5) * width,  (cp2.y / cp2.w * -0.5 + 0.5) * height);
            float2 p   = float2(pixel) + 0.5;

            float area2 = (sp1.x - sp0.x) * (sp2.y - sp0.y) - (sp2.x - sp0.x) * (sp1.y - sp0.y);
            use_fallback = abs(area2) <= 1e-10;
            if (!use_fallback) {
                float l0 = ((sp1.x - p.x) * (sp2.y - p.y) - (sp2.x - p.x) * (sp1.y - p.y)) / area2;
                float l1 = ((sp2.x - p.x) * (sp0.y - p.y) - (sp0.x - p.x) * (sp2.y - p.y)) / area2;
                float l2 = 1.0 - l0 - l1;

                float rw0 = 1.0 / cp0.w;
                float rw1 = 1.0 / cp1.w;
                float rw2 = 1.0 / cp2.w;

                float b0 = l0 * rw0;
                float b1 = l1 * rw1;
                float b2 = l2 * rw2;
                float bsum = b0 + b1 + b2;
                use_fallback = abs(bsum) <= 1e-10;
                if (!use_fallback) {
                    bary = float3(b0, b1, b2) / bsum;
                    world_pos = bary.x * wp0 + bary.y * wp1 + bary.z * wp2;
                }
            }
        }

        if (use_fallback) {
            // Fallback to previous world-space barycentrics path.
            world_pos = ReconstructWorldPos(pixel, frag.depth, width, height);
            bary      = ComputeBarycentrics(world_pos, wp0, wp1, wp2);
        }
    }

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
// Main — one thread per active pixel
// ============================================================================

#define SOFT_RASTER_SHADING_WG_SIZE 256

[numthreads(SOFT_RASTER_SHADING_WG_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint thread_id = dtid.x;

    // Bounds check: only process active pixels
    uint active_count = active_pixel_count_buf[0];
    if (thread_id >= active_count) {
        return;
    }

    // Read active pixel entry
    uint2 ap = active_pixel_buf[thread_id];
    uint packed_pos = ap.x;
    uint frag_offset = ap.y;

    uint2 pixel;
    pixel.x = packed_pos & 0xFFFF;
    pixel.y = packed_pos >> 16;

    // Get output image dimensions
    uint width = 0, height = 0;
    output_image.GetDimensions(width, height);

    // Read opaque background colour
    float4 opaque_color = output_image[pixel];
#if SOFT_RASTER_USE_VISIBILITY_BUFFER
    float4x4 world2clip = Inverse4x4(resolve_param.clip2world);
#endif

    // ---- Walk sorted fragment list, shade & alpha-composite front-to-back ----
    float3 accum_color = float3(0.0, 0.0, 0.0);
    float  accum_transmittance = 1.0;

    for (uint i = 0; i < SOFT_RASTER_MAX_FRAGS_PER_PIXEL; i++) {
        uint2 frag_data = fragment_buf[frag_offset + i];

        // End-marker check
        if (frag_data.x == SOFT_RASTER_FRAG_END_MARKER) {
            break;
        }

        float4 shaded;
#if SOFT_RASTER_USE_VISIBILITY_BUFFER
        // Unpack visibility data
        FragmentVisData vis;
        vis.depth        = asfloat(frag_data.x);
        vis.instance_id  = (frag_data.y >> AOIT_INSTANCE_ID_SHIFT) & AOIT_INSTANCE_ID_MASK;
        vis.triangle_id  = (frag_data.y >> AOIT_TRIANGLE_ID_SHIFT) & AOIT_TRIANGLE_ID_MASK;
        vis.is_front_face = (frag_data.y & (1u << AOIT_FRONT_FACE_BIT)) != 0;
        shaded = ShadeFragment(vis, pixel, width, height, world2clip);
#else
        shaded = fragment_shade_buf[frag_offset + i];
#endif
        float  a = shaded.a;

        accum_color += shaded.rgb * a * accum_transmittance;
        accum_transmittance *= (1.0 - a);

        // Early out if nearly opaque
        if (accum_transmittance < 0.001) {
            break;
        }
    }

    // Blend accumulated transparent colour over the opaque background
    float3 final_color = accum_color + opaque_color.rgb * accum_transmittance;

    output_image[pixel] = float4(final_color, opaque_color.a);
}
