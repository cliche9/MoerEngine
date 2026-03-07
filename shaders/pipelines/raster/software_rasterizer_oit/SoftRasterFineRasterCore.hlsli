/**
 * Soft Raster OIT — Fine Rasterization Core (Lucid-style bin list raster)
 *
 * One workgroup handles one tile selected from a compact bin list.
 * Within a tile, each thread owns one triangle per batch and rasterizes
 * covered scanline spans using precomputed edge equations.
 *
 * The including shader must provide:
 *   - WRITE_MODE define (0 = count, 1 = write)
 *   - buffers: triangle_buf, tile_count_buf, tile_tri_buf, tile_offset_buf,
 *              bin_list_buf, bin_count_buf, pixel_frag_count_buf,
 *              and in write mode pixel_frag_offset_buf + fragment_buf
 *   - push constant `param` of type SoftRasterTileParam
 */
#pragma once

#include "pipelines/raster/software_rasterizer_oit/utils/SoftRasterFineRasterUtils.hlsli"

#if WRITE_MODE && !SOFT_RASTER_USE_VISIBILITY_BUFFER
#include "shared/utils/Packing.h"
#include "pipelines/raster/software_rasterizer_oit/utils/SoftRasterMaterialSamplingUtils.hlsli"

float4 ShadeForwardFragment(uint instance_id, uint triangle_id, bool is_front,
    float3 bary)
{
    ArrayBuffer instance_buf = ArrayBuffer(param.instance_buf_hdl);
    ArrayBuffer primitive_buf = ArrayBuffer(param.primitive_buf_hdl);
    ArrayBuffer index_buf = ArrayBuffer(param.index_buf_hdl);

    Moer::GInstance inst = instance_buf.Load<Moer::GInstance>(instance_id);
    Moer::GPrimitive prim = primitive_buf.Load<Moer::GPrimitive>(inst.primitive_id);

    uint idx0 = index_buf.Load<uint>(prim.index_start_idx + triangle_id * 3 + 0);
    uint idx1 = index_buf.Load<uint>(prim.index_start_idx + triangle_id * 3 + 1);
    uint idx2 = index_buf.Load<uint>(prim.index_start_idx + triangle_id * 3 + 2);

    float4x4 model2world = inst.world_transform;
    float3x3 model2world_3x3 = (float3x3)model2world;

    ArrayBuffer position_buf = ArrayBuffer(param.position_buf_hdl);
    float3 local_p0 = position_buf.Load<float3>(prim.position_start_idx + idx0);
    float3 local_p1 = position_buf.Load<float3>(prim.position_start_idx + idx1);
    float3 local_p2 = position_buf.Load<float3>(prim.position_start_idx + idx2);

    float3 wp0 = mul(model2world, float4(local_p0, 1.0)).xyz;
    float3 wp1 = mul(model2world, float4(local_p1, 1.0)).xyz;
    float3 wp2 = mul(model2world, float4(local_p2, 1.0)).xyz;
    float3 world_pos = bary.x * wp0 + bary.y * wp1 + bary.z * wp2;

    float2 texcoord = float2(0.0, 0.0);
    if (prim.attribute_mask & Moer::GPrimitiveEAttributeMask::Texcoord0) {
        ArrayBuffer texcoord0_buf = ArrayBuffer(param.texcoord0_buf_hdl);
        float2 tc0 = texcoord0_buf.Load<float2>(prim.texcoord0_start_idx + idx0);
        float2 tc1 = texcoord0_buf.Load<float2>(prim.texcoord0_start_idx + idx1);
        float2 tc2 = texcoord0_buf.Load<float2>(prim.texcoord0_start_idx + idx2);
        texcoord = bary.x * tc0 + bary.y * tc1 + bary.z * tc2;
    }

    float3 normal = float3(0, 0, 1);
    if (prim.attribute_mask & Moer::GPrimitiveEAttributeMask::PackedNormal) {
        ArrayBuffer packed_normal_buf = ArrayBuffer(param.packed_normal_buf_hdl);
        float3 n0 = Moer::Unpack_Normal(
            packed_normal_buf.Load<uint>(prim.packed_normal_start_idx + idx0));
        float3 n1 = Moer::Unpack_Normal(
            packed_normal_buf.Load<uint>(prim.packed_normal_start_idx + idx1));
        float3 n2 = Moer::Unpack_Normal(
            packed_normal_buf.Load<uint>(prim.packed_normal_start_idx + idx2));
        normal = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
    }
    normal = normalize(mul(model2world_3x3, normal));

    float3 tangent = float3(0, 0, 0);
    if (prim.attribute_mask & Moer::GPrimitiveEAttributeMask::PackedTangent) {
        ArrayBuffer packed_tangent_buf = ArrayBuffer(param.packed_tangent_buf_hdl);
        float3 t0 = Moer::Unpack_Normal(
            packed_tangent_buf.Load<uint>(prim.packed_tangent_start_idx + idx0));
        float3 t1 = Moer::Unpack_Normal(
            packed_tangent_buf.Load<uint>(prim.packed_tangent_start_idx + idx1));
        float3 t2 = Moer::Unpack_Normal(
            packed_tangent_buf.Load<uint>(prim.packed_tangent_start_idx + idx2));
        tangent = normalize(bary.x * t0 + bary.y * t1 + bary.z * t2);
    } else {
        tangent = cross(normal, float3(0, 0, 1));
        if (length(tangent) < 1e-2) {
            tangent = cross(normal, float3(0, 1, 0));
        }
        tangent = normalize(tangent);
    }
    tangent = normalize(mul(model2world_3x3, tangent));

    ArrayBuffer material_buf = ArrayBuffer(param.material_buf_hdl);
    Moer::GMaterial mat = material_buf.Load<Moer::GMaterial>(prim.material_idx);

    float4 albedo_tex = GetTextureDataLevel<float4>(
        asint(mat.albedo_map_hdl), texcoord, float4(1.0, 1.0, 1.0, 1.0),
        float4(1.0, 1.0, 1.0, 1.0));
    float3 albedo_color = albedo_tex.rgb * mat.albedo_factor.rgb;
    float alpha = saturate(albedo_tex.a * mat.albedo_factor.a);

    float2 metallic_roughness = GetTextureDataLevel<float2>(
        asint(mat.metallic_roughness_map_hdl), texcoord,
        float2(mat.metallic_factor, mat.roughness_factor),
        float2(mat.metallic_factor, mat.roughness_factor));
    float metallic = metallic_roughness.x;
    float roughness = metallic_roughness.y;

    float3 N = GetNormalFromNormalMapLevel(asint(mat.normal_map_hdl), texcoord,
        normal, tangent);
    N = is_front ? N : -N;

    ArrayBuffer global_params = ArrayBuffer(param.global_param_handle);
    Moer::LightingData lighting_data = global_params.Load<Moer::LightingData>(0);

    float3 V = normalize(lighting_data.camera_position - world_pos);
    float NoV = saturate(dot(N, V));

    BRDFContext brdf_ctx;
    brdf_ctx.Init(roughness, albedo_color, metallic, N, V,
        TextureHandle(lighting_data.lut_ggx_emu_handle)
            .SampleLevel<float3>(float2(NoV, roughness), 0.0),
        TextureHandle(lighting_data.lut_ggx_eavg_handle)
            .SampleLevel<float3>(float2(0.0, roughness), 0.0));
    brdf_ctx.SetConfig(lighting_data.brdf_enable_multi_scatter,
        lighting_data.brdf_NDF_mode, lighting_data.brdf_G_mode,
        lighting_data.brdf_G_is_ibl);

    ArrayBuffer light_buf = ArrayBuffer(param.light_buf_hdl);
    LightContext light_ctx;
    light_ctx.Init(brdf_ctx, world_pos, lighting_data.lut_ggx_emu_handle);

    for (uint li = 0; li < lighting_data.light_count; ++li) {
        Moer::GLight light = light_buf.Load<Moer::GLight>(li);
        light_ctx.AccumulateLight(light, 1.0);
    }

    float3 shaded_color = light_ctx.GetResult();
    if (param.enable_extra_ambient != 0u) {
        shaded_color += param.extra_ambient_intensity * param.extra_ambient_color * brdf_ctx.albedo;
    }

    return float4(shaded_color, alpha);
}
#endif

[numthreads(SOFT_RASTER_TILE_SIZE, SOFT_RASTER_TILE_SIZE, 1)] void
main(uint3 group_id
     : SV_GroupID, uint3 local_id
     : SV_GroupThreadID, uint local_index
     : SV_GroupIndex) {
    uint bin_idx = group_id.x;
    uint bin_count = bin_count_buf[0];
    if (bin_idx >= bin_count) {
        return;
    }

    uint tile_id = bin_list_buf[bin_idx];
    uint tri_total = tile_count_buf[tile_id];
    uint tile_base = tile_offset_buf[tile_id];
    if (tri_total == 0u) {
        return;
    }

    uint tile_x = tile_id % param.tile_count_x;
    uint tile_y = tile_id / param.tile_count_x;

    int tile_min_x = (int)(tile_x * SOFT_RASTER_TILE_SIZE);
    int tile_min_y = (int)(tile_y * SOFT_RASTER_TILE_SIZE);
    int tile_max_x = min(tile_min_x + (int)SOFT_RASTER_TILE_SIZE - 1,
        (int)param.screen_width - 1);
    int tile_max_y = min(tile_min_y + (int)SOFT_RASTER_TILE_SIZE - 1,
        (int)param.screen_height - 1);
    if (tile_min_x > tile_max_x || tile_min_y > tile_max_y) {
        return;
    }

    for (uint batch_start = 0u; batch_start < tri_total;
         batch_start += SOFT_RASTER_TILE_WG_SIZE) {
        uint tri_slot = batch_start + local_index;
        if (tri_slot >= tri_total) {
            continue;
        }

        uint tri_id = tile_tri_buf[tile_base + tri_slot];
        uint tb = tri_id * SOFT_RASTER_TRI_STRIDE;

        uint4 d0 = triangle_buf[tb + 0];
        uint4 d1 = triangle_buf[tb + 1];
        uint4 d2 = triangle_buf[tb + 2];
        uint4 d3 = triangle_buf[tb + 3];
        uint4 d4 = triangle_buf[tb + 4];
        uint4 d5 = triangle_buf[tb + 5];

        uint tri_flags = d2.w;
        if ((tri_flags & SOFT_RASTER_TRI_FLAG_VALID) == 0u) {
            continue;
        }

        float2 v0 = float2(asfloat(d0.x), asfloat(d0.y));
        float2 v1 = float2(asfloat(d0.z), asfloat(d0.w));
        float2 v2 = float2(asfloat(d1.x), asfloat(d1.y));

        float z0 = asfloat(d1.z);
        float z1 = asfloat(d1.w);
        float z2 = asfloat(d2.x);

        float A0 = asfloat(d3.x);
        float A1 = asfloat(d3.y);
        float A2 = asfloat(d3.z);
        float inv_area = asfloat(d3.w);

        float B0 = asfloat(d4.x);
        float B1 = asfloat(d4.y);
        float B2 = asfloat(d4.z);

        float C0 = asfloat(d5.x);
        float C1 = asfloat(d5.y);
        float C2 = asfloat(d5.z);

        bool is_front = (tri_flags & SOFT_RASTER_TRI_FLAG_FRONT_FACE) != 0u;
        bool edge0_inclusive = (tri_flags & SOFT_RASTER_TRI_FLAG_EDGE0_TOP_LEFT) != 0u;
        bool edge1_inclusive = (tri_flags & SOFT_RASTER_TRI_FLAG_EDGE1_TOP_LEFT) != 0u;
        bool edge2_inclusive = (tri_flags & SOFT_RASTER_TRI_FLAG_EDGE2_TOP_LEFT) != 0u;

        int tri_min_x = max((int)floor(min(v0.x, min(v1.x, v2.x))), tile_min_x);
        int tri_max_x = min((int)ceil(max(v0.x, max(v1.x, v2.x))) - 1, tile_max_x);
        int tri_min_y = max((int)floor(min(v0.y, min(v1.y, v2.y))), tile_min_y);
        int tri_max_y = min((int)ceil(max(v0.y, max(v1.y, v2.y))) - 1, tile_max_y);

        if (tri_min_x > tri_max_x || tri_min_y > tri_max_y) {
            continue;
        }

        for (int y = tri_min_y; y <= tri_max_y; ++y) {
            float py = (float)y + 0.5;

            float row_term0 = B0 * py + C0;
            float row_term1 = B1 * py + C1;
            float row_term2 = B2 * py + C2;

            int row_min_x = tri_min_x;
            int row_max_x = tri_max_x;

            bool row_valid = true;
            row_valid = row_valid && ClipRowSpanByEdge(A0, row_term0, row_min_x, row_max_x);
            row_valid = row_valid && ClipRowSpanByEdge(A1, row_term1, row_min_x, row_max_x);
            row_valid = row_valid && ClipRowSpanByEdge(A2, row_term2, row_min_x, row_max_x);
            if (!row_valid) {
                continue;
            }

            float px = (float)row_min_x + 0.5;
            float w0 = A0 * px + row_term0;
            float w1 = A1 * px + row_term1;
            float w2 = A2 * px + row_term2;

            [loop] for (int x = row_min_x; x <= row_max_x; ++x)
            {
                if (EdgePass(w0, edge0_inclusive) && EdgePass(w1, edge1_inclusive) && EdgePass(w2, edge2_inclusive)) {
                    float depth = saturate((w0 * z0 + w1 * z1 + w2 * z2) * inv_area);
                    uint pixel_idx = (uint)y * param.screen_width + (uint)x;

#if WRITE_MODE
                    uint base_offset = pixel_frag_offset_buf[pixel_idx];
                    if (base_offset != SOFT_RASTER_INVALID_FRAG_OFFSET) {
                        uint packed_vis = ((d2.y & AOIT_INSTANCE_ID_MASK) << AOIT_INSTANCE_ID_SHIFT) | ((d2.z & AOIT_TRIANGLE_ID_MASK) << AOIT_TRIANGLE_ID_SHIFT) | (is_front ? 1u : 0u);

                        uint write_slot;
                        InterlockedAdd(pixel_frag_count_buf[pixel_idx], 1u, write_slot);

                        uint new_depth = asuint(depth);
                        if (write_slot < SOFT_RASTER_MAX_FRAGS_PER_PIXEL) {
                            uint frag_idx = base_offset + write_slot;
                            fragment_buf[frag_idx] = uint2(new_depth, packed_vis);
#if !SOFT_RASTER_USE_VISIBILITY_BUFFER
                            float3 bary = float3(w0, w1, w2) * inv_area;
                            float bary_sum = bary.x + bary.y + bary.z;
                            if (abs(bary_sum) > 1e-8) {
                                bary *= rcp(bary_sum);
                            } else {
                                bary = float3(1.0, 0.0, 0.0);
                            }
                            fragment_shade_buf[frag_idx] = ShadeForwardFragment(d2.y, d2.z, is_front, bary);
#endif
                        } else {
                            InterlockedAdd(
                                debug_stats_buf[SOFT_RASTER_STAT_PIXEL_WRITE_OVERFLOW], 1u);
                            // Drop overflowing fragments to avoid non-deterministic
                            // concurrent replacement races on fragment slots.
                        }
                    }
#else
                    InterlockedAdd(pixel_frag_count_buf[pixel_idx], 1u);
#endif
                }

                w0 += A0;
                w1 += A1;
                w2 += A2;
            }
        }
    }
}
