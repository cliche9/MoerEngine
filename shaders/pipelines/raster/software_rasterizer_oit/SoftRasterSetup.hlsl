/**
 * Soft Raster OIT — Setup (Lucid-style precompute stage)
 *
 * One thread per expanded triangle:
 *   - draw_cmd x instance x triangle mapping
 *   - object/world/clip transforms
 *   - conservative reject
 *   - precompute edge equations for scanline rasterization
 *
 * Output:
 *   triangle_buf — packed triangle data + scanline coefficients
 */

#include "core/common/Bindless.hlsl"
#include "core/common/Common.hlsl"
BINDLESS_BINDINGS(3, 2, 4, 5)

#include "shared/raster/software_rasterizer_oit/SoftRasterOITData.h"
#include "shared/raster/software_rasterizer_oit/SoftRasterParam.h"
#include "shared/scene/SharedSceneStruct.h"

[[vk::binding(0, 0)]] RWBuffer<uint4> triangle_buf;
[[vk::binding(1, 0)]] RWBuffer<uint>  debug_stats_buf;

[[vk::push_constant]] ConstantBuffer<Moer::SoftRasterSetupParam> param;

struct DrawCmd {
    uint index_cnt;
    uint instance_cnt;
    uint first_index;
    uint vertex_offset;
    uint first_instance;
};

float3 SelectAABBCorner(float3 aabb_min, float3 aabb_max, uint corner_id) {
    return float3(
        (corner_id & 1u) ? aabb_max.x : aabb_min.x,
        (corner_id & 2u) ? aabb_max.y : aabb_min.y,
        (corner_id & 4u) ? aabb_max.z : aabb_min.z
    );
}

bool AABBOutsideClipFrustum(
    float4x4 world2clip,
    float4x4 model2world,
    float3   aabb_min,
    float3   aabb_max
) {
    bool outside_left   = true;
    bool outside_right  = true;
    bool outside_bottom = true;
    bool outside_top    = true;
    bool outside_near   = true;
    bool outside_far    = true;
    bool outside_w      = true;

    [unroll]
    for (uint corner = 0u; corner < 8u; ++corner) {
        float3 local_pos = SelectAABBCorner(aabb_min, aabb_max, corner);
        float4 clip      = mul(world2clip, mul(model2world, float4(local_pos, 1.0)));

        outside_left   = outside_left && (clip.x < -clip.w);
        outside_right  = outside_right && (clip.x >  clip.w);
        outside_bottom = outside_bottom && (clip.y < -clip.w);
        outside_top    = outside_top && (clip.y >  clip.w);
        outside_near   = outside_near && (clip.z < 0.0);
        outside_far    = outside_far && (clip.z > clip.w);
        outside_w      = outside_w && (clip.w <= 0.0);
    }

    return outside_left || outside_right || outside_bottom || outside_top || outside_near ||
           outside_far || outside_w;
}

[numthreads(SOFT_RASTER_SETUP_WG_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint global_tri_id = dtid.x;
    if (global_tri_id >= param.total_triangles) return;

    uint tri_base = global_tri_id * SOFT_RASTER_TRI_STRIDE;
    triangle_buf[tri_base + 2].w = 0u;

    ArrayBuffer draw_cmd_buf = ArrayBuffer(param.draw_cmd_buf_hdl);

    DrawCmd the_cmd = (DrawCmd)0;
    uint    the_instance_id  = 0u;
    uint    the_triangle_idx = 0u;
    uint    running_offset   = 0u;
    bool    found_cmd        = false;

    for (uint d = 0u; d < param.num_draw_cmds; ++d) {
        DrawCmd cmd = draw_cmd_buf.Load<DrawCmd>(d);
        uint tris_per_inst  = cmd.index_cnt / 3u;
        uint expanded_count = tris_per_inst * cmd.instance_cnt;

        if (global_tri_id < running_offset + expanded_count) {
            the_cmd = cmd;
            uint local_id    = global_tri_id - running_offset;
            uint inst_in_cmd = local_id / tris_per_inst;
            the_triangle_idx = local_id % tris_per_inst;
            the_instance_id  = cmd.first_instance + inst_in_cmd;
            found_cmd        = true;
            break;
        }
        running_offset += expanded_count;
    }

    if (!found_cmd) return;

    ArrayBuffer instance_buf  = ArrayBuffer(param.instance_buf_hdl);
    ArrayBuffer primitive_buf = ArrayBuffer(param.primitive_buf_hdl);

    Moer::GInstance  inst = instance_buf.Load<Moer::GInstance>(the_instance_id);
    Moer::GPrimitive prim = primitive_buf.Load<Moer::GPrimitive>(inst.primitive_id);
    float4x4 model2world  = inst.world_transform;

    if (prim.local_aabb_min.w > 0.5) {
        float3 aabb_min = prim.local_aabb_min.xyz;
        float3 aabb_max = prim.local_aabb_max.xyz;
        if (all(aabb_min <= aabb_max) &&
            AABBOutsideClipFrustum(param.world2clip, model2world, aabb_min, aabb_max)) {
            return;
        }
    }

    ArrayBuffer index_buf = ArrayBuffer(param.index_buf_hdl);
    uint idx_base = the_cmd.first_index + the_triangle_idx * 3u;
    uint i0 = index_buf.Load<uint>(idx_base + 0u);
    uint i1 = index_buf.Load<uint>(idx_base + 1u);
    uint i2 = index_buf.Load<uint>(idx_base + 2u);

    ArrayBuffer position_buf = ArrayBuffer(param.position_buf_hdl);
    float3 pos0 = position_buf.Load<float3>(prim.position_start_idx + i0);
    float3 pos1 = position_buf.Load<float3>(prim.position_start_idx + i1);
    float3 pos2 = position_buf.Load<float3>(prim.position_start_idx + i2);

    float4 clip0 = mul(param.world2clip, mul(model2world, float4(pos0, 1.0)));
    float4 clip1 = mul(param.world2clip, mul(model2world, float4(pos1, 1.0)));
    float4 clip2 = mul(param.world2clip, mul(model2world, float4(pos2, 1.0)));

    if (clip0.x < -clip0.w && clip1.x < -clip1.w && clip2.x < -clip2.w) return;
    if (clip0.x >  clip0.w && clip1.x >  clip1.w && clip2.x >  clip2.w) return;
    if (clip0.y < -clip0.w && clip1.y < -clip1.w && clip2.y < -clip2.w) return;
    if (clip0.y >  clip0.w && clip1.y >  clip1.w && clip2.y >  clip2.w) return;
    if (clip0.z <  0.0     && clip1.z <  0.0     && clip2.z <  0.0    ) return;
    if (clip0.z >  clip0.w && clip1.z >  clip1.w && clip2.z >  clip2.w) return;

    if (clip0.w <= 0.0 || clip1.w <= 0.0 || clip2.w <= 0.0) return;

    float3 ndc0 = clip0.xyz / clip0.w;
    float3 ndc1 = clip1.xyz / clip1.w;
    float3 ndc2 = clip2.xyz / clip2.w;

    float2 ndc_min = min(ndc0.xy, min(ndc1.xy, ndc2.xy));
    float2 ndc_max = max(ndc0.xy, max(ndc1.xy, ndc2.xy));
    if (ndc_max.x < -1.0 || ndc_min.x > 1.0 || ndc_max.y < -1.0 || ndc_min.y > 1.0) return;

    float sw = (float)param.screen_width;
    float sh = (float)param.screen_height;
    float2 scr0 = float2((ndc0.x * 0.5 + 0.5) * sw, (ndc0.y * -0.5 + 0.5) * sh);
    float2 scr1 = float2((ndc1.x * 0.5 + 0.5) * sw, (ndc1.y * -0.5 + 0.5) * sh);
    float2 scr2 = float2((ndc2.x * 0.5 + 0.5) * sw, (ndc2.y * -0.5 + 0.5) * sh);

    float area2x = (scr1.x - scr0.x) * (scr2.y - scr0.y)
                 - (scr2.x - scr0.x) * (scr1.y - scr0.y);
    if (abs(area2x) < 1e-6) return;

    float A0 = scr1.y - scr2.y;
    float B0 = scr2.x - scr1.x;
    float C0 = scr1.x * scr2.y - scr1.y * scr2.x;

    float A1 = scr2.y - scr0.y;
    float B1 = scr0.x - scr2.x;
    float C1 = scr2.x * scr0.y - scr2.y * scr0.x;

    float A2 = scr0.y - scr1.y;
    float B2 = scr1.x - scr0.x;
    float C2 = scr0.x * scr1.y - scr0.y * scr1.x;

    bool  is_front = (area2x < 0.0);
    float area     = area2x;
    if (area < 0.0) {
        area = -area;
        A0 = -A0; B0 = -B0; C0 = -C0;
        A1 = -A1; B1 = -B1; C1 = -C1;
        A2 = -A2; B2 = -B2; C2 = -C2;
    }
    float inv_area = 1.0 / area;

    uint tri_flags = SOFT_RASTER_TRI_FLAG_VALID;
    if (is_front) tri_flags |= SOFT_RASTER_TRI_FLAG_FRONT_FACE;

    // Top-left rule in Y-down screen space:
    // edge is inclusive when (dy < 0) || (dy == 0 && dx > 0).
    // With edge equation A*x + B*y + C, dy = -A and dx = B.
    const float edge_eps = 1e-8;
    bool edge0_top_left = (A0 > edge_eps) || (abs(A0) <= edge_eps && B0 > edge_eps);
    bool edge1_top_left = (A1 > edge_eps) || (abs(A1) <= edge_eps && B1 > edge_eps);
    bool edge2_top_left = (A2 > edge_eps) || (abs(A2) <= edge_eps && B2 > edge_eps);
    if (edge0_top_left) tri_flags |= SOFT_RASTER_TRI_FLAG_EDGE0_TOP_LEFT;
    if (edge1_top_left) tri_flags |= SOFT_RASTER_TRI_FLAG_EDGE1_TOP_LEFT;
    if (edge2_top_left) tri_flags |= SOFT_RASTER_TRI_FLAG_EDGE2_TOP_LEFT;

    triangle_buf[tri_base + 0] = uint4(asuint(scr0.x), asuint(scr0.y), asuint(scr1.x), asuint(scr1.y));
    triangle_buf[tri_base + 1] = uint4(asuint(scr2.x), asuint(scr2.y), asuint(ndc0.z), asuint(ndc1.z));
    triangle_buf[tri_base + 2] = uint4(asuint(ndc2.z), the_instance_id, the_triangle_idx, tri_flags);
    triangle_buf[tri_base + 3] = uint4(asuint(A0), asuint(A1), asuint(A2), asuint(inv_area));
    triangle_buf[tri_base + 4] = uint4(asuint(B0), asuint(B1), asuint(B2), 0u);
    triangle_buf[tri_base + 5] = uint4(asuint(C0), asuint(C1), asuint(C2), 0u);
}
