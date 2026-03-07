/**
 * Culling helpers for software raster setup.
 */
#pragma once

float3 SelectAABBCorner(float3 aabb_min, float3 aabb_max, uint corner_id) {
    return float3(
        (corner_id & 1u) ? aabb_max.x : aabb_min.x,
        (corner_id & 2u) ? aabb_max.y : aabb_min.y,
        (corner_id & 4u) ? aabb_max.z : aabb_min.z
    );
}

bool AABBOutsideClipFrustum(float4x4 world2clip, float4x4 model2world, float3 aabb_min, float3 aabb_max) {
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
        float4 clip = mul(world2clip, mul(model2world, float4(local_pos, 1.0)));

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
