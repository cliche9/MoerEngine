/**
 * Soft Raster OIT — Tile List Write (Compute)
 *
 * Second binning pass. Re-evaluates tile overlap from the transformed
 * triangle buffer and writes triangle IDs into per-tile contiguous ranges.
 */

#include "shared/raster/software_rasterizer_oit/SoftRasterOITData.h"
#include "shared/raster/software_rasterizer_oit/SoftRasterParam.h"

// UAV bindings (set 0)
[[vk::binding(0, 0)]] RWBuffer<uint4> triangle_buf;
[[vk::binding(1, 0)]] RWBuffer<uint>  tile_count_buf;
[[vk::binding(2, 0)]] RWBuffer<uint>  tile_offset_buf;
[[vk::binding(3, 0)]] RWBuffer<uint>  tile_write_cursor_buf;
[[vk::binding(4, 0)]] RWBuffer<uint>  tile_tri_buf;
[[vk::binding(5, 0)]] RWBuffer<uint>  debug_stats_buf;

[[vk::push_constant]] ConstantBuffer<Moer::SoftRasterTileWriteParam> param;

bool PointInRect(float2 p, float2 rect_min, float2 rect_max) {
    return p.x >= rect_min.x && p.x <= rect_max.x && p.y >= rect_min.y && p.y <= rect_max.y;
}

float Orient2D(float2 a, float2 b, float2 c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool PointOnSegment(float2 a, float2 b, float2 p) {
    const float eps = 1e-5;
    if (abs(Orient2D(a, b, p)) > eps) return false;
    return p.x >= min(a.x, b.x) - eps && p.x <= max(a.x, b.x) + eps &&
           p.y >= min(a.y, b.y) - eps && p.y <= max(a.y, b.y) + eps;
}

bool SegmentsIntersect(float2 p0, float2 p1, float2 q0, float2 q1) {
    const float eps = 1e-5;

    float o1 = Orient2D(p0, p1, q0);
    float o2 = Orient2D(p0, p1, q1);
    float o3 = Orient2D(q0, q1, p0);
    float o4 = Orient2D(q0, q1, p1);

    bool straddle1 = (o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps);
    bool straddle2 = (o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps);
    if (straddle1 && straddle2) return true;

    if (abs(o1) <= eps && PointOnSegment(p0, p1, q0)) return true;
    if (abs(o2) <= eps && PointOnSegment(p0, p1, q1)) return true;
    if (abs(o3) <= eps && PointOnSegment(q0, q1, p0)) return true;
    if (abs(o4) <= eps && PointOnSegment(q0, q1, p1)) return true;
    return false;
}

bool PointInTriangle(float2 p, float2 a, float2 b, float2 c) {
    const float eps = 1e-5;
    float w0 = Orient2D(a, b, p);
    float w1 = Orient2D(b, c, p);
    float w2 = Orient2D(c, a, p);

    bool has_neg = (w0 < -eps) || (w1 < -eps) || (w2 < -eps);
    bool has_pos = (w0 > eps) || (w1 > eps) || (w2 > eps);
    return !(has_neg && has_pos);
}

bool TriangleIntersectsRect(float2 v0, float2 v1, float2 v2, float2 rect_min, float2 rect_max) {
    if (PointInRect(v0, rect_min, rect_max) ||
        PointInRect(v1, rect_min, rect_max) ||
        PointInRect(v2, rect_min, rect_max)) {
        return true;
    }

    float2 c0 = rect_min;
    float2 c1 = float2(rect_max.x, rect_min.y);
    float2 c2 = rect_max;
    float2 c3 = float2(rect_min.x, rect_max.y);

    if (PointInTriangle(c0, v0, v1, v2) ||
        PointInTriangle(c1, v0, v1, v2) ||
        PointInTriangle(c2, v0, v1, v2) ||
        PointInTriangle(c3, v0, v1, v2)) {
        return true;
    }

    if (SegmentsIntersect(v0, v1, c0, c1) || SegmentsIntersect(v0, v1, c1, c2) ||
        SegmentsIntersect(v0, v1, c2, c3) || SegmentsIntersect(v0, v1, c3, c0)) {
        return true;
    }
    if (SegmentsIntersect(v1, v2, c0, c1) || SegmentsIntersect(v1, v2, c1, c2) ||
        SegmentsIntersect(v1, v2, c2, c3) || SegmentsIntersect(v1, v2, c3, c0)) {
        return true;
    }
    if (SegmentsIntersect(v2, v0, c0, c1) || SegmentsIntersect(v2, v0, c1, c2) ||
        SegmentsIntersect(v2, v0, c2, c3) || SegmentsIntersect(v2, v0, c3, c0)) {
        return true;
    }

    return false;
}

[numthreads(SOFT_RASTER_SETUP_WG_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint tri_id = dtid.x;
    if (tri_id >= param.total_triangles) return;

    uint tb = tri_id * SOFT_RASTER_TRI_STRIDE;
    uint4 d0 = triangle_buf[tb + 0];
    uint4 d1 = triangle_buf[tb + 1];
    uint4 d2 = triangle_buf[tb + 2];

    // Skip invalid triangles marked by setup pass.
    if ((d2.w & SOFT_RASTER_TRI_FLAG_VALID) == 0u) return;

    float2 scr0 = float2(asfloat(d0.x), asfloat(d0.y));
    float2 scr1 = float2(asfloat(d0.z), asfloat(d0.w));
    float2 scr2 = float2(asfloat(d1.x), asfloat(d1.y));

    float2 bb_min = min(scr0, min(scr1, scr2));
    float2 bb_max = max(scr0, max(scr1, scr2));

    int2 tile_min = int2(floor(bb_min)) / (int)SOFT_RASTER_TILE_SIZE;
    int2 tile_max = (int2(ceil(bb_max)) - int2(1, 1)) / (int)SOFT_RASTER_TILE_SIZE;

    tile_min = max(tile_min, int2(0, 0));
    tile_max = min(tile_max, int2((int)param.tile_count_x - 1, (int)param.tile_count_y - 1));
    if (tile_min.x > tile_max.x || tile_min.y > tile_max.y) return;

    for (int ty = tile_min.y; ty <= tile_max.y; ++ty) {
        for (int tx = tile_min.x; tx <= tile_max.x; ++tx) {
            float2 tile_min_px = float2((float)(tx * SOFT_RASTER_TILE_SIZE), (float)(ty * SOFT_RASTER_TILE_SIZE)) - 0.5;
            float2 tile_max_px = tile_min_px + float2((float)SOFT_RASTER_TILE_SIZE + 1.0, (float)SOFT_RASTER_TILE_SIZE + 1.0);
            if (!TriangleIntersectsRect(scr0, scr1, scr2, tile_min_px, tile_max_px)) continue;

            uint tile_id = (uint)ty * param.tile_count_x + (uint)tx;
            uint tile_count = tile_count_buf[tile_id];
            if (tile_count == 0u) continue;

            uint slot;
            InterlockedAdd(tile_write_cursor_buf[tile_id], 1u, slot);
            if (slot < tile_count) {
                uint dst = tile_offset_buf[tile_id] + slot;
                tile_tri_buf[dst] = tri_id;
            } else {
                InterlockedAdd(debug_stats_buf[SOFT_RASTER_STAT_TILE_TRI_OVERFLOW], 1u);
            }
        }
    }
}
