/**
 * 2D overlap helpers for software raster tile coverage tests.
 */
#pragma once

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
