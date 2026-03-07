/**
 * Fine raster helpers for scanline span clipping and edge tests.
 */
#pragma once

bool ClipRowSpanByEdge(float A, float row_term, inout int x_min, inout int x_max) {
    const float k_eps = 1e-8;
    const float k_bias = 1e-5;
    if (abs(A) <= k_eps) {
        return row_term >= -k_bias;
    }

    // A * (x + 0.5) + row_term >= 0
    float x_center_bound = (-row_term) / A;
    float x_bound = x_center_bound - 0.5;

    if (A > 0.0) {
        x_min = max(x_min, (int)ceil(x_bound - k_bias));
    } else {
        x_max = min(x_max, (int)floor(x_bound + k_bias));
    }
    return x_min <= x_max;
}

bool EdgePass(float w, bool inclusive) {
    const float k_eps = 1e-6;
    return inclusive ? (w >= -k_eps) : (w > k_eps);
}
