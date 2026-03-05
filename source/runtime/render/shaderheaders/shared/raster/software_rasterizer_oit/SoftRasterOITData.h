/**
 * Software Rasterizer OIT (Lucid-inspired)
 *
 * Shared constants between HLSL and C++.
 *
 * Pipeline stages:
 *   1. Setup        — Transform triangles, precompute raster equations
 *   2. BinCounter   — Count triangle/tile overlaps
 *   3. BinPrefix    — Allocate contiguous tile ranges in global list
 *   4. BinDispatch  — Write triangle IDs into tile ranges
 *   5. BinCategorize— Split non-empty tiles into low/high bins
 *   6. FineCount    — Count fragments per pixel
 *   7. Alloc        — Per-pixel atomic allocation from global pool
 *   8. FineWrite    — Write fragments to allocated positions
 *   9. Sort         — Per-pixel insertion sort + build active pixel mapping
 *
 * Fragment buffer (RWBuffer<uint2>):
 *   Each entry = (asuint(depth), packed_vis_info).
 *   packed_vis_info reuses the 32-bit AOIT layout (see AOITData.h).
 *   Fragments for each pixel are stored contiguously, followed by an end
 *   marker uint2(0xFFFFFFFF, 0).
 *
 * Optional forward-shading payload (RWBuffer<float4>):
 *   When SOFT_RASTER_USE_VISIBILITY_BUFFER == 0, per-fragment shaded color
 *   (rgba) is written into a parallel fragment_shade_buf with the same index.
 *
 * Active pixel mapping (RWBuffer<uint2>):
 *   Per active-pixel entry: (packed_pixel_pos, frag_offset).
 *   packed_pixel_pos = (y << 16) | x.
 *
 * Triangle buffer (RWBuffer<uint4>):
 *   6 uint4 per transformed triangle:
 *     [0] = (scr0.x, scr0.y, scr1.x, scr1.y)      float bits
 *     [1] = (scr2.x, scr2.y, ndc_z0, ndc_z1)      float bits
 *     [2] = (ndc_z2, instance_id, triangle_id, flags)
 *     [3] = (edgeA0, edgeA1, edgeA2, inv_area)    float bits
 *     [4] = (edgeB0, edgeB1, edgeB2, reserved)    float bits
 *     [5] = (edgeC0, edgeC1, edgeC2, reserved)    float bits
 *
 * CPP:  #include "shaderheaders/shared/raster/software_rasterizer_oit/SoftRasterOITData.h"
 * HLSL: #include "shared/raster/software_rasterizer_oit/SoftRasterOITData.h"
 */
#pragma once

// 1 = visibility-buffer payload (instance/triangle id) + late shading
// 0 = forward payload (pre-shaded color) + lightweight resolve blend
#ifndef SOFT_RASTER_USE_VISIBILITY_BUFFER
#define SOFT_RASTER_USE_VISIBILITY_BUFFER 1
#endif

// ---- Tile configuration ----
#define SOFT_RASTER_TILE_SIZE    16
#define SOFT_RASTER_TILE_WG_SIZE (SOFT_RASTER_TILE_SIZE * SOFT_RASTER_TILE_SIZE)

// Per-tile list sizing heuristic (global tile list capacity = tile_count * this value)
// No hard per-tile cap is applied after tile-allocation pass.
#define SOFT_RASTER_MAX_TRIS_PER_TILE 2048

// Lucid-style tile split threshold:
// tri_count < threshold -> low path, otherwise high path.
#define SOFT_RASTER_BIN_HIGH_TRI_THRESHOLD 512

// Triangle buffer layout (uint4 slots per transformed triangle):
//   [0] screen position (v0.xy, v1.xy)
//   [1] screen position + depth (v2.xy, z0, z1)
//   [2] depth + visibility ids (z2, instance_id, triangle_id, flags)
//   [3] edge A coefficients + inv_area (A0, A1, A2, inv_area)
//   [4] edge B coefficients (B0, B1, B2, reserved)
//   [5] edge C coefficients (C0, C1, C2, reserved)
#define SOFT_RASTER_TRI_STRIDE 6

// Triangle flags stored in triangle_buf[2].w
#define SOFT_RASTER_TRI_FLAG_VALID      0x1u
#define SOFT_RASTER_TRI_FLAG_FRONT_FACE 0x2u
#define SOFT_RASTER_TRI_FLAG_EDGE0_TOP_LEFT 0x4u
#define SOFT_RASTER_TRI_FLAG_EDGE1_TOP_LEFT 0x8u
#define SOFT_RASTER_TRI_FLAG_EDGE2_TOP_LEFT 0x10u

// Fragment end marker (written after last fragment of a pixel)
#define SOFT_RASTER_FRAG_END_MARKER 0xFFFFFFFF

// Invalid fragment offset sentinel (marks pool-overflow pixels)
#define SOFT_RASTER_INVALID_FRAG_OFFSET 0xFFFFFFFFu

// Maximum fragments per pixel (caps the sort local array)
#define SOFT_RASTER_MAX_FRAGS_PER_PIXEL 64

// Workgroup sizes
#define SOFT_RASTER_SETUP_WG_SIZE 64
#define SOFT_RASTER_ALLOC_WG_SIZE 256
#define SOFT_RASTER_SORT_WG_SIZE  256

// Overflow statistic counters (RWBuffer<uint>)
#define SOFT_RASTER_STAT_TILE_TRI_OVERFLOW    0u
#define SOFT_RASTER_STAT_PIXEL_CLAMPED        1u
#define SOFT_RASTER_STAT_POOL_OVERFLOW        2u
#define SOFT_RASTER_STAT_PIXEL_WRITE_OVERFLOW 3u
#define SOFT_RASTER_STAT_MAX_TILE_TRI_COUNT   4u
#define SOFT_RASTER_STAT_TOTAL_TILE_ENTRIES   5u
#define SOFT_RASTER_STAT_COUNT                6u
