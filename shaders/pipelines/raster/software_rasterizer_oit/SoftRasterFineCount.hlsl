/**
 * Soft Raster OIT — Fine Rasterization (Count Mode)
 *
 * Counts fragments per pixel.  No fragment writes.
 */
#define WRITE_MODE 0

#include "shared/raster/software_rasterizer_oit/SoftRasterOITData.h"
#include "shared/raster/software_rasterizer_oit/SoftRasterParam.h"
#include "shared/raster/aoit/AOITData.h"

// UAV bindings (set 0) — count mode with Lucid-style bin lists
[[vk::binding(0, 0)]] RWBuffer<uint4> triangle_buf;
[[vk::binding(1, 0)]] RWBuffer<uint>  tile_count_buf;
[[vk::binding(2, 0)]] RWBuffer<uint>  tile_tri_buf;
[[vk::binding(3, 0)]] RWBuffer<uint>  tile_offset_buf;
[[vk::binding(4, 0)]] RWBuffer<uint>  bin_list_buf;
[[vk::binding(5, 0)]] RWBuffer<uint>  bin_count_buf;
[[vk::binding(6, 0)]] RWBuffer<uint>  pixel_frag_count_buf;
[[vk::binding(7, 0)]] RWBuffer<uint>  debug_stats_buf;

[[vk::push_constant]] ConstantBuffer<Moer::SoftRasterTileParam> param;

#include "pipelines/raster/software_rasterizer_oit/SoftRasterFineRasterCore.hlsli"
