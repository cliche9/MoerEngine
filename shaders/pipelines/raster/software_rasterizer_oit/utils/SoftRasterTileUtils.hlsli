/**
 * Tile-space helpers for software raster binning passes.
 */
#pragma once

#include "shared/raster/software_rasterizer_oit/SoftRasterOITData.h"

bool ComputeClampedTileBounds(
    float2 v0,
    float2 v1,
    float2 v2,
    uint   tile_count_x,
    uint   tile_count_y,
    out int2 tile_min,
    out int2 tile_max
) {
    float2 bb_min = min(v0, min(v1, v2));
    float2 bb_max = max(v0, max(v1, v2));

    tile_min = int2(floor(bb_min)) / (int)SOFT_RASTER_TILE_SIZE;
    tile_max = (int2(ceil(bb_max)) - int2(1, 1)) / (int)SOFT_RASTER_TILE_SIZE;

    tile_min = max(tile_min, int2(0, 0));
    tile_max = min(tile_max, int2((int)tile_count_x - 1, (int)tile_count_y - 1));

    return tile_min.x <= tile_max.x && tile_min.y <= tile_max.y;
}
