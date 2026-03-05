#pragma once

#include "scene/camera/Camera.h"
#include "shader/ShaderCommon.h"
#include "shader/ShaderPipeline.h"
#include "shaderheaders/shared/raster/aoit/AOITData.h"
#include "shaderheaders/shared/raster/aoit/AOITResolveParam.h"
#include "shaderheaders/shared/raster/software_rasterizer_oit/SoftRasterOITData.h"
#include "shaderheaders/shared/raster/software_rasterizer_oit/SoftRasterParam.h"

#include "RasterConfig.h"
#include "RasterResource.h"

#include <array>

namespace Moer::Render::Raster {

struct SoftRasterVisibilityBufferMutation {
    void SetEnabled(bool enabled) {
        m_enabled = enabled;
    }

    void SetCompileEnvironment(ShaderCompilerEnvironment& env) const {
        env.SetDefine("SOFT_RASTER_USE_VISIBILITY_BUFFER", m_enabled);
    }

    uint32_t GetMutationID() const {
        return m_enabled ? 1u : 0u;
    }

private:
    bool m_enabled = true;
};

// ============================================================================
// Pipeline Definitions
// ============================================================================

/// Triangle setup + precompute raster equations (uses bindless scene buffers)
class SoftRasterSetupPipeline : public ComputePipeline {
public:
    DEFINE_COMPUTE_PIPELINE_CLASS(SoftRasterSetupPipeline);
    DEFINE_SHADER_BUFFER(triangle_buf);
    DEFINE_SHADER_BUFFER(debug_stats_buf);
    DEFINE_SHADER_BINDLESS_ARRAY(bdls);
    DEFINE_SHADER_CONSTANT_STRUCT(SoftRasterSetupParam, param);
    DEFINE_SHADER_ARGS(triangle_buf, debug_stats_buf, bdls, param);
};

/// Bin counter pass (triangle -> tile overlap count)
class SoftRasterBinCounterPipeline : public ComputePipeline {
public:
    DEFINE_COMPUTE_PIPELINE_CLASS(SoftRasterBinCounterPipeline);
    DEFINE_SHADER_BUFFER(triangle_buf);
    DEFINE_SHADER_BUFFER(tile_count_buf);
    DEFINE_SHADER_BUFFER(debug_stats_buf);
    DEFINE_SHADER_CONSTANT_STRUCT(SoftRasterTileWriteParam, param);
    DEFINE_SHADER_ARGS(triangle_buf, tile_count_buf, debug_stats_buf, param);
};

/// Per-tile list allocation (count -> offset + clamped count)
class SoftRasterBinPrefixPipeline : public ComputePipeline {
public:
    DEFINE_COMPUTE_PIPELINE_CLASS(SoftRasterBinPrefixPipeline);
    DEFINE_SHADER_BUFFER(tile_count_buf);
    DEFINE_SHADER_BUFFER(tile_offset_buf);
    DEFINE_SHADER_BUFFER(tile_write_cursor_buf);
    DEFINE_SHADER_BUFFER(debug_stats_buf);
    DEFINE_SHADER_CONSTANT_STRUCT(SoftRasterTileAllocParam, param);
    DEFINE_SHADER_ARGS(tile_count_buf, tile_offset_buf, tile_write_cursor_buf, debug_stats_buf, param);
};

/// Tile list write pass (writes triangle IDs into contiguous per-tile ranges)
class SoftRasterBinDispatchPipeline : public ComputePipeline {
public:
    DEFINE_COMPUTE_PIPELINE_CLASS(SoftRasterBinDispatchPipeline);
    DEFINE_SHADER_BUFFER(triangle_buf);
    DEFINE_SHADER_BUFFER(tile_count_buf);
    DEFINE_SHADER_BUFFER(tile_offset_buf);
    DEFINE_SHADER_BUFFER(tile_write_cursor_buf);
    DEFINE_SHADER_BUFFER(tile_tri_buf);
    DEFINE_SHADER_BUFFER(debug_stats_buf);
    DEFINE_SHADER_CONSTANT_STRUCT(SoftRasterTileWriteParam, param);
    DEFINE_SHADER_ARGS(
        triangle_buf,
        tile_count_buf,
        tile_offset_buf,
        tile_write_cursor_buf,
        tile_tri_buf,
        debug_stats_buf,
        param
    );
};

/// Split non-empty tiles into low/high density bins (Lucid-style)
class SoftRasterBinCategorizePipeline : public ComputePipeline {
public:
    DEFINE_COMPUTE_PIPELINE_CLASS(SoftRasterBinCategorizePipeline);
    DEFINE_SHADER_BUFFER(tile_count_buf);
    DEFINE_SHADER_BUFFER(low_bin_list_buf);
    DEFINE_SHADER_BUFFER(high_bin_list_buf);
    DEFINE_SHADER_BUFFER(low_bin_count_buf);
    DEFINE_SHADER_BUFFER(high_bin_count_buf);
    DEFINE_SHADER_CONSTANT_STRUCT(SoftRasterBinCategorizeParam, param);
    DEFINE_SHADER_ARGS(
        tile_count_buf,
        low_bin_list_buf,
        high_bin_list_buf,
        low_bin_count_buf,
        high_bin_count_buf,
        param
    );
};

/// Fine rasterization — count mode
class SoftRasterFineCountPipeline : public ComputePipeline {
public:
    DEFINE_COMPUTE_PIPELINE_CLASS(SoftRasterFineCountPipeline);
    DEFINE_SHADER_BUFFER(triangle_buf);
    DEFINE_SHADER_BUFFER(tile_count_buf);
    DEFINE_SHADER_BUFFER(tile_tri_buf);
    DEFINE_SHADER_BUFFER(tile_offset_buf);
    DEFINE_SHADER_BUFFER(bin_list_buf);
    DEFINE_SHADER_BUFFER(bin_count_buf);
    DEFINE_SHADER_BUFFER(pixel_frag_count_buf);
    DEFINE_SHADER_BUFFER(debug_stats_buf);
    DEFINE_SHADER_CONSTANT_STRUCT(SoftRasterTileParam, param);
    DEFINE_SHADER_ARGS(
        triangle_buf,
        tile_count_buf,
        tile_tri_buf,
        tile_offset_buf,
        bin_list_buf,
        bin_count_buf,
        pixel_frag_count_buf,
        debug_stats_buf,
        param
    );
};

/// Fine rasterization — write mode
class SoftRasterFineWritePipeline : public ComputePipeline {
public:
    DEFINE_COMPUTE_PIPELINE_CLASS(SoftRasterFineWritePipeline);
    using MutationSet = SoftRasterVisibilityBufferMutation;
    DEFINE_SHADER_BUFFER(triangle_buf);
    DEFINE_SHADER_BUFFER(tile_count_buf);
    DEFINE_SHADER_BUFFER(tile_tri_buf);
    DEFINE_SHADER_BUFFER(tile_offset_buf);
    DEFINE_SHADER_BUFFER(bin_list_buf);
    DEFINE_SHADER_BUFFER(bin_count_buf);
    DEFINE_SHADER_BUFFER(pixel_frag_count_buf);
    DEFINE_SHADER_BUFFER(pixel_frag_offset_buf);
    DEFINE_SHADER_BUFFER(fragment_buf);
    DEFINE_SHADER_BUFFER(fragment_shade_buf);
    DEFINE_SHADER_BUFFER(debug_stats_buf);
    DEFINE_SHADER_BINDLESS_ARRAY(bdls);
    DEFINE_SHADER_CONSTANT_STRUCT(SoftRasterTileParam, param);
    DEFINE_SHADER_ARGS(
        triangle_buf,
        tile_count_buf,
        tile_tri_buf,
        tile_offset_buf,
        bin_list_buf,
        bin_count_buf,
        pixel_frag_count_buf,
        pixel_frag_offset_buf,
        fragment_buf,
        fragment_shade_buf,
        debug_stats_buf,
        bdls,
        param
    );
};

/// Per-pixel fragment space allocation
class SoftRasterAllocPipeline : public ComputePipeline {
public:
    DEFINE_COMPUTE_PIPELINE_CLASS(SoftRasterAllocPipeline);
    DEFINE_SHADER_BUFFER(pixel_frag_count_buf);
    DEFINE_SHADER_BUFFER(pixel_frag_offset_buf);
    DEFINE_SHADER_BUFFER(alloc_counter_buf);
    DEFINE_SHADER_BUFFER(debug_stats_buf);
    DEFINE_SHADER_CONSTANT_STRUCT(SoftRasterAllocParam, param);
    DEFINE_SHADER_ARGS(
        pixel_frag_count_buf,
        pixel_frag_offset_buf,
        alloc_counter_buf,
        debug_stats_buf,
        param
    );
};

/// Per-pixel insertion sort + active pixel mapping construction
class SoftRasterSortPipeline : public ComputePipeline {
public:
    DEFINE_COMPUTE_PIPELINE_CLASS(SoftRasterSortPipeline);
    using MutationSet = SoftRasterVisibilityBufferMutation;
    DEFINE_SHADER_BUFFER(fragment_buf);
    DEFINE_SHADER_BUFFER(fragment_shade_buf);
    DEFINE_SHADER_BUFFER(pixel_frag_count_buf);
    DEFINE_SHADER_BUFFER(pixel_frag_offset_buf);
    DEFINE_SHADER_BUFFER(active_pixel_buf);
    DEFINE_SHADER_BUFFER(active_pixel_count_buf);
    DEFINE_SHADER_CONSTANT_STRUCT(SoftRasterSortParam, param);
    DEFINE_SHADER_ARGS(
        fragment_buf,
        fragment_shade_buf,
        pixel_frag_count_buf,
        pixel_frag_offset_buf,
        active_pixel_buf,
        active_pixel_count_buf,
        param
    );
};

/// Shading — VB mode: reconstruct+shade; Forward mode: direct blend from shaded payload
class SoftRasterShadingPipeline : public ComputePipeline {
public:
    DEFINE_COMPUTE_PIPELINE_CLASS(SoftRasterShadingPipeline);
    using MutationSet = SoftRasterVisibilityBufferMutation;
    DEFINE_SHADER_BUFFER(fragment_buf);
    DEFINE_SHADER_BUFFER(fragment_shade_buf);
    DEFINE_SHADER_BUFFER(active_pixel_buf);
    DEFINE_SHADER_BUFFER(active_pixel_count_buf);
    DEFINE_SHADER_TEX(output_image);
    DEFINE_SHADER_BINDLESS_ARRAY(bdls);
    DEFINE_SHADER_CONSTANT_STRUCT(AOITResolveParam, resolve_param);
    DEFINE_SHADER_ARGS(
        fragment_buf,
        fragment_shade_buf,
        active_pixel_buf,
        active_pixel_count_buf,
        output_image,
        bdls,
        resolve_param
    );
};

// ============================================================================
// SoftRasterOITPass
// ============================================================================

/**
 * GPU software-rasterization OIT (Lucid-inspired).
 *
 * Rasterizes transparent geometry entirely in compute shaders, producing a
 * visibility buffer of sorted per-pixel fragment lists.
 *
 * Pipeline:
 *   1. Setup        — transform + precompute triangle equations
 *   2. BinCounter   — triangle -> tile overlap counting
 *   3. BinPrefix    — contiguous global tile-list allocation
 *   4. BinDispatch  — write triangle IDs into tile lists
 *   5. BinCategorize— split non-empty tiles into low/high bins
 *   6. FineCount    — count fragments per pixel (low + high bins)
 *   7. Alloc        — atomic allocation of contiguous fragment storage
 *   8. FineWrite    — write fragments to allocated slots (low + high bins)
 *   9. Sort         — per-pixel depth sort + build active-pixel mapping
 *
 * Output (for downstream shading):
 *   fragment_buf           — uint2[] sorted per pixel,  end-marker terminated
 *   fragment_shade_buf     — float4[] sorted per pixel (forward mode only)
 *   active_pixel_buf       — uint2[] (packed_pos, offset)  per active pixel
 *   active_pixel_count_buf — uint[1] number of active pixels
 *   pixel_frag_count_buf   — uint[]  fragment count per pixel
 *   pixel_frag_offset_buf  — uint[]  starting frag-buffer offset per pixel
 */
class SoftRasterOITPass {
public:
    explicit SoftRasterOITPass(
        RasterContext& context,
        uint           max_triangles = 1u << 18, // 256 K
        uint           max_fragments = 1u << 23
    ) // 8 M
        :
        m_max_triangles(max_triangles),
        m_max_fragments(max_fragments) {

        // ---- Compile pipelines ----
        m_setup_pipeline = context.manager.Compute<SoftRasterSetupPipeline>(
            "pipelines/raster/software_rasterizer_oit/SoftRasterSetup.hlsl"
        );

        m_bin_counter_pipeline = context.manager.Compute<SoftRasterBinCounterPipeline>(
            "pipelines/raster/software_rasterizer_oit/SoftRasterBinCounter.hlsl"
        );

        m_bin_prefix_pipeline = context.manager.Compute<SoftRasterBinPrefixPipeline>(
            "pipelines/raster/software_rasterizer_oit/SoftRasterTileAlloc.hlsl"
        );

        m_bin_dispatch_pipeline = context.manager.Compute<SoftRasterBinDispatchPipeline>(
            "pipelines/raster/software_rasterizer_oit/SoftRasterBinDispatch.hlsl"
        );

        m_bin_categorize_pipeline = context.manager.Compute<SoftRasterBinCategorizePipeline>(
            "pipelines/raster/software_rasterizer_oit/SoftRasterBinCategorize.hlsl"
        );

        m_fine_count_pipeline = context.manager.Compute<SoftRasterFineCountPipeline>(
            "pipelines/raster/software_rasterizer_oit/SoftRasterFineCount.hlsl"
        );

        {
            SoftRasterFineWritePipeline::MutationSet mut_vb{};
            mut_vb.SetEnabled(true);
            m_fine_write_pipeline_vb = context.manager.Compute<SoftRasterFineWritePipeline>(
                "pipelines/raster/software_rasterizer_oit/SoftRasterFineWrite.hlsl",
                mut_vb
            );

            SoftRasterFineWritePipeline::MutationSet mut_forward{};
            mut_forward.SetEnabled(false);
            m_fine_write_pipeline_forward = context.manager.Compute<SoftRasterFineWritePipeline>(
                "pipelines/raster/software_rasterizer_oit/SoftRasterFineWrite.hlsl",
                mut_forward
            );
        }

        m_alloc_pipeline = context.manager.Compute<SoftRasterAllocPipeline>(
            "pipelines/raster/software_rasterizer_oit/SoftRasterAlloc.hlsl"
        );

        {
            SoftRasterSortPipeline::MutationSet mut_vb{};
            mut_vb.SetEnabled(true);
            m_sort_pipeline_vb = context.manager.Compute<SoftRasterSortPipeline>(
                "pipelines/raster/software_rasterizer_oit/SoftRasterSort.hlsl",
                mut_vb
            );

            SoftRasterSortPipeline::MutationSet mut_forward{};
            mut_forward.SetEnabled(false);
            m_sort_pipeline_forward = context.manager.Compute<SoftRasterSortPipeline>(
                "pipelines/raster/software_rasterizer_oit/SoftRasterSort.hlsl",
                mut_forward
            );
        }

        {
            SoftRasterShadingPipeline::MutationSet mut_vb{};
            mut_vb.SetEnabled(true);
            m_shading_pipeline_vb = context.manager.Compute<SoftRasterShadingPipeline>(
                "pipelines/raster/software_rasterizer_oit/SoftRasterShading.hlsl",
                mut_vb
            );

            SoftRasterShadingPipeline::MutationSet mut_forward{};
            mut_forward.SetEnabled(false);
            m_shading_pipeline_forward = context.manager.Compute<SoftRasterShadingPipeline>(
                "pipelines/raster/software_rasterizer_oit/SoftRasterShading.hlsl",
                mut_forward
            );
        }

        // ---- Allocate GPU buffers ----
        AllocateBuffers(context);
    }

    /// Re-create resolution-dependent buffers after window resize.
    void OnResize(RasterContext& context) {
        AllocateBuffers(context);
    }

    /**
     * Execute the full software-raster OIT pipeline.
     * After this call the sorted fragment buffer and active-pixel mapping
     * are ready for a downstream shading pass.
     */
    void Process(RasterContext& context, const RasterConfig& ui_config, const Camera& camera) {

        const auto& gpu = context.scene.gpu_scene_res();
        const auto& cpu = context.scene.cpu_scene();
        const bool  use_visibility_buffer = ui_config.soft_raster_oit_visibility_buffer_enable;
        auto& fine_write_pipeline = use_visibility_buffer ? m_fine_write_pipeline_vb : m_fine_write_pipeline_forward;
        auto& sort_pipeline = use_visibility_buffer ? m_sort_pipeline_vb : m_sort_pipeline_forward;
        auto& shading_pipeline = use_visibility_buffer ? m_shading_pipeline_vb : m_shading_pipeline_forward;

        SyncCapacityFromConfig(context, ui_config);
        LogDebugStatsIfNeeded(ui_config);

        if (!gpu.draw_cmd_alpha_blend_buf.buf || gpu.draw_cmd_alpha_blend_buf.buf->GetNumElement() == 0) {
            return;
        }

        // ---- Compute total expanded triangle count from CPU data ----
        const auto& alpha_cmds      = cpu.GetAlphaBlendDrawCmds();
        const uint  num_draw_cmds   = static_cast<uint>(alpha_cmds.size());
        uint        total_triangles = 0;
        for (const auto& cmd : alpha_cmds) {
            total_triangles += (cmd.index_cnt / 3) * cmd.instance_cnt;
        }
        if (total_triangles == 0)
            return;
        m_last_expanded_triangles = total_triangles;
        if (total_triangles > m_max_triangles) {
            if ((m_triangle_clamp_log_frame_id++ % 60u) == 0u) {
                LOG_WARNING(
                    "SoftRaster OIT triangle clamp: expanded_triangles={} exceeds max_triangles={}",
                    total_triangles,
                    m_max_triangles
                );
            }
            total_triangles = m_max_triangles;
        }

        const uint2 res      = context.GetResolutionOriginal();
        const uint  tile_cx  = (res.x + SOFT_RASTER_TILE_SIZE - 1) / SOFT_RASTER_TILE_SIZE;
        const uint  tile_cy  = (res.y + SOFT_RASTER_TILE_SIZE - 1) / SOFT_RASTER_TILE_SIZE;
        const uint  tile_cnt = tile_cx * tile_cy;
        const uint  total_px = res.x * res.y;

        // ==================================================================
        // 0. PushScope forces a reorderer layer break so that the
        //    CopyFrom below is in a LATER layer than Skybox Pass.
        //    Without this, the reorderer may place our compute
        //    dispatches in the SAME layer as Skybox (because
        //    m_dispatch_layer is never reset between commands).
        //    HandleBindless would then override Skybox's
        //    COLOR_ATTACHMENT / DEPTH_STENCIL states with
        //    SAMPLED_READ@COMPUTE_SHADER, producing wrong barriers.
        // ==================================================================
        context.cmd_list.PushScope("SoftRaster OIT Copy");

        // Copy opaque lighting_output -> resolve_output.
        // This is isolated in its own reorderer layer (between the two
        // PushScope calls) so that the barrier correctly transitions
        // lighting_output from COLOR_ATTACHMENT → TRANSFER_SRC.
        context.cmd_list.CopyFrom(
            context.textures.lighting_output.tex->GetView(),
            m_resolve_output_tex->GetView(),
            "SoftRaster Copy Opaque To Resolve"
        );

        // Second PushScope: forces compute dispatches into yet another
        // layer, preventing HandleBindless from overriding the CopyFrom's
        // TRANSFER state for lighting_output within the same layer.
        //
        // NOTE: The reorderer does NOT enforce write-after-write (WAW)
        // dependencies — GetLastLayerWrite only checks READ layers (for
        // WAR), not WRITE layers.  Because all our buffer bindings are
        // RWBuffer (treated as writes by the reorderer), no READ layers
        // are ever recorded, and every command would collapse into a
        // single reorderer layer without explicit PushScope breaks.
        // We therefore wrap EACH pipeline stage in its own PushScope /
        // PopScope pair so the reorderer places them in separate layers,
        // guaranteeing Vulkan memory barriers between stages.
        // ==================================================================

        const uint setup_groups = (total_triangles + SOFT_RASTER_SETUP_WG_SIZE - 1) / SOFT_RASTER_SETUP_WG_SIZE;

        // ==================================================================
        // 1. Clear counters/buffers used by this frame
        // ==================================================================
        context.cmd_list.PushScope("SoftRaster Clear");
        context.cmd_list.ClearResource(m_tile_count_buf->GetView(), (uint32_t)0);
        context.cmd_list.ClearResource(m_low_bin_count_buf->GetView(), (uint32_t)0);
        context.cmd_list.ClearResource(m_high_bin_count_buf->GetView(), (uint32_t)0);
        context.cmd_list.ClearResource(m_pixel_frag_count_buf->GetView(), (uint32_t)0);
        context.cmd_list.ClearResource(m_alloc_counter_buf->GetView(), (uint32_t)0);
        context.cmd_list.ClearResource(m_active_pixel_count_buf->GetView(), (uint32_t)0);
        context.cmd_list.ClearResource(m_debug_stats_buf->GetView(), (uint32_t)0);
        context.cmd_list.PopScope();

        // ==================================================================
        // 2. Setup: transform + precompute triangle equations
        // ==================================================================
        context.cmd_list.PushScope("SoftRaster Setup");
        {
            SoftRasterSetupParam sp{};
            sp.world2clip        = Transpose(camera.GetViewProjectionMatrix());
            sp.instance_buf_hdl  = gpu.instance_buf.hdl;
            sp.primitive_buf_hdl = gpu.primitive_buf.hdl;
            sp.index_buf_hdl     = gpu.index_buf.hdl;
            sp.position_buf_hdl  = gpu.position_buf.hdl;
            sp.draw_cmd_buf_hdl  = gpu.draw_cmd_alpha_blend_buf.hdl;
            sp.screen_width      = res.x;
            sp.screen_height     = res.y;
            sp.tile_count_x      = tile_cx;
            sp.tile_count_y      = tile_cy;
            sp.num_draw_cmds     = num_draw_cmds;
            sp.total_triangles   = total_triangles;

            context.cmd_list
                .Compute(m_setup_pipeline, m_triangle_buf, m_debug_stats_buf, context.bdls, sp)
                .Dispatch(uint3(setup_groups, 1, 1), "SoftRaster Setup");
        }
        context.cmd_list.PopScope();

        // ==================================================================
        // 3. BinCounter: count tile overlaps per triangle
        // ==================================================================
        SoftRasterTileWriteParam twp{};
        twp.tile_count_x    = tile_cx;
        twp.tile_count_y    = tile_cy;
        twp.total_triangles = total_triangles;

        context.cmd_list.PushScope("SoftRaster BinCounter");
        {
            context.cmd_list
                .Compute(m_bin_counter_pipeline, m_triangle_buf, m_tile_count_buf, m_debug_stats_buf, twp)
                .Dispatch(uint3(setup_groups, 1, 1), "SoftRaster Bin Counter");
        }
        context.cmd_list.PopScope();

        // ==================================================================
        // 4. BinPrefix: allocate contiguous per-tile ranges
        // ==================================================================
        context.cmd_list.PushScope("SoftRaster BinPrefix");
        {
            SoftRasterTileAllocParam tap{};
            tap.total_tiles      = tile_cnt;
            tap.max_tile_entries = static_cast<uint>(m_tile_tri_buf->GetNumElement());

            context.cmd_list
                .Compute(
                    m_bin_prefix_pipeline,
                    m_tile_count_buf,
                    m_tile_offset_buf,
                    m_tile_write_cursor_buf,
                    m_debug_stats_buf,
                    tap
                )
                .Dispatch(uint3(1, 1, 1), "SoftRaster Bin Prefix");
        }
        context.cmd_list.PopScope();

        // ==================================================================
        // 5. BinDispatch: write triangle IDs into tile ranges
        // ==================================================================
        context.cmd_list.PushScope("SoftRaster BinDispatch");
        {
            context.cmd_list
                .Compute(
                    m_bin_dispatch_pipeline,
                    m_triangle_buf,
                    m_tile_count_buf,
                    m_tile_offset_buf,
                    m_tile_write_cursor_buf,
                    m_tile_tri_buf,
                    m_debug_stats_buf,
                    twp
                )
                .Dispatch(uint3(setup_groups, 1, 1), "SoftRaster Bin Dispatch");
        }
        context.cmd_list.PopScope();

        // ==================================================================
        // 6. BinCategorize: split tiles into low/high density bins
        // ==================================================================
        context.cmd_list.PushScope("SoftRaster BinCategorize");
        {
            SoftRasterBinCategorizeParam bcp{};
            bcp.total_tiles         = tile_cnt;
            bcp.high_tri_threshold  = SOFT_RASTER_BIN_HIGH_TRI_THRESHOLD;
            const uint categorize_groups = (tile_cnt + SOFT_RASTER_ALLOC_WG_SIZE - 1) / SOFT_RASTER_ALLOC_WG_SIZE;

            context.cmd_list
                .Compute(
                    m_bin_categorize_pipeline,
                    m_tile_count_buf,
                    m_low_bin_list_buf,
                    m_high_bin_list_buf,
                    m_low_bin_count_buf,
                    m_high_bin_count_buf,
                    bcp
                )
                .Dispatch(uint3(categorize_groups, 1, 1), "SoftRaster Bin Categorize");
        }
        context.cmd_list.PopScope();

        // ==================================================================
        // 7. Fine Raster — COUNT (Low bin)
        // ==================================================================
        SoftRasterTileParam tp{};
        tp.screen_width  = res.x;
        tp.screen_height = res.y;
        tp.tile_count_x  = tile_cx;
        tp.tile_count_y  = tile_cy;
        tp.instance_buf_hdl       = gpu.instance_buf.hdl;
        tp.primitive_buf_hdl      = gpu.primitive_buf.hdl;
        tp.index_buf_hdl          = gpu.index_buf.hdl;
        tp.position_buf_hdl       = gpu.position_buf.hdl;
        tp.packed_normal_buf_hdl  = gpu.packed_normal_buf.hdl;
        tp.packed_tangent_buf_hdl = gpu.packed_tangent_buf.hdl;
        tp.texcoord0_buf_hdl      = gpu.texcoord0_buf.hdl;
        tp.material_buf_hdl       = gpu.material_buf.hdl;
        tp.light_buf_hdl          = gpu.light_buf.hdl;
        tp.global_param_handle    = context.lighting_data_buffer.hdl;
        tp.enable_extra_ambient    = ui_config.shading_enable_extra_ambient ? 1u : 0u;
        tp._pad0                   = 0u;
        tp.extra_ambient_color     = ui_config.shading_extra_ambient_color;
        tp.extra_ambient_intensity = ui_config.shading_extra_ambient_intensity;

        context.cmd_list.PushScope("SoftRaster FineCount Low");
        {
            context.cmd_list
                .Compute(
                    m_fine_count_pipeline,
                    m_triangle_buf,
                    m_tile_count_buf,
                    m_tile_tri_buf,
                    m_tile_offset_buf,
                    m_low_bin_list_buf,
                    m_low_bin_count_buf,
                    m_pixel_frag_count_buf,
                    m_debug_stats_buf,
                    tp
                )
                .Dispatch(uint3(tile_cnt, 1, 1), "SoftRaster Fine Count Low");
        }
        context.cmd_list.PopScope();

        // ==================================================================
        // 8. Fine Raster — COUNT (High bin)
        // ==================================================================
        context.cmd_list.PushScope("SoftRaster FineCount High");
        {
            context.cmd_list
                .Compute(
                    m_fine_count_pipeline,
                    m_triangle_buf,
                    m_tile_count_buf,
                    m_tile_tri_buf,
                    m_tile_offset_buf,
                    m_high_bin_list_buf,
                    m_high_bin_count_buf,
                    m_pixel_frag_count_buf,
                    m_debug_stats_buf,
                    tp
                )
                .Dispatch(uint3(tile_cnt, 1, 1), "SoftRaster Fine Count High");
        }
        context.cmd_list.PopScope();

        // ==================================================================
        // 9. Alloc — per-pixel contiguous reservation
        // ==================================================================
        context.cmd_list.PushScope("SoftRaster Alloc");
        {
            SoftRasterAllocParam ap{};
            ap.total_pixels  = total_px;
            ap.max_fragments = m_max_fragments;

            const uint groups = (total_px + SOFT_RASTER_ALLOC_WG_SIZE - 1) / SOFT_RASTER_ALLOC_WG_SIZE;
            context.cmd_list
                .Compute(
                    m_alloc_pipeline,
                    m_pixel_frag_count_buf,
                    m_pixel_frag_offset_buf,
                    m_alloc_counter_buf,
                    m_debug_stats_buf,
                    ap
                )
                .Dispatch(uint3(groups, 1, 1), "SoftRaster Alloc");
        }
        context.cmd_list.PopScope();

        // ==================================================================
        // 10. Clear pixel_frag_count for reuse as per-pixel write counter
        // ==================================================================
        context.cmd_list.PushScope("SoftRaster ClearCount");
        context.cmd_list.ClearResource(m_pixel_frag_count_buf->GetView(), (uint32_t)0);
        context.cmd_list.PopScope();

        // ==================================================================
        // 11. Fine Raster — WRITE (Low bin)
        // ==================================================================
        context.cmd_list.PushScope("SoftRaster FineWrite Low");
        {
            context.cmd_list
                .Compute(
                    fine_write_pipeline,
                    m_triangle_buf,
                    m_tile_count_buf,
                    m_tile_tri_buf,
                    m_tile_offset_buf,
                    m_low_bin_list_buf,
                    m_low_bin_count_buf,
                    m_pixel_frag_count_buf,
                    m_pixel_frag_offset_buf,
                    m_fragment_buf,
                    m_fragment_shade_buf,
                    m_debug_stats_buf,
                    context.bdls,
                    tp
                )
                .Dispatch(uint3(tile_cnt, 1, 1), "SoftRaster Fine Write Low");
        }
        context.cmd_list.PopScope();

        // ==================================================================
        // 12. Fine Raster — WRITE (High bin)
        // ==================================================================
        context.cmd_list.PushScope("SoftRaster FineWrite High");
        {
            context.cmd_list
                .Compute(
                    fine_write_pipeline,
                    m_triangle_buf,
                    m_tile_count_buf,
                    m_tile_tri_buf,
                    m_tile_offset_buf,
                    m_high_bin_list_buf,
                    m_high_bin_count_buf,
                    m_pixel_frag_count_buf,
                    m_pixel_frag_offset_buf,
                    m_fragment_buf,
                    m_fragment_shade_buf,
                    m_debug_stats_buf,
                    context.bdls,
                    tp
                )
                .Dispatch(uint3(tile_cnt, 1, 1), "SoftRaster Fine Write High");
        }
        context.cmd_list.PopScope();

        // ==================================================================
        // 13. Sort — per-pixel insertion sort + active-pixel mapping
        // ==================================================================
        context.cmd_list.PushScope("SoftRaster Sort");
        {
            SoftRasterSortParam sp{};
            sp.screen_width = res.x;
            sp.total_pixels = total_px;

            const uint groups = (total_px + SOFT_RASTER_SORT_WG_SIZE - 1) / SOFT_RASTER_SORT_WG_SIZE;
            context.cmd_list
                .Compute(
                    sort_pipeline,
                    m_fragment_buf,
                    m_fragment_shade_buf,
                    m_pixel_frag_count_buf,
                    m_pixel_frag_offset_buf,
                    m_active_pixel_buf,
                    m_active_pixel_count_buf,
                    sp
                )
                .Dispatch(uint3(groups, 1, 1), "SoftRaster Sort");
        }
        context.cmd_list.PopScope();

        // ==================================================================
        // 14. Shading — VB: reconstruct+shade, Forward: direct blend
        //    (CopyFrom lighting_output→resolve was moved to stage 0)
        // ==================================================================
        context.cmd_list.PushScope("SoftRaster Shade");
        {
            AOITResolveParam rp{};
            rp.clip2world = Transpose(camera.GetViewProjectionMatrixInv());

            rp.instance_buf_hdl       = gpu.instance_buf.hdl;
            rp.primitive_buf_hdl      = gpu.primitive_buf.hdl;
            rp.index_buf_hdl          = gpu.index_buf.hdl;
            rp.position_buf_hdl       = gpu.position_buf.hdl;
            rp.packed_normal_buf_hdl  = gpu.packed_normal_buf.hdl;
            rp.packed_tangent_buf_hdl = gpu.packed_tangent_buf.hdl;
            rp.texcoord0_buf_hdl      = gpu.texcoord0_buf.hdl;
            rp.material_buf_hdl       = gpu.material_buf.hdl;

            rp.light_buf_hdl       = gpu.light_buf.hdl;
            rp.global_param_handle = context.lighting_data_buffer.hdl;

            rp.enable_extra_ambient    = ui_config.shading_enable_extra_ambient ? 1u : 0u;
            rp.extra_ambient_color     = ui_config.shading_extra_ambient_color;
            rp.extra_ambient_intensity = ui_config.shading_extra_ambient_intensity;

            // Dispatch one thread per active pixel (worst-case = total_px)
            const uint shading_wg     = 256u;
            const uint shading_groups = (total_px + shading_wg - 1) / shading_wg;
            context.cmd_list
                .Compute(
                    shading_pipeline,
                    m_fragment_buf,
                    m_fragment_shade_buf,
                    m_active_pixel_buf,
                    m_active_pixel_count_buf,
                    m_resolve_output_tex,
                    context.bdls,
                    rp
                )
                .Dispatch(uint3(shading_groups, 1, 1), "SoftRaster Shading");
        }
        context.cmd_list.PopScope();

        if (ui_config.soft_raster_oit_debug_stats) {
            context.cmd_list.PushScope("SoftRaster StatsReadback");
            context.cmd_list.CopyFrom(
                m_debug_stats_buf->GetView(),
                std::span<byte>(
                    reinterpret_cast<byte*>(m_debug_stats_readback.data()),
                    sizeof(uint32_t) * m_debug_stats_readback.size()
                ),
                "SoftRaster DebugStats Readback"
            );
            context.cmd_list.PopScope();
            m_has_debug_stats_readback = true;
        }

        // Keep resolve copy in a dedicated scope to force an extra reorderer layer/barrier.
        context.cmd_list.PushScope("SoftRaster ResolveCopy");
        context.cmd_list.CopyFrom(
            m_resolve_output_tex->GetView(),
            context.textures.lighting_output.tex->GetView(),
            "SoftRaster Resolve To Lighting"
        );
        context.cmd_list.PopScope();

        context.cmd_list.PopScope();
    }

    // ---- Accessors for downstream shading pass ----
    BufferRef GetFragmentBuf() const {
        return m_fragment_buf;
    }
    BufferRef GetActivePixelBuf() const {
        return m_active_pixel_buf;
    }
    BufferRef GetActivePixelCountBuf() const {
        return m_active_pixel_count_buf;
    }
    BufferRef GetPixelFragCountBuf() const {
        return m_pixel_frag_count_buf;
    }
    BufferRef GetPixelFragOffsetBuf() const {
        return m_pixel_frag_offset_buf;
    }

private:
    void SyncCapacityFromConfig(RasterContext& context, const RasterConfig& ui_config) {
        uint new_max_triangles = std::max(ui_config.soft_raster_oit_max_triangles, 1u);
        uint new_max_fragments = std::max(ui_config.soft_raster_oit_max_fragments, 1u);

        if (new_max_triangles == m_max_triangles && new_max_fragments == m_max_fragments) {
            return;
        }

        m_max_triangles = new_max_triangles;
        m_max_fragments = new_max_fragments;
        AllocateBuffers(context);

        LOG_INFO(
            "SoftRaster OIT capacity updated: max_triangles={}, max_fragments={}",
            m_max_triangles,
            m_max_fragments
        );
    }

    void LogDebugStatsIfNeeded(const RasterConfig& ui_config) {
        if (!ui_config.soft_raster_oit_debug_stats || !m_has_debug_stats_readback) {
            return;
        }

        uint log_interval = std::max(ui_config.soft_raster_oit_debug_log_interval, 1u);
        if ((m_debug_stats_frame_id++ % log_interval) != 0u) {
            return;
        }

        uint tile_tri_overflow    = m_debug_stats_readback[SOFT_RASTER_STAT_TILE_TRI_OVERFLOW];
        uint pixel_clamped        = m_debug_stats_readback[SOFT_RASTER_STAT_PIXEL_CLAMPED];
        uint pool_overflow        = m_debug_stats_readback[SOFT_RASTER_STAT_POOL_OVERFLOW];
        uint pixel_write_overflow = m_debug_stats_readback[SOFT_RASTER_STAT_PIXEL_WRITE_OVERFLOW];
        uint max_tile_tri_count   = m_debug_stats_readback[SOFT_RASTER_STAT_MAX_TILE_TRI_COUNT];
        uint total_tile_entries   = m_debug_stats_readback[SOFT_RASTER_STAT_TOTAL_TILE_ENTRIES];
        uint global_tile_capacity = m_tile_tri_buf->GetNumElement();
        float avg_tiles_per_tri = m_last_expanded_triangles == 0u ?
            0.0f :
            (float)total_tile_entries / (float)m_last_expanded_triangles;

        if (tile_tri_overflow == 0u && pixel_clamped == 0u && pool_overflow == 0u && pixel_write_overflow == 0u) {
            return;
        }

        LOG_WARNING(
            "SoftRaster OIT overflow stats: tile_tri_overflow={}, pixel_clamped={}, pool_overflow={}, pixel_write_overflow={}, max_tile_tri_count={}, total_tile_entries={}, global_tile_capacity={}, expanded_triangles={}, avg_tiles_per_tri={:.2f}",
            tile_tri_overflow,
            pixel_clamped,
            pool_overflow,
            pixel_write_overflow,
            max_tile_tri_count,
            total_tile_entries,
            global_tile_capacity,
            m_last_expanded_triangles,
            avg_tiles_per_tri
        );
    }

    void AllocateBuffers(RasterContext& context) {
        m_has_debug_stats_readback = false;
        m_debug_stats_readback.fill(0u);

        const uint2 res      = context.GetResolutionOriginal();
        const uint  total_px = res.x * res.y;
        const uint  tile_cx  = (res.x + SOFT_RASTER_TILE_SIZE - 1) / SOFT_RASTER_TILE_SIZE;
        const uint  tile_cy  = (res.y + SOFT_RASTER_TILE_SIZE - 1) / SOFT_RASTER_TILE_SIZE;
        const uint  tile_cnt = tile_cx * tile_cy;

        m_triangle_buf = context.device.CreateBuffer<uint4>(
            "SoftRaster::TriangleBuf",
            m_max_triangles * SOFT_RASTER_TRI_STRIDE,
            EBufferUsageFlags::UNORDERED_ACCESS
        );

        m_tile_count_buf = context.device.CreateBuffer<uint>(
            "SoftRaster::TileCount",
            tile_cnt,
            EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST
        );

        m_tile_offset_buf = context.device.CreateBuffer<uint>(
            "SoftRaster::TileOffset",
            tile_cnt,
            EBufferUsageFlags::UNORDERED_ACCESS
        );

        m_tile_write_cursor_buf = context.device.CreateBuffer<uint>(
            "SoftRaster::TileWriteCursor",
            tile_cnt,
            EBufferUsageFlags::UNORDERED_ACCESS
        );

        m_tile_tri_buf = context.device.CreateBuffer<uint>(
            "SoftRaster::TileTriBuf",
            tile_cnt * SOFT_RASTER_MAX_TRIS_PER_TILE,
            EBufferUsageFlags::UNORDERED_ACCESS
        );

        m_low_bin_list_buf = context.device.CreateBuffer<uint>(
            "SoftRaster::LowBinList",
            tile_cnt,
            EBufferUsageFlags::UNORDERED_ACCESS
        );

        m_high_bin_list_buf = context.device.CreateBuffer<uint>(
            "SoftRaster::HighBinList",
            tile_cnt,
            EBufferUsageFlags::UNORDERED_ACCESS
        );

        m_low_bin_count_buf = context.device.CreateBuffer<uint>(
            "SoftRaster::LowBinCount",
            1,
            EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST
        );

        m_high_bin_count_buf = context.device.CreateBuffer<uint>(
            "SoftRaster::HighBinCount",
            1,
            EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST
        );

        m_pixel_frag_count_buf = context.device.CreateBuffer<uint>(
            "SoftRaster::PixelFragCount",
            total_px,
            EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST
        );

        m_pixel_frag_offset_buf = context.device.CreateBuffer<uint>(
            "SoftRaster::PixelFragOffset", total_px, EBufferUsageFlags::UNORDERED_ACCESS
        );

        m_fragment_buf = context.device.CreateBuffer<uint2>(
            "SoftRaster::FragmentBuf", m_max_fragments, EBufferUsageFlags::UNORDERED_ACCESS
        );

        m_fragment_shade_buf = context.device.CreateBuffer<float4>(
            "SoftRaster::FragmentShadeBuf", m_max_fragments, EBufferUsageFlags::UNORDERED_ACCESS
        );

        m_active_pixel_buf = context.device.CreateBuffer<uint2>(
            "SoftRaster::ActivePixelBuf",
            total_px, // worst case
            EBufferUsageFlags::UNORDERED_ACCESS
        );

        m_active_pixel_count_buf = context.device.CreateBuffer<uint>(
            "SoftRaster::ActivePixelCount",
            1,
            EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST
        );

        m_alloc_counter_buf = context.device.CreateBuffer<uint>(
            "SoftRaster::AllocCounter",
            1,
            EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST
        );

        m_debug_stats_buf = context.device.CreateBuffer<uint>(
            "SoftRaster::DebugStats",
            SOFT_RASTER_STAT_COUNT,
            EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST
        );

        m_resolve_output_tex = context.device.CreateTexture(
            "SoftRaster::ResolveOutput",
            Extent2D(res.x, res.y),
            PF_R16G16B16A16_SFLOAT,
            ETextureUsageFlags::UNORDERED_ACCESS | ETextureUsageFlags::SAMPLED
        );
    }

    // Configuration
    uint m_max_triangles;
    uint m_max_fragments;

    // Pipelines
    SoftRasterSetupPipeline         m_setup_pipeline;
    SoftRasterBinCounterPipeline    m_bin_counter_pipeline;
    SoftRasterBinPrefixPipeline     m_bin_prefix_pipeline;
    SoftRasterBinDispatchPipeline   m_bin_dispatch_pipeline;
    SoftRasterBinCategorizePipeline m_bin_categorize_pipeline;
    SoftRasterFineCountPipeline     m_fine_count_pipeline;
    SoftRasterFineWritePipeline     m_fine_write_pipeline_vb;
    SoftRasterFineWritePipeline     m_fine_write_pipeline_forward;
    SoftRasterAllocPipeline         m_alloc_pipeline;
    SoftRasterSortPipeline          m_sort_pipeline_vb;
    SoftRasterSortPipeline          m_sort_pipeline_forward;
    SoftRasterShadingPipeline       m_shading_pipeline_vb;
    SoftRasterShadingPipeline       m_shading_pipeline_forward;

    // GPU Buffers
    BufferRef m_triangle_buf;
    BufferRef m_tile_count_buf;
    BufferRef m_tile_offset_buf;
    BufferRef m_tile_write_cursor_buf;
    BufferRef m_tile_tri_buf;
    BufferRef m_low_bin_list_buf;
    BufferRef m_high_bin_list_buf;
    BufferRef m_low_bin_count_buf;
    BufferRef m_high_bin_count_buf;
    BufferRef m_pixel_frag_count_buf;
    BufferRef m_pixel_frag_offset_buf;
    BufferRef m_fragment_buf;
    BufferRef m_fragment_shade_buf;
    BufferRef m_active_pixel_buf;
    BufferRef m_active_pixel_count_buf;
    BufferRef m_alloc_counter_buf;
    BufferRef m_debug_stats_buf;

    // Resolve output texture (same format as lighting_output, UAV for compute)
    TextureRef m_resolve_output_tex;

    // CPU readback for overflow statistics (updated asynchronously by CopyBackBuffer)
    std::array<uint32_t, SOFT_RASTER_STAT_COUNT> m_debug_stats_readback{};
    bool                                         m_has_debug_stats_readback = false;
    uint                                         m_debug_stats_frame_id     = 0;
    uint                                         m_triangle_clamp_log_frame_id = 0;
    uint                                         m_last_expanded_triangles     = 0;
};

} // namespace Moer::Render::Raster
