#pragma once

#include "scene/camera/Camera.h"
#include "shader/ShaderCommon.h"
#include "shader/ShaderMutation.h"
#include "shader/ShaderPipeline.h"
#include "shaderheaders/shared/raster/aoit/AOITData.h"
#include "shaderheaders/shared/raster/geometry_pass/ShaderParameters.h"

#include "RasterConfig.h"
#include "RasterResource.h"

namespace Moer::Render::Raster {

// ============================================================================
// Pipeline Definitions
// ============================================================================

/**
 * Raster pipeline for collecting transparent fragments into a per-pixel linked list.
 * head_pointer_tex = RWTexture2D<uint> (per-pixel linked list head)
 * counter_buf      = RWBuffer<uint>    (atomic allocation counter, 1 element)
 * fragment_pool_buf= RWBuffer<uint4>   (fragment pool)
 */
class AOITCollectPipeline : public RasterPipeline {
public:
    DEFINE_RASTER_PIPELINE_CLASS(AOITCollectPipeline);
    DEFINE_SHADER_TEX(head_pointer_tex);
    DEFINE_SHADER_BUFFER(counter_buf);
    DEFINE_SHADER_BUFFER(fragment_pool_buf);
    DEFINE_SHADER_BINDLESS_ARRAY(bdls);
    DEFINE_SHADER_CONSTANT_STRUCT(GeometryPassBindlessParam, param);
    DEFINE_SHADER_ARGS(head_pointer_tex, counter_buf, fragment_pool_buf, bdls, param);

    MUTATION_BOOL(SHADOW_DEPTH_PASS);
    MUTATION_SET(MutationSet, SHADOW_DEPTH_PASS);
};

/**
 * Compute pipeline for resolving the per-pixel linked list:
 * sort fragments by depth and composite over the opaque scene color.
 */
class AOITResolvePipeline : public ComputePipeline {
public:
    DEFINE_COMPUTE_PIPELINE_CLASS(AOITResolvePipeline);
    DEFINE_SHADER_TEX(head_pointer_tex);
    DEFINE_SHADER_BUFFER(counter_buf);
    DEFINE_SHADER_BUFFER(fragment_pool_buf);
    DEFINE_SHADER_TEX(output_image);
    DEFINE_SHADER_BINDLESS_ARRAY(bdls);
    DEFINE_SHADER_ARGS(head_pointer_tex, counter_buf, fragment_pool_buf, output_image, bdls);
};

// ============================================================================
// AOITPass
// ============================================================================

class AOITPass {
public:
    explicit AOITPass(RasterContext& context, uint max_fragments = 1u << 20) :
        m_pool_allocated_size(max_fragments) {

        // ---- Collect raster pipeline ----
        {
            GfxPsoCreateInfo pso_info(
                RHIRasterizeInfo::Preset(), // cull none (default)
                {},                         // vertex stream (bindless)
                {RHIColorAttachmentInfo::Preset<Blend::ALPHA_BLEND>(
                    context.textures.lighting_output.tex->GetFormat()
                )},
                RHIDepthStencilStateInfo(false, CO_GREATER), // depth test, no write, reversed-Z
                context.textures.depth_linear_sampler.tex->GetFormat()
            );

            AOITCollectPipeline::MutationSet mutation_set{};
            mutation_set.SetMutation<AOITCollectPipeline::SHADOW_DEPTH_PASS>(false);

            Shader& vtx = ShaderManager::Get().CompileShader(
                ST_VERTEX, "pipelines/raster/deferred/geometry/GeometryPassVertex.hlsl", mutation_set
            );
            Shader& frag = ShaderManager::Get().CompileShader(
                ST_FRAGMENT, "pipelines/raster/aoit/AOITCollect.hlsl", mutation_set
            );

            m_collect_pso = ShaderManager::Get().Raster().Vertex(vtx).Pixel(frag).Build<AOITCollectPipeline>(
                std::move(pso_info)
            );
        }

        // ---- Resolve compute pipeline ----
        m_resolve_pipeline =
            context.manager.Compute<AOITResolvePipeline>("pipelines/raster/aoit/AOITResolve.hlsl");

        // ---- Allocate GPU resources ----
        uint2 res = context.GetResolutionOriginal();

        // Head pointer texture: R32_UINT, one texel per pixel
        m_head_pointer_tex = context.device.CreateTexture(
            "AOIT::HeadPointers",
            Extent2D(res.x, res.y),
            PF_R32_UINT,
            ETextureUsageFlags::UNORDERED_ACCESS | ETextureUsageFlags::TRANSFER_DST
        );

        // Counter buffer: [0]=atomic counter, [1]=pool capacity
        // Both set every frame via ClearResource (vkCmdFillBuffer) - no CopyFrom needed
        m_counter_buf = context.device.CreateBuffer<uint>(
            "AOIT::Counter", 2, EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST
        );

        // Fragment pool: each fragment = uint4 (next, depth, packed_rg, packed_ba)
        m_fragment_pool_buf = context.device.CreateBuffer<uint4>(
            "AOIT::FragmentPool", max_fragments, EBufferUsageFlags::UNORDERED_ACCESS
        );

        // Resolve output (same format as lighting_output, with UAV for compute read/write)
        m_resolve_output_tex = context.device.CreateTexture(
            "AOIT::ResolveOutput",
            Extent2D(res.x, res.y),
            PF_R16G16B16A16_SFLOAT,
            ETextureUsageFlags::UNORDERED_ACCESS | ETextureUsageFlags::SAMPLED
        );
    }

    /// Recreate resolution-dependent textures after window resize.
    void OnResize(RasterContext& context) {
        uint2 res = context.GetResolutionOriginal();

        m_head_pointer_tex = context.device.CreateTexture(
            "AOIT::HeadPointers",
            Extent2D(res.x, res.y),
            PF_R32_UINT,
            ETextureUsageFlags::UNORDERED_ACCESS | ETextureUsageFlags::TRANSFER_DST
        );

        m_resolve_output_tex = context.device.CreateTexture(
            "AOIT::ResolveOutput",
            Extent2D(res.x, res.y),
            PF_R16G16B16A16_SFLOAT,
            ETextureUsageFlags::UNORDERED_ACCESS | ETextureUsageFlags::SAMPLED
        );
    }

    void Process(RasterContext& context, const RasterConfig& ui_config, const Camera& camera) {
        const auto& gpu_scene_res = context.scene.gpu_scene_res();

        if (gpu_scene_res.draw_cmd_alpha_blend_count == 0) {
            return;
        }

        // ====================================================================
        // 1. Clear: reset head pointers and atomic counter
        // ====================================================================
        context.cmd_list.ClearResource(m_head_pointer_tex->GetView(), (uint32_t)AOIT_INVALID_POINTER);
        // counter_buf[0] = 0 (atomic counter), counter_buf[1] = pool capacity (clamped to actual buffer)
        uint effective_max = std::min(ui_config.aoit_max_fragments, m_pool_allocated_size);
        context.cmd_list.ClearResource(m_counter_buf->GetView(0, sizeof(uint)), (uint32_t)0);
        context.cmd_list.ClearResource(
            m_counter_buf->GetView(sizeof(uint), sizeof(uint)), (uint32_t)effective_max
        );

        // ====================================================================
        // 2. Collect: rasterize transparent geometry, insert into linked list
        // ====================================================================
        GeometryPassBindlessParam param;
        param.world2clip = Transpose(camera.GetViewProjectionMatrix());

        param.instance_buf_hdl       = gpu_scene_res.instance_buf.hdl;
        param.primitive_buf_hdl      = gpu_scene_res.primitive_buf.hdl;
        param.position_buf_hdl       = gpu_scene_res.position_buf.hdl;
        param.packed_normal_buf_hdl  = gpu_scene_res.packed_normal_buf.hdl;
        param.packed_tangent_buf_hdl = gpu_scene_res.packed_tangent_buf.hdl;
        param.texcoord0_buf_hdl      = gpu_scene_res.texcoord0_buf.hdl;
        param.material_buf_hdl       = gpu_scene_res.material_buf.hdl;

        param.enable_alpha_test             = ui_config.geometry_enable_alpha_test ? 1 : 0;
        param.alpha_test_blend_pixel_cutoff = ui_config.geometry_alpha_test_blend_pixel_cutoff;
        param.light_buf_hdl                 = gpu_scene_res.light_buf.hdl;
        param.global_param_handle           = context.lighting_data_buffer.hdl;
        param.extra_ambient_color           = ui_config.shading_extra_ambient_color;
        param.extra_ambient_intensity       = ui_config.shading_extra_ambient_intensity;
        param.enable_extra_ambient          = ui_config.shading_enable_extra_ambient ? 1u : 0u;

        auto rect2d = context.textures.lighting_output.GetRect2D();

        context.cmd_list
            .Gfx(m_collect_pso, m_head_pointer_tex, m_counter_buf, m_fragment_pool_buf, context.bdls, param)
            .DrawIndirect(
                "AOIT Collect",
                rect2d,
                {},
                IndexBuffer{gpu_scene_res.index_buf.buf->GetView(), EIndexElementType::IET_UINT32},
                gpu_scene_res.draw_cmd_alpha_blend_buf.buf->GetView(),
                gpu_scene_res.draw_cmd_alpha_blend_count,
                gpu_scene_res.draw_cmd_alpha_blend_buf.buf->GetStride(),
                [&]() {
                    DepthAttachment depth_attachment(
                        context.textures.depth_linear_sampler.tex->GetView().GetTexture()
                    );
                    depth_attachment.action = AC_DS_LOAD_STORE;
                    return depth_attachment;
                }(),
                [&]() {
                    ColorAttachment color_attachment{context.textures.lighting_output.tex};
                    color_attachment.action = AC_LOAD_STORE;
                    return color_attachment;
                }()
            );

        // ====================================================================
        // 3. Resolve: sort & composite fragments over opaque scene color
        // ====================================================================

        // Copy opaque lighting_output -> resolve_output (so compute can read-modify-write)
        context.cmd_list.CopyFrom(
            context.textures.lighting_output.tex->GetView(),
            m_resolve_output_tex->GetView(),
            "AOIT Copy Opaque To Resolve"
        );

        uint2 res = context.GetResolutionOriginal();
        context.cmd_list
            .Compute(
                m_resolve_pipeline,
                m_head_pointer_tex,
                m_counter_buf,
                m_fragment_pool_buf,
                m_resolve_output_tex,
                context.bdls
            )
            .Dispatch(uint3((res.x + 7u) / 8u, (res.y + 7u) / 8u, 1u), "AOIT Resolve");

        // Copy the composited result back to lighting_output
        context.cmd_list.CopyFrom(
            m_resolve_output_tex->GetView(),
            context.textures.lighting_output.tex->GetView(),
            "AOIT Resolve To Lighting"
        );
    }

private:
    uint m_pool_allocated_size; // actual GPU buffer element count (fixed at construction)

    AOITCollectPipeline m_collect_pso;
    AOITResolvePipeline m_resolve_pipeline;

    TextureRef m_head_pointer_tex;
    BufferRef  m_counter_buf;
    BufferRef  m_fragment_pool_buf;
    TextureRef m_resolve_output_tex;
};

} // namespace Moer::Render::Raster
