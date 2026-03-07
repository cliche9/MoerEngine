#pragma once

#include "scene/camera/Camera.h"
#include "shader/ShaderCommon.h"
#include "shader/ShaderMutation.h"
#include "shader/ShaderPipeline.h"
#include "shaderheaders/shared/raster/geometry_pass/ShaderParameters.h"

#include "RasterConfig.h"
#include "RasterResource.h"

namespace Moer::Render::Raster {

// ============================================================================
// Pipeline Definition
// ============================================================================

/**
 * Forward-shading raster pipeline for transparent (alpha-blend) geometry.
 * Draws all meshes, discards non-Blend materials in the pixel shader.
 * Output is composited onto lighting_output with hardware alpha blending.
 */
class TransparentBlendPipeline : public RasterPipeline {
public:
    DEFINE_RASTER_PIPELINE_CLASS(TransparentBlendPipeline);
    DEFINE_SHADER_BINDLESS_ARRAY(bdls);
    DEFINE_SHADER_CONSTANT_STRUCT(GeometryPassBindlessParam, param);
    DEFINE_SHADER_ARGS(bdls, param);

    MUTATION_BOOL(SHADOW_DEPTH_PASS);
    MUTATION_SET(MutationSet, SHADOW_DEPTH_PASS);
};

// ============================================================================
// TransparentBlendPass
// ============================================================================

class TransparentBlendPass {
public:
    explicit TransparentBlendPass(RasterContext& context) {

        GfxPsoCreateInfo pso_info(
            RHIRasterizeInfo::Preset(), // cull none (default)
            {},                         // vertex stream (bindless)
            {RHIColorAttachmentInfo::Preset<Blend::ALPHA_BLEND>(
                context.textures.lighting_output.tex->GetFormat()
            )},
            RHIDepthStencilStateInfo(false, CO_GREATER), // depth test, no write, reversed-Z
            context.textures.depth_linear_sampler.tex->GetFormat()
        );

        TransparentBlendPipeline::MutationSet mutation_set{};
        mutation_set.SetMutation<TransparentBlendPipeline::SHADOW_DEPTH_PASS>(false);

        Shader& vtx = ShaderManager::Get().CompileShader(
            ST_VERTEX, "pipelines/raster/deferred/geometry/GeometryPassVertex.hlsl", mutation_set
        );
        Shader& frag = ShaderManager::Get().CompileShader(
            ST_FRAGMENT, "pipelines/raster/deferred/geometry/TransparentBlendPixel.hlsl", mutation_set
        );

        m_pso = ShaderManager::Get().Raster().Vertex(vtx).Pixel(frag).Build<TransparentBlendPipeline>(
            std::move(pso_info)
        );
    }

    void Process(RasterContext& context, const RasterConfig& ui_config, const Camera& camera) {
        const auto& gpu_scene_res = context.scene.gpu_scene_res();

        if (!gpu_scene_res.draw_cmd_alpha_blend_buf.buf ||
            gpu_scene_res.draw_cmd_alpha_blend_buf.buf->GetNumElement() == 0) {
            return;
        }

        context.cmd_list.PushScopeWithTimeScope("Hardware Alpha Blend");

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

        // Forward-shading fields
        param.light_buf_hdl           = gpu_scene_res.light_buf.hdl;
        param.global_param_handle     = context.lighting_data_buffer.hdl;
        param.extra_ambient_color     = ui_config.shading_extra_ambient_color;
        param.extra_ambient_intensity = ui_config.shading_extra_ambient_intensity;
        param.enable_extra_ambient    = ui_config.shading_enable_extra_ambient ? 1u : 0u;

        auto rect2d = context.textures.lighting_output.GetRect2D();

        context.cmd_list.Gfx(m_pso, context.bdls, param)
            .DrawIndirect(
                "Transparent Blend Pass",
                rect2d,
                {},
                IndexBuffer{gpu_scene_res.index_buf.buf->GetView(), EIndexElementType::IET_UINT32},
                gpu_scene_res.draw_cmd_alpha_blend_buf.buf->GetView(),
                gpu_scene_res.draw_cmd_alpha_blend_buf.buf->GetNumElement(),
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

        context.cmd_list.PopScopeWithTimeScope();
    }

private:
    TransparentBlendPipeline m_pso;
};

} // namespace Moer::Render::Raster
