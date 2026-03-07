#include "Parser.h"

#include "log/LogSystem.h"
#include "math/Function.h"
#include "misc/STL.h"
#include "scene/LogicalScene.h"
#include "scene/loader/LoaderInterface.h"

#include <entt/entt.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string_view>

namespace Moer::gaussian {

namespace {

enum class EPlyScalarType : uint8 {
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Float32,
    Float64,
    Invalid
};

struct PlyProperty {
    EPlyScalarType type   = EPlyScalarType::Invalid;
    std::string    name;
    size_t         offset = 0;
};

struct PlyHeader {
    std::string        format;
    uint32             vertex_count = 0;
    Array<PlyProperty> vertex_properties;
    size_t             vertex_stride = 0;
};

EPlyScalarType ParseScalarType(std::string_view token) {
    if (token == "char" || token == "int8") {
        return EPlyScalarType::Int8;
    }
    if (token == "uchar" || token == "uint8") {
        return EPlyScalarType::Uint8;
    }
    if (token == "short" || token == "int16") {
        return EPlyScalarType::Int16;
    }
    if (token == "ushort" || token == "uint16") {
        return EPlyScalarType::Uint16;
    }
    if (token == "int" || token == "int32") {
        return EPlyScalarType::Int32;
    }
    if (token == "uint" || token == "uint32") {
        return EPlyScalarType::Uint32;
    }
    if (token == "float" || token == "float32") {
        return EPlyScalarType::Float32;
    }
    if (token == "double" || token == "float64") {
        return EPlyScalarType::Float64;
    }

    return EPlyScalarType::Invalid;
}

size_t GetScalarTypeSize(EPlyScalarType type) {
    switch (type) {
        case EPlyScalarType::Int8:
        case EPlyScalarType::Uint8:
            return 1;
        case EPlyScalarType::Int16:
        case EPlyScalarType::Uint16:
            return 2;
        case EPlyScalarType::Int32:
        case EPlyScalarType::Uint32:
        case EPlyScalarType::Float32:
            return 4;
        case EPlyScalarType::Float64:
            return 8;
        default:
            return 0;
    }
}

template<typename T>
T ReadScalar(const char* data, size_t offset) {
    T value{};
    std::memcpy(&value, data + offset, sizeof(T));
    return value;
}

float ReadScalarAsFloat(const char* record, const PlyProperty& property) {
    switch (property.type) {
        case EPlyScalarType::Int8:
            return static_cast<float>(ReadScalar<int8_t>(record, property.offset));
        case EPlyScalarType::Uint8:
            return static_cast<float>(ReadScalar<uint8_t>(record, property.offset));
        case EPlyScalarType::Int16:
            return static_cast<float>(ReadScalar<int16_t>(record, property.offset));
        case EPlyScalarType::Uint16:
            return static_cast<float>(ReadScalar<uint16_t>(record, property.offset));
        case EPlyScalarType::Int32:
            return static_cast<float>(ReadScalar<int32_t>(record, property.offset));
        case EPlyScalarType::Uint32:
            return static_cast<float>(ReadScalar<uint32_t>(record, property.offset));
        case EPlyScalarType::Float32:
            return ReadScalar<float>(record, property.offset);
        case EPlyScalarType::Float64:
            return static_cast<float>(ReadScalar<double>(record, property.offset));
        default:
            return 0.0f;
    }
}

bool ParseIndexedProperty(std::string_view name, std::string_view prefix, uint32& out_index) {
    if (!name.starts_with(prefix)) {
        return false;
    }

    const std::string_view index_sv = name.substr(prefix.size());
    if (index_sv.empty()) {
        return false;
    }

    uint32 index = 0;
    for (char ch : index_sv) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        index = index * 10 + static_cast<uint32>(ch - '0');
    }

    out_index = index;
    return true;
}

bool ParsePlyHeader(std::ifstream& file, PlyHeader& out_header) {
    std::string line;
    bool        in_vertex_element = false;

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line == "end_header") {
            return true;
        }

        std::istringstream iss(line);
        std::string        token;
        iss >> token;

        if (token == "ply") {
            continue;
        }

        if (token == "format") {
            iss >> out_header.format;
            continue;
        }

        if (token == "element") {
            std::string element_name;
            uint32      element_count = 0;
            iss >> element_name >> element_count;

            in_vertex_element = (element_name == "vertex");
            if (in_vertex_element) {
                out_header.vertex_count = element_count;
            }
            continue;
        }

        if (token == "property" && in_vertex_element) {
            std::string type_name;
            iss >> type_name;

            if (type_name == "list") {
                continue;
            }

            std::string property_name;
            iss >> property_name;

            const EPlyScalarType scalar_type = ParseScalarType(type_name);
            if (scalar_type == EPlyScalarType::Invalid) {
                LOG_ERROR("Unsupported PLY scalar type: {}", type_name);
                return false;
            }

            out_header.vertex_properties.emplace_back(PlyProperty{
                .type   = scalar_type,
                .name   = std::move(property_name),
                .offset = out_header.vertex_stride
            });
            out_header.vertex_stride += GetScalarTypeSize(scalar_type);
        }
    }

    return false;
}

bool LoadGaussianSplatsFromPly(
    const std::filesystem::path& file_path,
    Array<GGaussianSplatVertex>& out_vertices
) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open 3DGS point cloud: {}", file_path.string());
        return false;
    }

    PlyHeader header;
    if (!ParsePlyHeader(file, header)) {
        LOG_ERROR("Failed to parse PLY header: {}", file_path.string());
        return false;
    }

    if (header.format != "binary_little_endian") {
        LOG_ERROR(
            "Unsupported 3DGS PLY format: {}. Only binary_little_endian is supported.",
            header.format
        );
        return false;
    }

    const PlyProperty* x_prop       = nullptr;
    const PlyProperty* y_prop       = nullptr;
    const PlyProperty* z_prop       = nullptr;
    const PlyProperty* opacity_prop = nullptr;

    std::array<const PlyProperty*, 3> scale_props{};
    std::array<const PlyProperty*, 4> rotation_props{};
    std::array<const PlyProperty*, 3> dc_props{};
    Array<const PlyProperty*>         rest_props;

    for (const auto& property : header.vertex_properties) {
        if (property.name == "x") {
            x_prop = &property;
            continue;
        }
        if (property.name == "y") {
            y_prop = &property;
            continue;
        }
        if (property.name == "z") {
            z_prop = &property;
            continue;
        }
        if (property.name == "opacity") {
            opacity_prop = &property;
            continue;
        }

        uint32 index = 0;
        if (ParseIndexedProperty(property.name, "scale_", index) && index < scale_props.size()) {
            scale_props[index] = &property;
            continue;
        }
        if (ParseIndexedProperty(property.name, "rot_", index) && index < rotation_props.size()) {
            rotation_props[index] = &property;
            continue;
        }
        if (ParseIndexedProperty(property.name, "f_dc_", index) && index < dc_props.size()) {
            dc_props[index] = &property;
            continue;
        }
        if (ParseIndexedProperty(property.name, "f_rest_", index)) {
            if (index >= rest_props.size()) {
                rest_props.resize(index + 1, nullptr);
            }
            rest_props[index] = &property;
        }
    }

    const bool has_required_props =
        x_prop && y_prop && z_prop && opacity_prop && scale_props[0] && scale_props[1] && scale_props[2] &&
        rotation_props[0] && rotation_props[1] && rotation_props[2] && rotation_props[3] && dc_props[0] &&
        dc_props[1] && dc_props[2];

    if (!has_required_props) {
        LOG_ERROR("PLY file is not a default 3DGS point cloud: {}", file_path.string());
        return false;
    }

    uint32 rest_property_count = 0;
    for (const auto* property : rest_props) {
        if (property != nullptr) {
            rest_property_count++;
        }
    }

    if (rest_property_count != 45) {
        LOG_WARNING(
            "3DGS PLY '{}' has {} f_rest_* properties. Default Graphdeco output usually has 45.",
            file_path.string(),
            rest_property_count
        );
    }

    out_vertices.clear();
    out_vertices.reserve(header.vertex_count);

    std::string record(header.vertex_stride, '\0');
    for (uint32 vertex_index = 0; vertex_index < header.vertex_count; ++vertex_index) {
        file.read(record.data(), static_cast<std::streamsize>(header.vertex_stride));
        if (file.gcount() != static_cast<std::streamsize>(header.vertex_stride)) {
            LOG_ERROR("Unexpected EOF while reading 3DGS vertex data: {}", file_path.string());
            return false;
        }

        const float3 position = float3(
            ReadScalarAsFloat(record.data(), *x_prop),
            ReadScalarAsFloat(record.data(), *y_prop),
            ReadScalarAsFloat(record.data(), *z_prop)
        );

        const float3 scale_log = float3(
            ReadScalarAsFloat(record.data(), *scale_props[0]),
            ReadScalarAsFloat(record.data(), *scale_props[1]),
            ReadScalarAsFloat(record.data(), *scale_props[2])
        );
        const float3 scale = Exp(scale_log);

        const float opacity = 1.0f / (1.0f + std::exp(-ReadScalarAsFloat(record.data(), *opacity_prop)));

        float4 rotation = float4(
            ReadScalarAsFloat(record.data(), *rotation_props[0]),
            ReadScalarAsFloat(record.data(), *rotation_props[1]),
            ReadScalarAsFloat(record.data(), *rotation_props[2]),
            ReadScalarAsFloat(record.data(), *rotation_props[3])
        );
        const float rotation_length = Lengthf(rotation);
        if (rotation_length > EPS) {
            rotation /= rotation_length;
        } else {
            rotation = float4(1.0f, 0.0f, 0.0f, 0.0f);
        }

        GGaussianSplatVertex vertex{};
        vertex.position      = float4(position, 1.0f);
        vertex.scale_opacity = float4(scale, opacity);
        vertex.rotation      = rotation;

        std::fill(std::begin(vertex.sh), std::end(vertex.sh), 0.0f);
        vertex.sh[0] = ReadScalarAsFloat(record.data(), *dc_props[0]);
        vertex.sh[1] = ReadScalarAsFloat(record.data(), *dc_props[1]);
        vertex.sh[2] = ReadScalarAsFloat(record.data(), *dc_props[2]);

        if (!rest_props.empty() && rest_property_count > 0 && rest_property_count % 3 == 0) {
            const uint32 coeffs_per_channel = rest_property_count / 3;
            const uint32 basis_count        = Min<uint32>(coeffs_per_channel + 1, 16);

            for (uint32 basis = 1; basis < basis_count; ++basis) {
                for (uint32 channel = 0; channel < 3; ++channel) {
                    const uint32 src_index = (basis - 1) + channel * coeffs_per_channel;
                    if (src_index >= rest_props.size() || rest_props[src_index] == nullptr) {
                        continue;
                    }
                    vertex.sh[basis * 3 + channel] =
                        ReadScalarAsFloat(record.data(), *rest_props[src_index]);
                }
            }
        }

        out_vertices.emplace_back(vertex);
    }

    return true;
}

} // namespace

bool Parser::LoadSceneFromFile(
    ecs::LogicalScene&           out_logical_scene,
    const std::filesystem::path& file_path,
    const SceneImportOptions&    import_options
) {
    (void)import_options;

    const auto canonical_path = std::filesystem::weakly_canonical(file_path);
    if (!std::filesystem::exists(canonical_path)) {
        LOG_ERROR("3DGS file does not exist: {}", canonical_path.string());
        return false;
    }

    Array<GGaussianSplatVertex> vertices;
    if (!LoadGaussianSplatsFromPly(canonical_path, vertices)) {
        return false;
    }

    auto& r = out_logical_scene.r();
    if (!r.ctx().contains<ecs::CtxMegaBuffers>()) {
        r.ctx().emplace<ecs::CtxMegaBuffers>();
    }

    entt::entity root_node_entt = entt::null;
    {
        auto root_view = r.view<ecs::CTagRootNode>();
        if (root_view.begin() != root_view.end()) {
            root_node_entt = *root_view.begin();
        } else {
            root_node_entt = r.create();
            r.emplace<ecs::CTagRootNode>(root_node_entt);
            r.emplace<ecs::CNode>(root_node_entt);
            r.emplace<ecs::CTransform>(root_node_entt);
        }
    }

    {
        auto meta_view = r.view<ecs::CSceneMetaData>();
        if (meta_view.begin() == meta_view.end()) {
            const entt::entity scene_meta_entt = r.create();
            auto&              scene_meta      = r.emplace<ecs::CSceneMetaData>(scene_meta_entt);
            scene_meta.root_node_entt          = root_node_entt;
            scene_meta.scene_path              = canonical_path.string();
        }
    }

    const entt::entity gaussian_asset_entt = r.create();
    {
        auto& c_name   = r.emplace<ecs::CName>(gaussian_asset_entt);
        auto& c_splats = r.emplace<ecs::CGaussianSplatting>(gaussian_asset_entt);

        c_name.name      = canonical_path.filename().string();
        c_splats.vertices = std::move(vertices);
    }

    const entt::entity gaussian_node_entt = r.create();
    {
        auto& root_node          = r.get<ecs::CNode>(root_node_entt);
        auto& gaussian_node      = r.emplace<ecs::CNode>(gaussian_node_entt);
        auto& gaussian_transform = r.emplace<ecs::CTransform>(gaussian_node_entt);
        auto& gaussian_renderable =
            r.emplace<ecs::CGaussianSplattingRenderable>(gaussian_node_entt);
        auto& gaussian_name = r.emplace<ecs::CName>(gaussian_node_entt);

        out_logical_scene.UEmplaceNodeToParent(root_node_entt, root_node, gaussian_node_entt, gaussian_node);

        gaussian_transform = ecs::CTransform{};
        gaussian_transform.is_dirty = true;
        gaussian_renderable.gaussian_splatting_entt = gaussian_asset_entt;
        gaussian_name.name                      = canonical_path.stem().string();
    }

    out_logical_scene.SUpdateAllNodeTransformAndAABB();

    LOG_INFO(
        "3DGS scene imported: {} splats from '{}'",
        r.get<ecs::CGaussianSplatting>(gaussian_asset_entt).vertices.size(),
        canonical_path.string()
    );

    return true;
}

} // namespace Moer::gaussian
