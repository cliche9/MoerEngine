#pragma once

#include <filesystem>

namespace Moer {
class SceneImportOptions;
namespace ecs {
class LogicalScene;
}
} // namespace Moer

namespace Moer::gaussian {
class Parser {
public:
    Parser()  = default;
    ~Parser() = default;

    static bool LoadSceneFromFile(
        ecs::LogicalScene&           out_logical_scene,
        const std::filesystem::path& file_path,
        const SceneImportOptions&    import_options
    );
};
} // namespace Moer::gaussian
