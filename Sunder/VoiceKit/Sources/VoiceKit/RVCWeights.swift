import Foundation
import MLX
import MLXNN

// Loading/saving RVC weights as MLX-layout safetensors (see
// tools/convert/export_rvc_base.py for the layout).

enum RVCWeights {
    /// Loads `url` into `model`, requiring every model parameter to be present
    /// with a matching shape. Keys in `extras` are returned instead of applied.
    /// Everything is upcast to float32 (the downloaded base weights are float16).
    @discardableResult
    static func load(_ model: Module, from url: URL, extras: Set<String> = [],
                     verify: Module.VerifyUpdate = .all) throws -> [String: MLXArray] {
        var arrays = try loadArrays(url: url).mapValues { $0.asType(.float32) }
        var extra: [String: MLXArray] = [:]
        for key in extras { extra[key] = arrays.removeValue(forKey: key) }
        try model.update(parameters: ModuleParameters.unflattened(arrays), verify: verify)
        eval(model)
        return extra
    }

    static func save(_ model: Module, to url: URL, extras: [String: MLXArray] = [:], dtype: DType = .float32) throws {
        var arrays = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) })
        arrays.merge(extras) { $1 }
        try MLX.save(arrays: arrays, url: url)
    }
}
