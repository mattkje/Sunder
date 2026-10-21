import Foundation
import MLX

/// RVC's retrieval index without faiss: k-means centroids of the training
/// voice's ContentVec frames. At conversion time each input frame is blended
/// toward the inverse-square-distance-weighted mean of its 8 nearest
/// centroids, pulling the phonetic features toward the target speaker.
struct VoiceIndex {
    let centroids: MLXArray // [K, 768] float32

    static let maxCentroids = 10000

    /// Mini-batch-free Lloyd's k-means on the GPU (RVC also k-means-reduces
    /// large feature sets to 10k centers before building its index).
    static func build(from frames: MLXArray, iterations: Int = 12) -> VoiceIndex {
        let x = frames.asType(.float32)
        let n = x.dim(0)
        guard n > maxCentroids else { return VoiceIndex(centroids: x) }
        let k = maxCentroids
        var centroids = x.take(MLXArray((0 ..< n).shuffled().prefix(k).map { Int32($0) }), axis: 0)
        let xNorm = sum(x * x, axis: 1, keepDims: true)
        let ids = MLXArray(0 ..< Int32(k)).expandedDimensions(axis: 0)
        for _ in 0 ..< iterations {
            // Chunked so the one-hot assignment stays ~300 MB at most.
            var sums = MLXArray.zeros([k, 768]), counts = MLXArray.zeros([k, 1])
            let cNorm = sum(centroids * centroids, axis: 1).expandedDimensions(axis: 0)
            for start in stride(from: 0, to: n, by: 8192) {
                let range = start ..< min(n, start + 8192)
                let chunk = x[range]
                let d = xNorm[range] - 2 * matmul(chunk, centroids.transposed()) + cNorm
                let oneHot = (argMin(d, axis: 1).expandedDimensions(axis: 1) .== ids).asType(.float32)
                sums = sums + matmul(oneHot.transposed(), chunk)
                counts = counts + sum(oneHot, axis: 0).expandedDimensions(axis: 1)
                eval(sums, counts)
            }
            centroids = which(counts .> 0, sums / maximum(counts, 1), centroids)
            eval(centroids)
        }
        return VoiceIndex(centroids: centroids)
    }

    /// feats: [T, 768] -> blended [T, 768]; `rate` 0 = off, 1 = fully retrieved.
    func blend(_ feats: MLXArray, rate: Float) -> MLXArray {
        guard rate > 0, centroids.dim(0) > 0 else { return feats }
        let k = min(8, centroids.dim(0))
        let cNorm = sum(centroids * centroids, axis: 1).expandedDimensions(axis: 0)
        var out: [MLXArray] = []
        for start in stride(from: 0, to: feats.dim(0), by: 4096) {
            let q = feats[start ..< min(feats.dim(0), start + 4096)]
            let d = maximum(sum(q * q, axis: 1, keepDims: true) - 2 * matmul(q, centroids.transposed()) + cNorm, 0)
            let nearest = argPartition(d, kth: k - 1, axis: 1)[0..., ..<k]
            let dist = takeAlong(d, nearest, axis: 1)
            var weight = 1 / square(maximum(sqrt(dist), 1e-6))
            weight = weight / sum(weight, axis: 1, keepDims: true)
            let retrieved = sum(centroids.take(nearest, axis: 0) * weight.expandedDimensions(axis: -1), axis: 1)
            out.append(retrieved * rate + q * (1 - rate))
        }
        return concatenated(out)
    }

    func save(to url: URL) throws {
        try MLX.save(arrays: ["centroids": centroids.asType(.float16)], url: url)
    }

    static func load(from url: URL) throws -> VoiceIndex {
        VoiceIndex(centroids: try loadArrays(url: url)["centroids"]!.asType(.float32))
    }
}
