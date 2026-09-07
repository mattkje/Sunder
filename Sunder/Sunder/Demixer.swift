import Foundation

/// Chunked overlap-add separation, matching ZFTurbo/Music-Source-Separation-Training's
/// `demix()` (generic mode) exactly: reflect-pad the mix by `border`, slide
/// `chunkSize`-sample windows with a linear fade-in/out taper at `step`
/// stride, run the model per chunk, overlap-add into a result+counter
/// buffer, normalize, then crop the border back off.
nonisolated final class Demixer {
    private let stft: STFTProcessor
    private let model: SeparatorModel
    private let spec: ModelSpec

    private let chunkSize: Int
    private let step: Int
    private let fadeSize: Int
    private let border: Int

    init(model: SeparatorModel) {
        self.model = model
        spec = model.spec
        chunkSize = spec.chunkSize
        stft = STFTProcessor(nFFT: spec.nFFT, hopLength: spec.hopLength, winLength: spec.winLength, chunkSize: spec.chunkSize)
        step = chunkSize / spec.numOverlap
        fadeSize = chunkSize / 10
        border = chunkSize - step
    }

    private func windowingArray() -> [Float] {
        var w = [Float](repeating: 1, count: chunkSize)
        for i in 0..<fadeSize {
            w[i] = Float(i) / Float(fadeSize - 1)
        }
        for i in 0..<fadeSize {
            w[chunkSize - fadeSize + i] = 1 - Float(i) / Float(fadeSize - 1)
        }
        return w
    }

    private func reflectPad(_ x: [Float], pad: Int) -> [Float] {
        let L = x.count
        var out = [Float](repeating: 0, count: L + 2 * pad)
        for i in 0..<pad { out[i] = x[pad - i] }
        for i in 0..<L { out[pad + i] = x[i] }
        for i in 0..<pad { out[pad + L + i] = x[L - 2 - i] }
        return out
    }

    private func padChunk(_ part: [Float], to length: Int, reflect: Bool) -> [Float] {
        if part.count == length { return part }
        let missing = length - part.count
        if reflect && part.count > 1 {
            // reflect-pad at the end only, matching torch.nn.functional.pad(..., mode="reflect")
            var out = part
            var j = part.count - 2
            for _ in 0..<missing {
                out.append(part[max(j, 0)])
                j -= 1
            }
            return out
        } else {
            return part + [Float](repeating: 0, count: missing)
        }
    }

    /// mix: one array per channel, all the same length. Returns, per stem,
    /// one array per channel, each the same length as the input.
    func separate(mix: [[Float]], isCancelled: () -> Bool = { false }, progress: (Double) -> Void) throws -> [[[Float]]] {
        let channels = mix.count
        let lengthInit = mix[0].count
        let usesBorder = lengthInit > 2 * border && border > 0

        let paddedMix: [[Float]] = usesBorder ? mix.map { reflectPad($0, pad: border) } : mix
        let totalLen = paddedMix[0].count

        var result = Array(repeating: Array(repeating: [Float](repeating: 0, count: totalLen), count: channels), count: spec.numStems)
        var counter = [Float](repeating: 0, count: totalLen)
        let window = windowingArray()

        var i = 0
        let totalSteps = max(1, (totalLen + step - 1) / step)
        var stepIndex = 0

        while i < totalLen {
            if isCancelled() { throw CancellationError() }

            // Each iteration round-trips through CoreML (MLMultiArray,
            // MLFeatureProvider, model.prediction's own ObjC-bridged
            // temporaries). This loop runs on a background Task with no
            // RunLoop to ever turn over and drain the autorelease pool, so
            // without this those temporaries all stay alive until the
            // *whole* loop finishes instead of per chunk -- fine on macOS
            // (no hard per-app ceiling) but on iOS a multi-minute track's
            // worth of them blows past the jetsam memory limit and the app
            // gets killed (EXC_RESOURCE / high watermark exceeded).
            try autoreleasepool {
                let end = min(i + chunkSize, totalLen)
                let segLen = end - i
                let useReflect = segLen > chunkSize / 2

                var channelReal: [[Float]] = []
                var channelImag: [[Float]] = []
                for c in 0..<channels {
                    let rawPart = Array(paddedMix[c][i..<end])
                    let part = padChunk(rawPart, to: chunkSize, reflect: useReflect)
                    let (real, imag) = stft.forward(part)
                    channelReal.append(real)
                    channelImag.append(imag)
                }

                let separated = try model.separate(channelReal: channelReal, channelImag: channelImag)

                let isFirst = (i == 0)
                i += step
                let isLast = i >= totalLen

                var win = window
                if isFirst {
                    for k in 0..<fadeSize { win[k] = 1 }
                }
                if isLast {
                    for k in 0..<fadeSize { win[chunkSize - fadeSize + k] = 1 }
                }

                let chunkStart = i - step // start index this iteration processed, before `i` advanced by `step`
                for n in 0..<spec.numStems {
                    for c in 0..<channels {
                        let (real, imag) = separated[n][c]
                        let chunkAudio = stft.inverse(real: real, imag: imag, T: spec.timeFrames)
                        for n2 in 0..<segLen {
                            result[n][c][chunkStart + n2] += chunkAudio[n2] * win[n2]
                        }
                    }
                }
                for n2 in 0..<segLen {
                    counter[chunkStart + n2] += win[n2]
                }

                stepIndex += 1
                progress(Double(stepIndex) / Double(totalSteps))
            }
        }

        // Normalize in place and crop into `result` itself rather than
        // allocating a second full-track-length copy -- for a multi-minute
        // track at 2 stems x 2 channels that's another few hundred MB of
        // peak memory that's easy to just not need.
        for n in 0..<spec.numStems {
            for c in 0..<channels {
                for k in 0..<totalLen {
                    result[n][c][k] = counter[k] > 0 ? result[n][c][k] / counter[k] : 0
                }
                if usesBorder {
                    result[n][c] = Array(result[n][c][border..<(border + lengthInit)])
                } else if totalLen != lengthInit {
                    result[n][c] = Array(result[n][c][0..<lengthInit])
                }
            }
        }
        return result
    }
}
