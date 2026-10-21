import Foundation
import MLX

/// Compares the MLX port against PyTorch fixtures written by
/// `export_rvc_base.py --reference`. Used by `voicekit-cli parity`.
public enum RVCParity {
    public static func run(baseDirectory dir: URL) throws -> [(name: String, relError: Float)] {
        let g = RVCSynthesizer()
        let extra = try RVCWeights.load(g, from: dir.appending(path: "RVCBaseG.safetensors"), extras: ["mel_basis"])
        let d = RVCDiscriminator()
        try RVCWeights.load(d, from: dir.appending(path: "RVCBaseD.safetensors"))
        let ref = try loadArrays(url: dir.appending(path: "rvc_reference.safetensors"))
        func r(_ k: String) -> MLXArray { ref[k]! }

        var results: [(String, Float)] = []
        func check(_ name: String, _ got: MLXArray, _ want: MLXArray) {
            let err = (abs(got - want).max() / (abs(want).max() + 1e-9)).item(Float.self)
            results.append((name, err))
        }

        let t = r("phone").dim(1)
        let mask = MLXArray.ones([1, t, 1])
        let gvec = g.speaker(batch: 1)
        let (mP, logsP) = g.enc_p(phone: r("phone"), pitch: r("pitch").asType(.int32), mask: mask)
        check("enc_p.m", mP, r("m_p"))
        check("enc_p.logs", logsP, r("logs_p"))
        let q = g.enc_q(r("spec"), mask: mask, g: gvec)
        check("enc_q.m", q.m, r("m_q"))
        check("enc_q.logs", q.logs, r("logs_q"))
        check("flow.forward", g.flow(r("m_q"), mask: mask, g: gvec), r("z_fwd"))
        let zRev = g.flow(r("m_p"), mask: mask, g: gvec, reverse: true)
        check("flow.reverse", zRev, r("z_rev"))

        g.dec.m_source.deterministic = true
        let wav = g.dec(r("z_rev")[0..., 130 ..< 194], f0: r("pitchf")[0..., 130 ..< 194], g: gvec)
        check("dec.wave", wav, r("wav"))

        let (logits, fmaps) = d(r("wav")[0..., ..<12800])
        for i in logits.indices {
            check("disc\(i).logits", logits[i], r("d\(i)"))
            check("disc\(i).fmap0", fmaps[i][0], r("fmap\(i)_0"))
        }
        let spectrogram = RVCSpectrogram(melBasis: extra["mel_basis"]!)
        check("mel", spectrogram.mel(fromLinear: spectrogram.linear(r("wav")[0..., ..<12800])), r("mel"))
        return results
    }
}
