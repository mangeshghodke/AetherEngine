import Foundation
import AetherLibavfilter
import AetherLibavutil

/// AE#923: the dynamics and the stereo dialogue lift of audio shaping, as a libavfilter graph.
///
/// The remix (channel cap, centre lift) happens before this, in the caller's resampler. The graph
/// takes that interleaved Float32 and runs, in order:
///
/// - `dialoguenhance` + `pan` on a stereo output whose source has no centre: the filter extracts
///   the dialogue into a third channel and `pan` folds it back onto both sides at the asked gain.
///   Measured against a mid/side lift on a stereo mix of centred speech and wide music, it separates
///   the speech 3 to 5 dB further from the wide content.
/// - the low shelf (`bassGainDB`), applied to the frame in `push()` before it enters the graph:
///   the coefficient formula of libavfilter's `bass`, which this FFmpeg build does not compile in
///   (see `LowShelfBiquad` at the end of this file). It sits ahead of the dialogue fold, the
///   compressor and the limiter, so the dynamics and the ceiling both see the lifted signal; IIR,
///   so no latency and no pts work; on mono, stereo, 5.1 and 7.1 alike.
/// - `acompressor` for night mode.
/// - `alimiter` at `ceilingDBFS` whenever shaping is active, with lookahead and its own latency
///   compensation: measured against an instant-attack limiter, its splatter above 3 kHz on a
///   two-tone signal under limiting bass hits was 25 dB lower, at the input's own floor.
/// - `aformat` to the format the consumer wants (interleaved Float32 for the renderer, the bridge
///   encoder's own format for the bridge), so no sample conversion is left to the caller.
///
/// **Timing.** The graph is treated as a sample transformer, never as a clock. `dialoguenhance`
/// delays its content by `fft_size - overlap` samples and stamps its output with its input's pts
/// regardless, so trusting graph pts would move the audio 32 ms late at 48 kHz. Instead the graph
/// runs on its own sample counter, the caller's pts are queued here per input chunk, the first
/// `latency` output samples (the priming) are dropped, and a drain pushes the same number of
/// silent samples so the tail comes out too. Output sample k is then input sample k, and carries
/// that sample's pts. `alimiter` compensates its own lookahead (`latency=1`).
///
/// Not thread safe: the owner serialises every call under its own lock.
final class AudioShapingGraph {

    static let ceilingDBFS: Double = -1
    static let limiterAttackMilliseconds: Double = 5
    static let limiterReleaseMilliseconds: Double = 50

    private let chain: String
    /// Only the format pin, for when the full chain cannot be rebuilt: the owner's swresample
    /// output format is chosen for this graph, so the audio still has to come out through one.
    private let passthroughChain: String
    /// The bass lift as a pre-graph biquad; nil while `bassGainDB` is 0.
    private var shelf: LowShelfBiquad?
    /// Whether the graph currently built carries `alimiter`. The shelf only applies behind it: a
    /// boost with no ceiling could clip, and `reset()`'s passthrough fallback has no limiter.
    private var limiterInGraph = false
    private let sampleRate: Int32
    private let channels: Int32
    private let layoutDescription: String
    /// The layout every pushed frame carries; abuffer refuses a frame whose layout differs.
    private var inputLayout = AVChannelLayout()
    /// Content delay the graph adds and this class removes, in samples.
    let latency: Int
    /// The latency of the chain actually built; 0 once `reset()` fell back to the passthrough.
    private var builtLatency = 0

    private var graph: UnsafeMutablePointer<AVFilterGraph>?
    private var source: UnsafeMutablePointer<AVFilterContext>?
    private var sink: UnsafeMutablePointer<AVFilterContext>?
    private var inputSamples: Int64 = 0
    private var trimRemaining = 0
    private var ended = false
    /// Caller pts per queued input chunk, in samples; nil where the caller had none.
    private var pending: [(pts: Int64?, count: Int)] = []

    /// Whether `dialoguenhance` runs: stereo out, no front centre in the source to lift directly.
    static func usesStereoDialogueEnhance(shaping: AudioShaping, outputChannels: Int32, sourceHasCentre: Bool) -> Bool {
        shaping.clamped.dialogueGainDB > 0 && outputChannels == 2 && !sourceHasCentre
    }

    /// `dialoguenhance` processes `fft_size / 4` samples per step and delays its content by the
    /// rest of its window. The window follows the rate, as in af_dialoguenhance.c.
    static func dialogueEnhanceLatency(sampleRate: Int32) -> Int {
        let fft = sampleRate > 100_000 ? 8192 : sampleRate > 50_000 ? 4096 : 2048
        return fft - fft / 4
    }

    /// The filter chain, pure so it can be pinned.
    static func chain(shaping: AudioShaping, stereoDialogueEnhance: Bool,
                      outputFormat: AVSampleFormat, layoutDescription: String) -> String {
        let s = shaping.clamped
        let f = { (v: Double) in String(format: "%.6f", v) }
        var parts: [String] = []
        if stereoDialogueEnhance {
            // L and R come out as the delayed original and the third channel as the dialogue the
            // filter found, so the fold adds `gain - 1` of it to raise the dialogue by `gain`.
            let k = f(s.dialogueGain - 1)
            parts.append("dialoguenhance=original=1:enhance=1:voice=2")
            parts.append("pan=stereo|c0=c0+\(k)*c2|c1=c1+\(k)*c2")
        }
        if let k = s.compressor {
            let lin = { (db: Double) in pow(10, db / 20) }
            parts.append("acompressor=threshold=\(f(max(lin(k.thresholdDB), 0.000976563)))"
                         + ":ratio=\(f(k.ratio)):attack=\(f(k.attackMilliseconds))"
                         + ":release=\(f(k.releaseMilliseconds)):makeup=\(f(lin(k.makeupGainDB)))"
                         + ":knee=\(f(min(max(lin(k.kneeDB), 1), 8))):link=maximum:detection=peak")
        }
        parts.append("alimiter=limit=\(f(pow(10, ceilingDBFS / 20))):attack=\(f(limiterAttackMilliseconds))"
                     + ":release=\(f(limiterReleaseMilliseconds)):level=0:latency=1")
        parts.append(formatPin(outputFormat: outputFormat, layoutDescription: layoutDescription))
        return parts.joined(separator: ",")
    }

    static func formatPin(outputFormat: AVSampleFormat, layoutDescription: String) -> String {
        let fmt = av_get_sample_fmt_name(outputFormat).map { String(cString: $0) } ?? "flt"
        return "aformat=sample_fmts=\(fmt):channel_layouts=\(layoutDescription)"
    }

    /// nil when the graph cannot be built (a filter missing from the FFmpeg build, an option it
    /// refuses); the caller then plays unshaped and says so.
    init?(shaping: AudioShaping, sampleRate: Int32, layout: UnsafePointer<AVChannelLayout>,
          sourceHasCentre: Bool, outputFormat: AVSampleFormat) {
        var described = [CChar](repeating: 0, count: 64)
        guard av_channel_layout_describe(layout, &described, described.count) >= 0 else { return nil }
        let layoutDescription = String(cString: described)
        let enhance = Self.usesStereoDialogueEnhance(shaping: shaping, outputChannels: layout.pointee.nb_channels,
                                                      sourceHasCentre: sourceHasCentre)
        self.chain = Self.chain(shaping: shaping, stereoDialogueEnhance: enhance,
                                outputFormat: outputFormat, layoutDescription: layoutDescription)
        self.passthroughChain = Self.formatPin(outputFormat: outputFormat, layoutDescription: layoutDescription)
        self.sampleRate = sampleRate
        self.channels = layout.pointee.nb_channels
        self.layoutDescription = layoutDescription
        self.latency = enhance ? Self.dialogueEnhanceLatency(sampleRate: sampleRate) : 0
        self.shelf = LowShelfBiquad(gainDB: shaping.clamped.bassGainDB,
                                    frequency: shaping.clamped.bassFrequencyHz,
                                    sampleRate: sampleRate,
                                    channels: Int(layout.pointee.nb_channels))
        av_channel_layout_copy(&inputLayout, layout)
        guard build(chain, latency: latency, limiterInGraph: true) else {
            free()
            return nil
        }
    }

    deinit {
        free()
        av_channel_layout_uninit(&inputLayout)
    }

    private func free() {
        avfilter_graph_free(&graph)
        source = nil
        sink = nil
    }

    private func build(_ chain: String, latency: Int, limiterInGraph: Bool) -> Bool {
        free()
        guard let g = avfilter_graph_alloc(),
              let abuffer = avfilter_get_by_name("abuffer"),
              let abuffersink = avfilter_get_by_name("abuffersink") else { return false }
        graph = g
        let args = "time_base=1/\(sampleRate):sample_rate=\(sampleRate):sample_fmt=flt:channel_layout=\(layoutDescription)"
        guard avfilter_graph_create_filter(&source, abuffer, "in", args, nil, g) >= 0,
              avfilter_graph_create_filter(&sink, abuffersink, "out", nil, nil, g) >= 0 else { return false }

        var inputs = avfilter_inout_alloc()
        var outputs = avfilter_inout_alloc()
        defer { avfilter_inout_free(&inputs); avfilter_inout_free(&outputs) }
        guard let i = inputs, let o = outputs else { return false }
        o.pointee.name = strdup("in"); o.pointee.filter_ctx = source; o.pointee.pad_idx = 0; o.pointee.next = nil
        i.pointee.name = strdup("out"); i.pointee.filter_ctx = sink; i.pointee.pad_idx = 0; i.pointee.next = nil
        let parsed = avfilter_graph_parse_ptr(g, chain, &inputs, &outputs, nil)
        let configured = parsed >= 0 ? avfilter_graph_config(g, nil) : parsed
        guard configured >= 0 else {
            EngineLog.emit("[AudioShaping] filter graph refused (\(configured)): \(chain)", category: .session)
            return false
        }
        inputSamples = 0
        builtLatency = latency
        trimRemaining = latency
        ended = false
        pending.removeAll(keepingCapacity: true)
        self.limiterInGraph = limiterInGraph
        shelf?.reset()
        return true
    }

    /// Drop everything in flight, for a seek or a restart: the filters keep history (the limiter's
    /// lookahead, the compressor's envelope, the enhancer's window) that belongs to the old position.
    func reset() {
        guard !build(chain, latency: latency, limiterInGraph: true) else { return }
        if build(passthroughChain, latency: 0, limiterInGraph: false) {
            EngineLog.emit("[AudioShaping] filter graph could not be rebuilt; audio passes unshaped", category: .session)
        } else {
            free()
            EngineLog.emit("[AudioShaping] filter graph could not be rebuilt, not even unshaped; audio drops until the next rebuild", category: .session)
        }
    }

    /// Hands `frames` frames of interleaved Float32 in, and every output the graph has ready out.
    /// `emit` gets the output's plane pointers (already past any dropped priming), its sample count
    /// and the caller pts of its first sample.
    func process(_ samples: UnsafePointer<Float>, frames: Int, pts: Int64?,
                 emit: ([UnsafeMutableRawPointer?], Int, Int64?) -> Void) {
        guard frames > 0, !ended, graph != nil else { return }
        pending.append((pts, frames))
        push(samples, frames: frames)
        pull(emit: emit)
    }

    /// Pushes the latency's worth of silence and the end of stream, and emits the tail. The graph
    /// needs a `reset()` before it takes input again.
    func drain(emit: ([UnsafeMutableRawPointer?], Int, Int64?) -> Void) {
        guard !ended, graph != nil else { return }
        if builtLatency > 0 {
            let silence = [Float](repeating: 0, count: builtLatency * Int(channels))
            silence.withUnsafeBufferPointer { push($0.baseAddress!, frames: builtLatency) }
        }
        _ = av_buffersrc_add_frame(source, nil)
        ended = true
        pull(emit: emit)
    }

    private func push(_ samples: UnsafePointer<Float>, frames: Int) {
        guard let frame = av_frame_alloc() else { return }
        var owned: UnsafeMutablePointer<AVFrame>? = frame
        defer { av_frame_free(&owned) }
        frame.pointee.format = AV_SAMPLE_FMT_FLT.rawValue
        frame.pointee.sample_rate = sampleRate
        frame.pointee.nb_samples = Int32(frames)
        guard av_channel_layout_copy(&frame.pointee.ch_layout, &inputLayout) >= 0 else { return }
        guard av_frame_get_buffer(frame, 0) >= 0, let dst = frame.pointee.data.0 else { return }
        memcpy(dst, samples, frames * Int(channels) * MemoryLayout<Float>.size)
        if limiterInGraph {
            shelf?.apply(UnsafeMutableRawPointer(dst).assumingMemoryBound(to: Float.self), frames: frames)
        }
        frame.pointee.pts = inputSamples
        inputSamples += Int64(frames)
        let ret = av_buffersrc_add_frame_flags(source, frame, 0)
        if ret < 0 {
            EngineLog.emit("[AudioShaping] filter graph rejected \(frames) samples (\(ret))", category: .session)
        }
    }

    private func pull(emit: ([UnsafeMutableRawPointer?], Int, Int64?) -> Void) {
        guard let out = av_frame_alloc() else { return }
        var owned: UnsafeMutablePointer<AVFrame>? = out
        defer { av_frame_free(&owned) }
        while av_buffersink_get_frame(sink, out) >= 0 {
            defer { av_frame_unref(out) }
            var offset = 0
            var count = Int(out.pointee.nb_samples)
            if trimRemaining > 0 {
                let dropped = min(trimRemaining, count)
                trimRemaining -= dropped
                offset = dropped
                count -= dropped
            }
            guard count > 0 else { continue }
            let fmt = AVSampleFormat(rawValue: out.pointee.format)
            let planar = av_sample_fmt_is_planar(fmt) != 0
            let bytes = Int(av_get_bytes_per_sample(fmt))
            let ch = Int(out.pointee.ch_layout.nb_channels)
            let planeCount = planar ? ch : 1
            let step = planar ? bytes : bytes * ch
            var planes: [UnsafeMutableRawPointer?] = []
            for p in 0..<planeCount {
                let base = out.pointee.extended_data[p].map { UnsafeMutableRawPointer($0) }
                planes.append(base.map { $0 + offset * step })
            }
            emit(planes, count, takePTS(count))
        }
    }

    /// The caller pts of the next `count` output samples' first sample, consuming them.
    private func takePTS(_ count: Int) -> Int64? {
        guard let head = pending.first else { return nil }
        let pts = head.pts
        var left = count
        while left > 0, !pending.isEmpty {
            if pending[0].count > left {
                pending[0].count -= left
                pending[0].pts = pending[0].pts.map { $0 + Int64(left) }
                left = 0
            } else {
                left -= pending[0].count
                pending.removeFirst()
            }
        }
        return pts
    }
}

/// The bass lift as a low-shelf biquad: libavfilter's `bass` coefficient formula applied here,
/// on the frame, because the FFmpeg this engine links is built from an allow-list that does not
/// compile `bass` (or any EQ) in — a chain stage for it would simply never load.
///
/// The coefficients are `af_biquads.c` `case bass:` (`poles == 2`) verbatim: `A = 10^(gain/40)`,
/// `alpha = sin(w0)/(2Q)` with Q = 0.5 (the filter's default `width` for `width_type=q`), and
/// `beta = sqrt(2A)` — the `bass`-specific override of the generic shelf's `2*sqrt(A)`, and what
/// makes it a hi-fi tone control rather than the textbook RBJ shelf. Checked against a real
/// `ffmpeg -af bass=...` across 5 Hz..8 kHz, gains 4/7/10 dB, f0 100/180 Hz: worst gain
/// difference 0.006 dB. Gain is the lift at 0 Hz, the same reference libavfilter documents.
///
/// Transposed direct form II, one state pair per channel, applied in place to interleaved Float32
/// as `push()` copies a frame in: sample-in equals sample-out, so no latency, no pts work, and the
/// limiter still sees the lifted signal as its last stage. Not thread safe; the owner serialises
/// every call under its own lock, like the rest of the graph.
private struct LowShelfBiquad {
    private let b0, b1, b2, a1, a2: Float
    private let channels: Int
    private var z1: [Float]
    private var z2: [Float]

    /// nil when the lift is off, or when the parameters cannot describe a stable shelf.
    init?(gainDB: Double, frequency: Double, sampleRate: Int32, channels: Int) {
        guard gainDB > 0, channels > 0, sampleRate > 0 else { return nil }
        let A = pow(10.0, gainDB / 40.0)
        let w0 = 2.0 * Double.pi * frequency / Double(sampleRate)
        guard w0 > 0, w0 < Double.pi else { return nil }
        let alpha = sin(w0)                    // sin(w0)/(2Q), Q = 0.5
        let beta = sqrt(2.0 * A)               // `case bass:` override, not 2*sqrt(A)
        let c = cos(w0)
        let ba = beta * alpha
        let a0 = (A + 1) + (A - 1) * c + ba
        guard a0.isFinite, a0 != 0 else { return nil }
        b0 = Float(A * ((A + 1) - (A - 1) * c + ba) / a0)
        b1 = Float(2 * A * ((A - 1) - (A + 1) * c) / a0)
        b2 = Float(A * ((A + 1) - (A - 1) * c - ba) / a0)
        a1 = Float(-2 * ((A - 1) + (A + 1) * c) / a0)
        a2 = Float(((A + 1) + (A - 1) * c - ba) / a0)
        self.channels = channels
        z1 = [Float](repeating: 0, count: channels)
        z2 = [Float](repeating: 0, count: channels)
    }

    /// Drops the filter's history, for a seek: it belongs to the old position.
    mutating func reset() {
        for i in z1.indices { z1[i] = 0; z2[i] = 0 }
    }

    /// Applies the shelf in place to `frames` frames of interleaved Float32.
    mutating func apply(_ samples: UnsafeMutablePointer<Float>, frames: Int) {
        guard frames > 0 else { return }
        for ch in 0..<channels {
            var s1 = z1[ch], s2 = z2[ch]
            var i = ch
            for _ in 0..<frames {
                let x = samples[i]
                let y = b0 * x + s1
                s1 = b1 * x - a1 * y + s2
                s2 = b2 * x - a2 * y
                samples[i] = y
                i += channels
            }
            z1[ch] = s1
            z2[ch] = s2
        }
    }
}
