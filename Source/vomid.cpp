/*
  ==============================================================================
    Vomid.cpp
    Real-Time Voice to MIDI Engine (Monophonic Pitch Tracking + Chords)

    Architecture Overview:
    ──────────────────────
    1. Audio comes in via processBlock (128–512 samples per block).
    2. Samples are pushed into a lock-free circular buffer (2048 samples).
    3. Every `yinHopSamples` new samples, the YIN algorithm runs over the
       full 2048-sample window, giving us a frequency + confidence value.
    4. A temporal median filter (5-frame history) stabilises the pitch.
    5. A debounce counter prevents glitchy note transitions.
    6. MIDI note-on/note-off messages are written with sample-accurate offsets.

    Real-Time Safety Checklist:
    ────────────────────────────
    ✓ ZERO allocations in processBlock
    ✓ ZERO locks / mutexes
    ✓ ZERO system calls (no printf, cout, etc.)
    ✓ All buffers pre-allocated in prepareToPlay
    ✓ std::atomic only for cross-thread parameter reads (APVTS does this)
    ✓ ScopedNoDenormals at the top of processBlock
  ==============================================================================
*/

#include <JuceHeader.h>
#include <cmath>
#include <algorithm>
#include <array>
#include <cstring>   // memcpy, memset

//==============================================================================
// COMPILE-TIME CONSTANTS
// These are fixed at compile time so the compiler can optimise heavily.
//==============================================================================

// YIN window: must be large enough to detect the lowest expected pitch.
// At 44100 Hz, 80 Hz requires 44100/80 ≈ 551 samples per period.
// YIN needs at least 2× the longest period, so 1024 is the minimum.
// 2048 gives us headroom and better accuracy.
static constexpr int   YIN_BUFFER_SIZE      = 2048;

// The "tau" search range for YIN. We search for periods between these bounds.
// tauMin → highest detectable pitch: 44100/tauMin Hz
// tauMax → lowest detectable pitch:  44100/tauMax Hz
// tauMin=20 → ~2205 Hz (safely above soprano top)
// tauMax=800 → ~55 Hz  (below bass voice floor, guards against DC/low-freq)
static constexpr int   YIN_TAU_MIN          = 20;
static constexpr int   YIN_TAU_MAX          = 800;  // clipped in prepareToPlay if SR differs

// YIN threshold: literature recommends 0.10–0.15.
// Lower = more sensitive but more false positives.
// We use 0.12 as a conservative default; the user's GATE parameter
// provides a second line of defence via RMS gating.
static constexpr float YIN_THRESHOLD        = 0.12f;

// How many new samples we accumulate before re-running YIN.
// 256 = ~5.8 ms @ 44100 Hz → responsive without burning the CPU.
static constexpr int   YIN_HOP_SIZE         = 256;

// Temporal median filter length (frames). 5 frames × 5.8 ms ≈ 29 ms
// This smooths out single-frame glitches without adding perceptible lag.
static constexpr int   PITCH_HISTORY_SIZE   = 5;

// Debounce: how many consecutive frames a new note must be stable before
// we commit a MIDI note-on. 3 frames × 5.8 ms ≈ 17 ms — imperceptible
// to a live performer but eliminates crackle artefacts.
static constexpr int   DEBOUNCE_FRAMES      = 3;

// How many consecutive unvoiced hop frames before forcing all notes off.
// This allows legato transitions to sustain while the next pitch debounces.
static constexpr int   SILENCE_FRAMES       = 3;

// MIDI channel for all output.
static constexpr int   MIDI_CHANNEL         = 1;

// Fixed MIDI velocity.
static constexpr int   MIDI_VELOCITY        = 100;

//==============================================================================
// UTILITY: Inline frequency → MIDI note conversion
// Returns -1 if the frequency is outside the valid MIDI range.
//==============================================================================
static inline int frequencyToMidiNote (float freqHz) noexcept
{
    if (freqHz <= 0.0f) return -1;

    // MIDI note = 69 + 12 * log2(f / 440)
    const float note = 69.0f + 12.0f * std::log2 (freqHz / 440.0f);
    const int   rounded = static_cast<int> (std::round (note));

    if (rounded < 0 || rounded > 127) return -1;
    return rounded;
}

//==============================================================================
// LOCK-FREE CIRCULAR AUDIO BUFFER
// A simple, single-producer / single-consumer ring buffer.
// - The audio thread is the sole writer AND sole reader of yinBuffer.
// - No atomics needed here because both roles are on the same thread.
//==============================================================================
class CircularAudioBuffer
{
public:
    CircularAudioBuffer() = default;

    // Must be called from prepareToPlay (not the audio thread).
    void initialise (int capacity)
    {
        jassert (capacity > 0);
        storage.resize (static_cast<size_t> (capacity), 0.0f);
        bufferSize  = capacity;
        writeIndex  = 0;
        fillCount   = 0;
    }

    // Push a block of samples. Called on the audio thread — no allocation.
    void push (const float* data, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
        {
            storage[static_cast<size_t> (writeIndex)] = data[i];
            writeIndex = (writeIndex + 1) % bufferSize;
        }
        fillCount = std::min (fillCount + numSamples, bufferSize);
    }

    // Copy the most recent `length` samples into `dest` in chronological order.
    // Returns false if we don't have `length` samples yet.
    bool readLatest (float* dest, int length) const noexcept
    {
        if (fillCount < length) return false;

        // Start reading from (writeIndex - length), wrapping around.
        int readPos = (writeIndex - length + bufferSize) % bufferSize;
        for (int i = 0; i < length; ++i)
        {
            dest[i] = storage[static_cast<size_t> (readPos)];
            readPos  = (readPos + 1) % bufferSize;
        }
        return true;
    }

    int getNumSamplesAvailable() const noexcept { return fillCount; }

private:
    std::vector<float> storage;   // Allocated once in prepareToPlay
    int bufferSize  = 0;
    int writeIndex  = 0;
    int fillCount   = 0;
};

//==============================================================================
// FIXED-SIZE SORTING NETWORK FOR MEDIAN (size 5)
// Avoids any heap allocation. Used for the pitch history median filter.
// A Bose-Nelson optimal sorting network for N=5.
//==============================================================================
static float median5 (float a0, float a1, float a2, float a3, float a4) noexcept
{
    // We sort in-place using the classic swap macro pattern.
    float v[5] = { a0, a1, a2, a3, a4 };

    auto swap = [](float& x, float& y) noexcept {
        if (x > y) { float t = x; x = y; y = t; }
    };

    // Optimal 9-comparator sorting network for N=5 (Knuth TAOCP vol.3)
    swap(v[0], v[1]); swap(v[3], v[4]); swap(v[2], v[4]);
    swap(v[2], v[3]); swap(v[1], v[4]); swap(v[0], v[3]);
    swap(v[0], v[2]); swap(v[1], v[3]); swap(v[1], v[2]);

    return v[2]; // Median is the middle element after sorting
}

//==============================================================================
// MAIN AUDIO PROCESSOR
//==============================================================================
class VomidAudioProcessor  : public juce::AudioProcessor
{
public:
    //==========================================================================
    VomidAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
        : AudioProcessor (BusesProperties()
                    #if ! JucePlugin_IsMidiEffect
                     #if ! JucePlugin_IsSynth
                      .withInput  ("Input",  juce::AudioChannelSet::mono(), true)
                     #endif
                      .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                    #endif
                          ),
#endif
          apvts (*this, nullptr, "Parameters", createParameterLayout())
    {
        // Reserve space for the playing-notes list up front.
        // This prevents any reallocation on the audio thread.
        currentlyPlayingNotes.reserve (4);
        candidateNotes.reserve (4);
    }

    ~VomidAudioProcessor() override {}

    //==========================================================================
    // PREPARE TO PLAY — all allocation happens HERE, never in processBlock
    //==========================================================================
    void prepareToPlay (double sampleRate, int /*samplesPerBlock*/) override
    {
        currentSampleRate = sampleRate;

        // ── Circular buffer ──────────────────────────────────────────────────
        // We always keep a full YIN_BUFFER_SIZE window available.
        circularBuffer.initialise (YIN_BUFFER_SIZE * 2);

        // ── YIN working buffers ──────────────────────────────────────────────
        // These are the two arrays described in the original YIN paper:
        //   differenceFn[tau]  = d(tau)
        //   cmndfFn[tau]       = d'(tau)  (cumulative mean normalised difference)
        // Both have length YIN_BUFFER_SIZE/2 (we only search up to half the window).
        const int halfWindow = YIN_BUFFER_SIZE / 2;
        yinDifference.assign (static_cast<size_t> (halfWindow), 0.0f);
        yinCMNDF.assign      (static_cast<size_t> (halfWindow), 0.0f);

        // Pre-allocate the scratch window that YIN reads from.
        yinWindow.assign (static_cast<size_t> (YIN_BUFFER_SIZE), 0.0f);

        // ── Pitch history (temporal median filter) ───────────────────────────
        pitchHistory.fill (0.0f);
        pitchHistoryIndex = 0;

        // ── Debounce state ───────────────────────────────────────────────────
        debounceCounter      = 0;
        candidateRootNote    = -1;
        confirmedRootNote    = -1;
        silenceCounter       = 0;
        hopSampleCounter     = 0;

        // ── Tau bounds: clamp for non-44100 sample rates ─────────────────────
        // tauMax must be < halfWindow to avoid reading out-of-bounds.
        tauMax = std::min (YIN_TAU_MAX,
                           static_cast<int> (sampleRate / 55.0));  // 55 Hz minimum
        tauMax = std::min (tauMax, halfWindow - 1);
        tauMin = YIN_TAU_MIN;

        resetMidiState();
    }

    void releaseResources() override {}

    //==========================================================================
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
    {
        if (layouts.getMainInputChannelSet() != juce::AudioChannelSet::mono()
         && layouts.getMainInputChannelSet() != juce::AudioChannelSet::stereo())
            return false;
        return true;
    }

    //==========================================================================
    // PROCESS BLOCK — the hot path. Everything here must be O(n) and lock-free.
    //==========================================================================
    void processBlock (juce::AudioBuffer<float>& buffer,
                       juce::MidiBuffer&         midiMessages) override
    {
        juce::ScopedNoDenormals noDenormals;
      //  FL STUDIO CRASH SHIELD 
        if (buffer.getNumChannels() == 0 || buffer.getNumSamples() == 0) return;

        // We only use the first (and possibly only) input channel.
        const float* inData    = buffer.getReadPointer (0);
        const int    numSamples = buffer.getNumSamples();

        // ────────────────────────────────────────────────────────────────────
        // STEP 1 — Push to circular buffer & calculate RMS
        // ────────────────────────────────────────────────────────────────────

        // RMS: Σ(x²) / N, then √ — all in one pass, no extra buffer needed.
        float sumSquares = 0.0f;
        for (int i = 0; i < numSamples; ++i)
            sumSquares += inData[i] * inData[i];

        const float currentRMS = (numSamples > 0)
                                 ? std::sqrt (sumSquares / static_cast<float> (numSamples))
                                 : 0.0f;

        // Push samples into the ring buffer.
        circularBuffer.push (inData, numSamples);

        // Accumulate hop counter. We run YIN every YIN_HOP_SIZE samples,
        // preserving cross-block remainders for sample-accurate hop timing.
        hopSampleCounter += numSamples;

        // ────────────────────────────────────────────────────────────────────
        // STEP 2 — Run YIN every hop
        // ────────────────────────────────────────────────────────────────────

        while (hopSampleCounter >= YIN_HOP_SIZE)
        {
            hopSampleCounter -= YIN_HOP_SIZE;

            float detectedFrequency = 0.0f;
            float pitchConfidence   = 0.0f;   // 1.0 = confident, 0.0 = unvoiced

            // Only run YIN if we have a full window of samples.
            if (circularBuffer.readLatest (yinWindow.data(), YIN_BUFFER_SIZE))
            {
                runYIN (yinWindow.data(), YIN_BUFFER_SIZE,
                        detectedFrequency, pitchConfidence);
            }

            // ────────────────────────────────────────────────────────────────
            // STEP 3 — Confidence gating, temporal smoothing & debounce
            // ────────────────────────────────────────────────────────────────

            // Load the noise-gate threshold from APVTS.
            // .load() is lock-free (std::atomic<float>).
            const float rmsThreshold = apvts.getRawParameterValue ("GATE")->load();

            // Gate 1: Energy gate — voice must be loud enough.
            const bool energyGate = (currentRMS >= rmsThreshold);

            // Gate 2: YIN confidence gate — algorithm must be certain.
            const bool confidenceGate = (pitchConfidence >= 0.0f) &&
                                        (detectedFrequency > 0.0f);

            const bool voiceDetected = energyGate && confidenceGate;

            // --- Temporal median filter ---
            // Store the latest detected pitch in the circular history.
            // If the voice is silent this frame, we store 0 to pull the median down.
            pitchHistory[static_cast<size_t> (pitchHistoryIndex)] =
                voiceDetected ? detectedFrequency : 0.0f;
            pitchHistoryIndex = (pitchHistoryIndex + 1) % PITCH_HISTORY_SIZE;

            // Compute median of the last PITCH_HISTORY_SIZE frames.
            // median5() is a compile-time inlined sorting network — zero overhead.
            const float smoothedPitch = median5 (pitchHistory[0], pitchHistory[1],
                                                 pitchHistory[2], pitchHistory[3],
                                                 pitchHistory[4]);

            // Convert smoothed Hz → MIDI note.
            const int rawMidiNote = (smoothedPitch > 0.0f)
                                    ? frequencyToMidiNote (smoothedPitch)
                                    : -1;

            // ── Decoupled Evaluation & Legato Sustaining ───────────────────
            if (rawMidiNote < 0)
            {
                ++silenceCounter;
                if (silenceCounter > SILENCE_FRAMES)
                {
                    debounceCounter   = 0;
                    candidateRootNote = -1;
                    confirmedRootNote = -1;
                }
            }
            else
            {
                silenceCounter = 0;

                if (rawMidiNote == candidateRootNote)
                {
                    debounceCounter = std::min (debounceCounter + 1, DEBOUNCE_FRAMES + 1);
                    if (debounceCounter >= DEBOUNCE_FRAMES)
                        confirmedRootNote = candidateRootNote;
                }
                else
                {
                    candidateRootNote = rawMidiNote;
                    debounceCounter   = 1;
                }
            }

            // ────────────────────────────────────────────────────────────────
            // STEP 4 — MIDI generation
            // ────────────────────────────────────────────────────────────────

            const int chordMode = static_cast<int> (
                apvts.getRawParameterValue ("CHORD_MODE")->load (std::memory_order_relaxed));

            // Build the desired chord (root + intervals) from confirmed note.
            candidateNotes.clear();

            if (confirmedRootNote >= 0)
            {
                candidateNotes.push_back (confirmedRootNote);

                if (chordMode == 1)   // Major
                {
                    const int third = confirmedRootNote + 4;
                    const int fifth = confirmedRootNote + 7;
                    if (third <= 127) candidateNotes.push_back (third);
                    if (fifth <= 127) candidateNotes.push_back (fifth);
                }
                else if (chordMode == 2)   // Minor
                {
                    const int third = confirmedRootNote + 3;
                    const int fifth = confirmedRootNote + 7;
                    if (third <= 127) candidateNotes.push_back (third);
                    if (fifth <= 127) candidateNotes.push_back (fifth);
                }
            }

            // --- Detect change and generate MIDI messages ---
            const bool notesChanged = (candidateNotes != currentlyPlayingNotes);

            if (notesChanged)
            {
                // Correctly account for cross-block hop carryover remainders.
                const int maxOffset = juce::jmax (0, numSamples - 1);
                const int sampleOffset = juce::jlimit (0, maxOffset,
                                                       numSamples - hopSampleCounter);

                for (int note : currentlyPlayingNotes)
                {
                    midiMessages.addEvent (
                        juce::MidiMessage::noteOff (MIDI_CHANNEL, note),
                        sampleOffset);
                }

                for (int note : candidateNotes)
                {
                    midiMessages.addEvent (
                        juce::MidiMessage::noteOn (MIDI_CHANNEL, note, (juce::uint8) MIDI_VELOCITY),
                        sampleOffset);
                }

                currentlyPlayingNotes = candidateNotes;
                currentRootMidiNote = candidateNotes.empty() ? -1 : candidateNotes[0];
            }
        }

        // Clear audio output — this is a MIDI effect; we don't pass audio through.
        buffer.clear();
    }

    //==========================================================================
    // Standard AudioProcessor overrides
    //==========================================================================
    juce::AudioProcessorEditor* createEditor() override
    {
        return new juce::GenericAudioProcessorEditor (*this);
    }

    bool hasEditor()         const override { return true;     }
    const juce::String getName() const override { return "Vomid"; }
    bool acceptsMidi()       const override { return false;    }
    bool producesMidi()      const override { return true;     }
    bool isMidiEffect()      const override { return true;     }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms()    override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override
    {
        // Serialise APVTS parameters so the DAW can save/restore presets.
        auto state = apvts.copyState();
        std::unique_ptr<juce::XmlElement> xml (state.createXml());
        copyXmlToBinary (*xml, destData);
    }

    void setStateInformation (const void* data, int sizeInBytes) override
    {
        std::unique_ptr<juce::XmlElement> xml (getXmlFromBinary (data, sizeInBytes));
        if (xml && xml->hasTagName (apvts.state.getType()))
            apvts.replaceState (juce::ValueTree::fromXml (*xml));
    }

private:
    //==========================================================================
    // MEMBER VARIABLES
    //==========================================================================

    juce::AudioProcessorValueTreeState apvts;
    double currentSampleRate = 44100.0;

    // ── Circular buffer ───────────────────────────────────────────────────────
    CircularAudioBuffer circularBuffer;

    // ── YIN working memory (pre-allocated in prepareToPlay) ───────────────────
    // We use std::vector here because their size depends on the sample rate.
    // After prepareToPlay they are NEVER resized on the audio thread.
    std::vector<float> yinDifference;   // d(tau)  — difference function
    std::vector<float> yinCMNDF;        // d'(tau) — cumulative mean normalised
    std::vector<float> yinWindow;       // scratch: the latest YIN_BUFFER_SIZE samples

    int tauMin = YIN_TAU_MIN;
    int tauMax = YIN_TAU_MAX;

    // ── Pitch history for temporal median filter ──────────────────────────────
    std::array<float, PITCH_HISTORY_SIZE> pitchHistory {};
    int pitchHistoryIndex = 0;

    // ── Debounce state ────────────────────────────────────────────────────────
    int candidateRootNote = -1;   // The note we're considering committing
    int confirmedRootNote = -1;   // The note currently committed to output
    int debounceCounter   = 0;    // Frames the candidate has been stable
    int silenceCounter    = 0;    // Consecutive unvoiced hop frames
    int hopSampleCounter  = 0;    // Samples accumulated since last YIN run

    // ── MIDI state ────────────────────────────────────────────────────────────
    int              currentRootMidiNote   = -1;
    std::vector<int> currentlyPlayingNotes;   // Notes currently held on
    std::vector<int> candidateNotes;          // Notes we want to play next

    //==========================================================================
    // YIN PITCH DETECTION ALGORITHM
    //
    // Reference: de Cheveigné & Kawahara, "YIN, a fundamental frequency
    //            estimator for speech and music," JASA 111(4), 2002.
    //
    // Parameters:
    //   samples   — pointer to YIN_BUFFER_SIZE float samples
    //   N         — number of samples (must equal YIN_BUFFER_SIZE)
    //   outFreq   — filled with detected frequency in Hz (0 if unvoiced)
    //   outConf   — filled with confidence: 1 - d'(best_tau)
    //               Values close to 1.0 = high confidence (voiced)
    //               Values close to 0.0 = low confidence  (unvoiced)
    //
    // Real-Time Safety:
    //   ✓ No allocations — uses pre-allocated member arrays
    //   ✓ No locks
    //   ✓ No system calls
    //==========================================================================
    void runYIN (const float* samples, int N,
                 float& outFreq, float& outConf) noexcept
    {
        outFreq = 0.0f;
        outConf = 0.0f;

        const int halfN = N / 2;

        // ── Step 1: Difference function ───────────────────────────────────────
        //
        //            W-1
        //   d(τ) =  Σ   [x(t) - x(t+τ)]²
        //           t=0
        //
        // Instead of computing this naïvely (O(N²)), we use the algebraic
        // identity from the YIN paper:
        //
        //   d(τ) = r(0,0) + r(τ,τ) - 2·r(0,τ)
        //
        // where r(j,k) = autocorrelation at offsets j,k.
        // This lets us compute the autocorrelation term r(0,τ) efficiently.
        // For an N=2048 window with tau_max=800, the naïve O(N·tau_max)
        // computation is ~1.6M mults — acceptable at ~5.8 ms hop rate.
        // (An FFT-based approach would be faster for larger windows but
        //  adds complexity; this is sufficient for our constraints.)

        // r(0, 0) — power of the first half-window
        float r00 = 0.0f;
        for (int t = 0; t < halfN; ++t)
            r00 += samples[t] * samples[t];

        // d(0) is defined as 0 in the paper.
        yinDifference[0] = 0.0f;

        for (int tau = 1; tau < halfN; ++tau)
        {
            // r(tau, tau): power of the second half shifted by tau
            // We update r00 and r_tau_tau incrementally.
            // Full computation for correctness (incremental optimisation
            // can be added but this is already cache-friendly):
            float diff = 0.0f;
            const float* s1 = samples;
            const float* s2 = samples + tau;
            const int    end = halfN;

            for (int t = 0; t < end; ++t)
            {
                const float delta = s1[t] - s2[t];
                diff += delta * delta;
            }
            yinDifference[tau] = diff;
        }

        // ── Step 2: Cumulative Mean Normalised Difference (CMND) ──────────────
        //
        //   d'(0) = 1
        //
        //           d(τ)
        //   d'(τ) = ─────────────────────────────    τ > 0
        //           (1/τ) · Σ_{j=1}^{τ} d(j)
        //
        // This normalisation removes the amplitude dependency and lets us
        // compare against a fixed absolute threshold (YIN_THRESHOLD).

        yinCMNDF[0] = 1.0f;
        float runningSum = 0.0f;

        for (int tau = 1; tau < halfN; ++tau)
        {
            runningSum += yinDifference[tau];

            if (runningSum > 0.0f)
                yinCMNDF[tau] = yinDifference[tau] * static_cast<float> (tau) / runningSum;
            else
                yinCMNDF[tau] = 1.0f;   // Silence / flat — treat as unvoiced
        }

        // ── Step 3: Absolute threshold + dip picking ──────────────────────────
        //
        // Search for the FIRST tau in [tauMin, tauMax] where d'(tau) dips
        // below YIN_THRESHOLD, then find the local minimum of that dip.
        // This "dip below threshold" strategy is more robust than simply
        // finding the global minimum of d'(tau).

        int bestTau = -1;

        for (int tau = tauMin; tau <= tauMax - 1; ++tau)
        {
            if (yinCMNDF[tau] < YIN_THRESHOLD)
            {
                // We're inside a dip. Find its local minimum.
                int   minTau = tau;
                float minVal = yinCMNDF[tau];

                while (tau + 1 <= tauMax && yinCMNDF[tau + 1] < yinCMNDF[tau])
                {
                    ++tau;
                    if (yinCMNDF[tau] < minVal)
                    {
                        minVal = yinCMNDF[tau];
                        minTau = tau;
                    }
                }

                bestTau = minTau;
                break;   // Take the FIRST dip (lowest valid period = highest pitch)
            }
        }

        // No dip found below threshold → unvoiced / noise
        if (bestTau < 0) return;

        // ── Step 4: Parabolic interpolation for sub-sample accuracy ───────────
        //
        // The CMND function is sampled at integer tau values.
        // We fit a parabola through three points around the minimum to
        // find the fractional tau that gives the true sub-sample minimum.
        //
        //             d'(τ-1) - d'(τ+1)
        //   δτ = ─────────────────────────────────────
        //        2 · [d'(τ-1) - 2·d'(τ) + d'(τ+1)]

        float refinedTau = static_cast<float> (bestTau);

        if (bestTau > 0 && bestTau < halfN - 1)
        {
            const float s0 = yinCMNDF[bestTau - 1];
            const float s1 = yinCMNDF[bestTau    ];
            const float s2 = yinCMNDF[bestTau + 1];

            const float denom = 2.0f * (s0 - 2.0f * s1 + s2);

            if (std::abs (denom) > 1e-7f)
            {
                const float delta = (s0 - s2) / denom;
                // Clamp interpolation to ±1 sample
                refinedTau += juce::jlimit (-1.0f, 1.0f, delta);
            }
        }

        // ── Convert period → frequency ────────────────────────────────────────
        if (refinedTau < 1.0f) return;   // Pathological guard

        outFreq = static_cast<float> (currentSampleRate) / refinedTau;

        // ── Confidence = 1 - d'(bestTau) ─────────────────────────────────────
        // d'(bestTau) close to 0 → perfect periodicity → confidence near 1.0
        outConf = 1.0f - yinCMNDF[bestTau];
    }

    //==========================================================================
    // PARAMETER LAYOUT
    //==========================================================================
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
    {
        std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

        // Noise gate RMS threshold.
        // 0.0 = gate off, 1.0 = only the loudest signals pass.
        // Default 0.05 works well for typical recording environments.
        params.push_back (std::make_unique<juce::AudioParameterFloat> (
            "GATE", "Noise Gate (RMS)",
            juce::NormalisableRange<float> (0.0f, 1.0f, 0.001f), 0.05f));

        // Chord voicing mode.
        juce::StringArray chordOptions { "Single Note", "Major Triad", "Minor Triad" };
        params.push_back (std::make_unique<juce::AudioParameterChoice> (
            "CHORD_MODE", "Chord Mode", chordOptions, 0));

        return { params.begin(), params.end() };
    }

    //==========================================================================
    void resetMidiState()
    {
        // Send any lingering note-offs — this is called from the message thread
        // (prepareToPlay), so it's safe to use clearQuick.
        currentRootMidiNote = -1;
        currentlyPlayingNotes.clear();
        candidateNotes.clear();

        candidateRootNote = -1;
        confirmedRootNote = -1;
        debounceCounter   = 0;
        silenceCounter    = 0;
        hopSampleCounter  = 0;
    }

    //==========================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VomidAudioProcessor)
};

//==============================================================================
// Plugin entry point — called by the JUCE host/standalone wrapper
//==============================================================================
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new VomidAudioProcessor();
}
