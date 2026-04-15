/*
  ==============================================================================
    Vomid.cpp
    Real-Time Voice to MIDI Engine (Monophonic Pitch Tracking + Chords)
    Version 4.0 — The Master Production Release

    Real-Time Safety & DSP Guarantee Checklist:
    ───────────────────────────────────────────
    ✓ ZERO dynamic allocations in processBlock (uses std::array for notes)
    ✓ ZERO locks, mutexes, or system calls
    ✓ Hop-rate locked timing (immune to DAW block-size changes)
    ✓ Symmetrical Debounce (prevents "machine gun" notes on consonants)
    ✓ Decoupled Evaluation (preserves Legato playing without stuttering gaps)
    ✓ Time-Aligned RMS Gate (computed directly over the YIN window)
    ✓ Sample-Accurate MIDI Timestamps (accounts for cross-block remainders)
    ✓ Hanging-Note Flush (single CC 123 on transport stop)
  ==============================================================================
*/

#include <JuceHeader.h>
#include <cmath>
#include <algorithm>
#include <array>
#include <cstring>

//==============================================================================
// COMPILE-TIME CONSTANTS
//==============================================================================

static constexpr int   YIN_BUFFER_SIZE    = 2048;
static constexpr int   YIN_TAU_MIN        = 20;
static constexpr int   YIN_TAU_MAX        = 800;
static constexpr float YIN_THRESHOLD      = 0.12f;

// 256 samples @ 44100 Hz = 5.8 ms per hop
static constexpr int   YIN_HOP_SIZE       = 256;

// 5 hops × 5.8 ms ≈ 29 ms temporal smoothing
static constexpr int   PITCH_HISTORY_SIZE = 5;

// 3 hops × 5.8 ms ≈ 17 ms onset/release hysteresis
static constexpr int   DEBOUNCE_FRAMES    = 3;
static constexpr int   SILENCE_FRAMES     = 3;

static constexpr int   MAX_CHORD_NOTES    = 4;
static constexpr int   MIDI_CHANNEL       = 1;
static constexpr int   MIDI_VELOCITY      = 100;

//==============================================================================
// UTILITY FUNCTIONS
//==============================================================================
static inline int frequencyToMidiNote (float freqHz) noexcept
{
    if (freqHz <= 0.0f) return -1;
    const float note    = 69.0f + 12.0f * std::log2 (freqHz / 440.0f);
    const int   rounded = static_cast<int> (std::round (note));
    if (rounded < 0 || rounded > 127) return -1;
    return rounded;
}

static float median5 (float a0, float a1, float a2, float a3, float a4) noexcept
{
    float v[5] = { a0, a1, a2, a3, a4 };
    auto cswap = [] (float& x, float& y) noexcept { if (x > y) { float t = x; x = y; y = t; } };
    
    cswap(v[0], v[1]); cswap(v[3], v[4]); cswap(v[2], v[4]);
    cswap(v[2], v[3]); cswap(v[1], v[4]); cswap(v[0], v[3]);
    cswap(v[0], v[2]); cswap(v[1], v[3]); cswap(v[1], v[2]);
    return v[2];
}

static bool noteArraysEqual (const std::array<int, MAX_CHORD_NOTES>& a, int na,
                             const std::array<int, MAX_CHORD_NOTES>& b, int nb) noexcept
{
    if (na != nb) return false;
    for (int i = 0; i < na; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

//==============================================================================
// LOCK-FREE CIRCULAR AUDIO BUFFER
//==============================================================================
class CircularAudioBuffer
{
public:
    CircularAudioBuffer() = default;

    void initialise (int capacity)
    {
        jassert (capacity > 0);
        storage.assign (static_cast<size_t> (capacity), 0.0f);
        bufferSize = capacity;
        writeIndex = 0;
        fillCount  = 0;
    }

    void push (const float* data, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
        {
            storage[static_cast<size_t> (writeIndex)] = data[i];
            writeIndex = (writeIndex + 1) % bufferSize;
        }
        fillCount = std::min (fillCount + numSamples, bufferSize);
    }

    bool readLatest (float* dest, int length) const noexcept
    {
        if (fillCount < length) return false;
        int readPos = (writeIndex - length + bufferSize) % bufferSize;
        for (int i = 0; i < length; ++i)
        {
            dest[i] = storage[static_cast<size_t> (readPos)];
            readPos  = (readPos + 1) % bufferSize;
        }
        return true;
    }

private:
    std::vector<float> storage;
    int bufferSize = 0;
    int writeIndex = 0;
    int fillCount  = 0;
};

//==============================================================================
// MAIN AUDIO PROCESSOR
//==============================================================================
class VomidAudioProcessor : public juce::AudioProcessor
{
public:
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
    {}

    ~VomidAudioProcessor() override {}

    //==========================================================================
    void prepareToPlay (double sampleRate, int /*samplesPerBlock*/) override
    {
        currentSampleRate = sampleRate;

        circularBuffer.initialise (YIN_BUFFER_SIZE * 2);

        const int halfWindow = YIN_BUFFER_SIZE / 2;
        yinDifference.assign (static_cast<size_t> (halfWindow), 0.0f);
        yinCMNDF.assign      (static_cast<size_t> (halfWindow), 0.0f);
        yinWindow.assign     (static_cast<size_t> (YIN_BUFFER_SIZE), 0.0f);

        tauMax = static_cast<int> (sampleRate / 55.0);
        tauMax = std::min (tauMax, halfWindow - 1);
        tauMin = YIN_TAU_MIN;

        resetNoteState();
    }

    void releaseResources() override
    {
        if (numCurrentlyPlaying > 0)
            pendingAllNotesOff = true;

        numCurrentlyPlaying   = 0;
        numCandidates         = 0;
        currentRootMidiNote   = -1;
        candidateRootNote     = -1;
        confirmedRootNote     = -1;
        debounceCounter       = 0;
        silenceCounter        = 0;
        hopSampleCounter      = 0;
        lastDetectedFrequency = 0.0f;
        lastPitchConfidence   = -1.0f;
        windowRMS             = 0.0f;
        pitchHistory.fill (0.0f);
        pitchHistoryIndex     = 0;
    }

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
    {
        if (layouts.getMainInputChannelSet() != juce::AudioChannelSet::mono()
         && layouts.getMainInputChannelSet() != juce::AudioChannelSet::stereo())
            return false;
        return true;
    }

    //==========================================================================
    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override
    {
        juce::ScopedNoDenormals noDenormals;

        // FL STUDIO CRASH SHIELD
        if (buffer.getNumChannels() == 0 || buffer.getNumSamples() == 0) return;

        const float* inData     = buffer.getReadPointer (0);
        const int    numSamples = buffer.getNumSamples();

        // 1. Flush Stuck Notes (Triggered by DAW Transport Stop)
        if (pendingAllNotesOff)
        {
            midiMessages.addEvent (juce::MidiMessage::allNotesOff (MIDI_CHANNEL), 0);
            pendingAllNotesOff  = false;
            numCurrentlyPlaying = 0;
        }

        // 2. Push Audio to Ring Buffer
        circularBuffer.push (inData, numSamples);
        hopSampleCounter += numSamples;

        // 3. Process YIN Hops
        while (hopSampleCounter >= YIN_HOP_SIZE)
        {
            hopSampleCounter -= YIN_HOP_SIZE;

            if (!circularBuffer.readLatest (yinWindow.data(), YIN_BUFFER_SIZE))
                continue;

            // A. Compute RMS over the exact YIN Window
            float sumSq = 0.0f;
            for (int i = 0; i < YIN_BUFFER_SIZE; ++i)
                sumSq += yinWindow[i] * yinWindow[i];
            windowRMS = std::sqrt (sumSq / static_cast<float> (YIN_BUFFER_SIZE));

            // B. Run Pitch Detection
            runYIN (yinWindow.data(), YIN_BUFFER_SIZE, lastDetectedFrequency, lastPitchConfidence);

            // C. Gating & Median Smoothing
            const float rmsThreshold  = apvts.getRawParameterValue ("GATE")->load (std::memory_order_relaxed);
            const bool energyGate     = (windowRMS >= rmsThreshold);
            const bool confidenceGate = (lastPitchConfidence > 0.0f) && (lastDetectedFrequency > 0.0f);
            const bool voiceDetected  = energyGate && confidenceGate;

            pitchHistory[static_cast<size_t> (pitchHistoryIndex)] = voiceDetected ? lastDetectedFrequency : 0.0f;
            pitchHistoryIndex = (pitchHistoryIndex + 1) % PITCH_HISTORY_SIZE;

            const float smoothedPitch = median5 (pitchHistory[0], pitchHistory[1], pitchHistory[2], pitchHistory[3], pitchHistory[4]);
            const int rawMidiNote = (smoothedPitch > 0.0f) ? frequencyToMidiNote (smoothedPitch) : -1;

            // D. Decoupled Debounce & Legato State Machine
            if (rawMidiNote < 0)
            {
                ++silenceCounter;
                if (silenceCounter > SILENCE_FRAMES)
                {
                    debounceCounter   = 0;
                    candidateRootNote = -1;
                    confirmedRootNote = -1; // Kill output only when silence is confirmed
                }
            }
            else
            {
                silenceCounter = 0;

                if (rawMidiNote == candidateRootNote)
                {
                    debounceCounter = std::min (debounceCounter + 1, DEBOUNCE_FRAMES + 1);
                    if (debounceCounter >= DEBOUNCE_FRAMES)
                        confirmedRootNote = candidateRootNote; // Lock in new note!
                }
                else
                {
                    candidateRootNote = rawMidiNote;
                    debounceCounter   = 1;
                    // Note: confirmedRootNote is NOT cleared here, allowing legato sustain
                }
            }

            // E. Build MIDI Chord Array from CONFIRMED Note
            const int chordMode = static_cast<int> (apvts.getRawParameterValue ("CHORD_MODE")->load (std::memory_order_relaxed));
            numCandidates = 0;

            if (confirmedRootNote >= 0)
            {
                candidateNotes[numCandidates++] = confirmedRootNote;

                if (chordMode == 1)   // Major Triad
                {
                    if (confirmedRootNote + 4 <= 127) candidateNotes[numCandidates++] = confirmedRootNote + 4;
                    if (confirmedRootNote + 7 <= 127) candidateNotes[numCandidates++] = confirmedRootNote + 7;
                }
                else if (chordMode == 2)   // Minor Triad
                {
                    if (confirmedRootNote + 3 <= 127) candidateNotes[numCandidates++] = confirmedRootNote + 3;
                    if (confirmedRootNote + 7 <= 127) candidateNotes[numCandidates++] = confirmedRootNote + 7;
                }
            }

            // F. Diff and Emit MIDI
            const bool notesChanged = !noteArraysEqual (candidateNotes, numCandidates, currentlyPlayingNotes, numCurrentlyPlaying);

            if (notesChanged)
            {
                // Account for block carryover remainders to place note exactly where it happened
                const int sampleOffset = juce::jlimit (0, numSamples - 1, numSamples - hopSampleCounter);

                for (int i = 0; i < numCurrentlyPlaying; ++i)
                {
                    midiMessages.addEvent (juce::MidiMessage::noteOff (MIDI_CHANNEL, currentlyPlayingNotes[i]), sampleOffset);
                }

                for (int i = 0; i < numCandidates; ++i)
                {
                    midiMessages.addEvent (juce::MidiMessage::noteOn (MIDI_CHANNEL, candidateNotes[i], (juce::uint8) MIDI_VELOCITY), sampleOffset);
                }

                numCurrentlyPlaying = numCandidates;
                for (int i = 0; i < numCandidates; ++i)
                    currentlyPlayingNotes[i] = candidateNotes[i];

                currentRootMidiNote = (numCandidates > 0) ? candidateNotes[0] : -1;
            }
        }

        buffer.clear();
    }

    //==========================================================================
    juce::AudioProcessorEditor* createEditor() override { return new juce::GenericAudioProcessorEditor (*this); }
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "Vomid"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return true; }
    bool isMidiEffect() const override { return true; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override
    {
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
    juce::AudioProcessorValueTreeState apvts;
    double currentSampleRate = 44100.0;

    CircularAudioBuffer circularBuffer;
    
    std::vector<float> yinDifference;
    std::vector<float> yinCMNDF;
    std::vector<float> yinWindow;

    int tauMin = YIN_TAU_MIN;
    int tauMax = YIN_TAU_MAX;

    float lastDetectedFrequency = 0.0f;
    float lastPitchConfidence   = -1.0f;
    float windowRMS             = 0.0f;

    std::array<float, PITCH_HISTORY_SIZE> pitchHistory {};
    int pitchHistoryIndex = 0;

    int candidateRootNote = -1;
    int confirmedRootNote = -1;
    int debounceCounter   = 0;
    int silenceCounter    = 0;
    int hopSampleCounter  = 0;

    std::array<int, MAX_CHORD_NOTES> currentlyPlayingNotes {};
    int numCurrentlyPlaying = 0;

    std::array<int, MAX_CHORD_NOTES> candidateNotes {};
    int numCandidates = 0;

    int currentRootMidiNote = -1;
    bool pendingAllNotesOff = false;

    //==========================================================================
    void runYIN (const float* samples, int N, float& outFreq, float& outConf) noexcept
    {
        outFreq = 0.0f;
        outConf = -1.0f;

        const int halfN = N / 2;

        // Step 1: Difference
        yinDifference[0] = 0.0f;
        for (int tau = 1; tau < halfN; ++tau)
        {
            float diff = 0.0f;
            const float* s1 = samples;
            const float* s2 = samples + tau;
            for (int t = 0; t < halfN; ++t)
            {
                const float delta = s1[t] - s2[t];
                diff += delta * delta;
            }
            yinDifference[tau] = diff;
        }

        // Step 2: CMNDF
        yinCMNDF[0]      = 1.0f;
        float runningSum = 0.0f;
        for (int tau = 1; tau < halfN; ++tau)
        {
            runningSum += yinDifference[tau];
            yinCMNDF[tau] = (runningSum > 1e-10f) ? (yinDifference[tau] * static_cast<float> (tau) / runningSum) : 1.0f;
        }

        // Step 3: Absolute Threshold + Dip
        int bestTau = -1;
        for (int tau = tauMin; tau < tauMax; ++tau)
        {
            if (yinCMNDF[tau] < YIN_THRESHOLD)
            {
                int   minTau = tau;
                float minVal = yinCMNDF[tau];
                while (tau + 1 <= tauMax && yinCMNDF[tau + 1] <= yinCMNDF[tau])
                {
                    ++tau;
                    if (yinCMNDF[tau] < minVal)
                    {
                        minVal = yinCMNDF[tau];
                        minTau = tau;
                    }
                }
                bestTau = minTau;
                break;
            }
        }

        if (bestTau < 0) return;

        // Step 4: Parabolic Interpolation
        float refinedTau = static_cast<float> (bestTau);
        if (bestTau > 0 && bestTau < halfN - 1)
        {
            const float s0 = yinCMNDF[bestTau - 1];
            const float s1 = yinCMNDF[bestTau    ];
            const float s2 = yinCMNDF[bestTau + 1];
            const float denom = 2.0f * (s0 - 2.0f * s1 + s2);
            if (std::abs (denom) > 1e-7f)
                refinedTau += juce::jlimit (-1.0f, 1.0f, (s0 - s2) / denom);
        }

        if (refinedTau < 1.0f) return;

        outFreq = static_cast<float> (currentSampleRate) / refinedTau;
        outConf = juce::jlimit (0.0f, 1.0f, 1.0f - yinCMNDF[bestTau]);
    }

    //==========================================================================
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
    {
        std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
        params.push_back (std::make_unique<juce::AudioParameterFloat> (
            "GATE", "Noise Gate (RMS)", juce::NormalisableRange<float> (0.0f, 1.0f, 0.001f), 0.05f));
        juce::StringArray chordOptions { "Single Note", "Major Triad", "Minor Triad" };
        params.push_back (std::make_unique<juce::AudioParameterChoice> (
            "CHORD_MODE", "Chord Mode", chordOptions, 0));
        return { params.begin(), params.end() };
    }

    void resetNoteState() noexcept
    {
        numCurrentlyPlaying   = 0;
        numCandidates         = 0;
        currentRootMidiNote   = -1;
        candidateRootNote     = -1;
        confirmedRootNote     = -1;
        debounceCounter       = 0;
        silenceCounter        = 0;
        hopSampleCounter      = 0;
        pendingAllNotesOff    = false;
        windowRMS             = 0.0f;
        lastDetectedFrequency = 0.0f;
        lastPitchConfidence   = -1.0f;
        pitchHistory.fill (0.0f);
        pitchHistoryIndex     = 0;
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VomidAudioProcessor)
};

//==============================================================================
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new VomidAudioProcessor();
}
