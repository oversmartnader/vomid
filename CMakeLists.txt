cmake_minimum_required(VERSION 3.20)
project(Vomid VERSION 1.0.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include cstring")
add_subdirectory(JUCE)

juce_add_plugin(Vomid
    PLUGIN_MANUFACTURER_CODE VoMd
    PLUGIN_CODE Vmid
    FORMATS VST3 Standalone
    PRODUCT_NAME "Vomid"
    IS_MIDI_EFFECT TRUE 
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT TRUE
    IS_SYNTH FALSE
)

target_sources(Vomid PRIVATE Source/Vomid.cpp)

target_link_libraries(Vomid
    PRIVATE
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_plugin_client
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
)

target_compile_features(Vomid PRIVATE cxx_std_17)
