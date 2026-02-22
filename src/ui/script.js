// ui/script.js
// Client-side JavaScript for the Kitten TTS Server web interface.
// Handles UI interactions, API communication, audio playback, and settings management.

document.addEventListener('DOMContentLoaded', async function () {
    // --- Global Flags & State ---
    let uiReady = false;
    let listenersAttached = false;
    let isGenerating = false;
    let wavesurfer = null;
    let currentAudioBlobUrl = null;
    let saveStateTimeout = null;

    let currentConfig = {};
    let currentUiState = {};
    let appPresets = [];
    let availableVoices = [];

    let hideGenerationWarning = false;
    let currentVoice = 'Jasper';

    const IS_LOCAL_FILE = window.location.protocol === 'file:';
    // If you always access the server via localhost
    const API_BASE_URL = IS_LOCAL_FILE ? 'http://localhost:8005' : '';
    const UI_STATE_STORAGE_KEY = 'kitten_tts_ui_state_v1';

    const DEBOUNCE_DELAY_MS = 750;

    const KITTEN_TTS_VOICES = [
        'Bella', 'Jasper', 'Luna', 'Bruno',
        'Rosie', 'Hugo', 'Kiki', 'Leo'
    ];

    // --- DOM Element Selectors ---
    const appTitleLink = document.getElementById('app-title-link');
    const themeToggleButton = document.getElementById('theme-toggle-btn');
    const themeSwitchThumb = themeToggleButton ? themeToggleButton.querySelector('.theme-switch-thumb') : null;
    const notificationArea = document.getElementById('notification-area');
    const ttsForm = document.getElementById('tts-form');
    const ttsFormHeader = document.getElementById('tts-form-header');
    const textArea = document.getElementById('text');
    const charCount = document.getElementById('char-count');
    const generateBtn = document.getElementById('generate-btn');
    const splitTextToggle = document.getElementById('split-text-toggle');
    const chunkSizeControls = document.getElementById('chunk-size-controls');
    const chunkSizeSlider = document.getElementById('chunk-size-slider');
    const chunkSizeValue = document.getElementById('chunk-size-value');
    const chunkExplanation = document.getElementById('chunk-explanation');
    const voiceSelect = document.getElementById('voice-select');
    const presetsContainer = document.getElementById('presets-container');
    const presetsPlaceholder = document.getElementById('presets-placeholder');
    const speedSlider = document.getElementById('speed');
    const speedValueDisplay = document.getElementById('speed-value');
    const languageSelectContainer = document.getElementById('language-select-container');
    const languageSelect = document.getElementById('language');
    const outputFormatSelect = document.getElementById('output-format');
    const textProfileSelect = document.getElementById('text-profile');
    const pauseStrengthSelect = document.getElementById('pause-strength');
    const speakerLabelModeSelect = document.getElementById('speaker-label-mode');
    const maxPunctRunSlider = document.getElementById('max-punct-run');
    const maxPunctRunValueDisplay = document.getElementById('max-punct-run-value');
    const normalizePausePunctuationToggle = document.getElementById('normalize-pause-punctuation');
    const dialogueTurnSplittingToggle = document.getElementById('dialogue-turn-splitting');
    const removePunctuationToggle = document.getElementById('remove-punctuation');
    const resetSettingsBtn = document.getElementById('reset-settings-btn');
    const audioPlayerContainer = document.getElementById('audio-player-container');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMessage = document.getElementById('loading-message');
    const loadingStatusText = document.getElementById('loading-status');
    const loadingCancelBtn = document.getElementById('loading-cancel-btn');
    const generationWarningModal = document.getElementById('generation-warning-modal');
    const generationWarningAcknowledgeBtn = document.getElementById('generation-warning-acknowledge');
    const hideGenerationWarningCheckbox = document.getElementById('hide-generation-warning-checkbox');

    // --- Utility Functions ---
    function showNotification(message, type = 'info', duration = 5000) {
        if (!notificationArea) return null;
        const icons = {
            success: '<svg class="notification-icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" /></svg>',
            error: '<svg class="notification-icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" /></svg>',
            warning: '<svg class="notification-icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clip-rule="evenodd" /></svg>',
            info: '<svg class="notification-icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clip-rule="evenodd" /></svg>'
        };
        const typeClassMap = { success: 'notification-success', error: 'notification-error', warning: 'notification-warning', info: 'notification-info' };
        const notificationDiv = document.createElement('div');
        notificationDiv.className = `notification-base ${typeClassMap[type] || 'notification-info'}`;
        notificationDiv.setAttribute('role', 'alert');
        // Create content wrapper
        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'flex items-start flex-grow';
        contentWrapper.innerHTML = `${icons[type] || icons['info']} <span class="block sm:inline">${message}</span>`;

        // Create close button
        const closeButton = document.createElement('button');
        closeButton.type = 'button';
        closeButton.className = 'ml-auto -mx-1.5 -my-1.5 bg-transparent rounded-lg p-1.5 inline-flex h-8 w-8 items-center justify-center text-current hover:bg-slate-200 dark:hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-slate-400 flex-shrink-0';
        closeButton.innerHTML = '<span class="sr-only">Close</span><svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path></svg>';
        closeButton.onclick = () => {
            notificationDiv.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            notificationDiv.style.opacity = '0';
            notificationDiv.style.transform = 'translateY(-20px)';
            setTimeout(() => notificationDiv.remove(), 300);
        };

        // Add both to notification
        notificationDiv.appendChild(contentWrapper);
        notificationDiv.appendChild(closeButton);
        notificationArea.appendChild(notificationDiv);
        if (duration > 0) setTimeout(() => closeButton.click(), duration);
        return notificationDiv;
    }

    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
        return `${minutes}:${secs}`;
    }

    function loadUiStateFromStorage() {
        try {
            const serialized = localStorage.getItem(UI_STATE_STORAGE_KEY);
            return serialized ? JSON.parse(serialized) : {};
        } catch (error) {
            console.warn('Failed to parse persisted UI state, resetting local UI state.', error);
            return {};
        }
    }

    function getProfileDefaults(profileName) {
        const defaults = {
            remove_punctuation: false,
            normalize_pause_punctuation: true,
            pause_strength: 'medium',
            dialogue_turn_splitting: false,
            speaker_label_mode: 'drop',
            max_punct_run: 3
        };
        const profiles = currentConfig?.text_processing?.profiles || {};
        const selectedProfile = profiles?.[profileName] || {};
        return { ...defaults, ...selectedProfile };
    }

    function applyTextOptionsToControls(textOptions = {}) {
        if (pauseStrengthSelect && textOptions.pause_strength !== undefined) {
            pauseStrengthSelect.value = textOptions.pause_strength;
        }
        if (speakerLabelModeSelect && textOptions.speaker_label_mode !== undefined) {
            speakerLabelModeSelect.value = textOptions.speaker_label_mode;
        }
        if (normalizePausePunctuationToggle && textOptions.normalize_pause_punctuation !== undefined) {
            normalizePausePunctuationToggle.checked = !!textOptions.normalize_pause_punctuation;
        }
        if (dialogueTurnSplittingToggle && textOptions.dialogue_turn_splitting !== undefined) {
            dialogueTurnSplittingToggle.checked = !!textOptions.dialogue_turn_splitting;
        }
        if (removePunctuationToggle && textOptions.remove_punctuation !== undefined) {
            removePunctuationToggle.checked = !!textOptions.remove_punctuation;
        }
        if (maxPunctRunSlider && textOptions.max_punct_run !== undefined) {
            maxPunctRunSlider.value = String(textOptions.max_punct_run);
        }
        if (maxPunctRunValueDisplay && maxPunctRunSlider) {
            maxPunctRunValueDisplay.textContent = maxPunctRunSlider.value;
        }
    }

    // --- Theme Management ---
    function applyTheme(theme) {
        const isDark = theme === 'dark';
        document.documentElement.classList.toggle('dark', isDark);
        if (themeSwitchThumb) {
            themeSwitchThumb.classList.toggle('translate-x-6', isDark);
            themeSwitchThumb.classList.toggle('bg-indigo-500', isDark);
            themeSwitchThumb.classList.toggle('bg-white', !isDark);
        }
        if (wavesurfer) {
            wavesurfer.setOptions({
                waveColor: isDark ? '#6366f1' : '#a5b4fc',
                progressColor: isDark ? '#4f46e5' : '#6366f1',
                cursorColor: isDark ? '#cbd5e1' : '#475569',
            });
        }
        localStorage.setItem('uiTheme', theme);
    }

    if (themeToggleButton) {
        themeToggleButton.addEventListener('click', () => {
            const newTheme = document.documentElement.classList.contains('dark') ? 'light' : 'dark';
            applyTheme(newTheme);
            debouncedSaveState();
        });
    }

    // --- UI State Persistence ---
    function saveCurrentUiState() {
        const stateToSave = {
            last_text: textArea ? textArea.value : '',
            last_voice: currentVoice,
            last_chunk_size: chunkSizeSlider ? parseInt(chunkSizeSlider.value, 10) : 120,
            last_split_text_enabled: splitTextToggle ? splitTextToggle.checked : true,
            last_text_profile: textProfileSelect ? textProfileSelect.value : 'balanced',
            last_pause_strength: pauseStrengthSelect ? pauseStrengthSelect.value : 'medium',
            last_speaker_label_mode: speakerLabelModeSelect ? speakerLabelModeSelect.value : 'drop',
            last_max_punct_run: maxPunctRunSlider ? parseInt(maxPunctRunSlider.value, 10) : 3,
            last_normalize_pause_punctuation: normalizePausePunctuationToggle ? normalizePausePunctuationToggle.checked : true,
            last_dialogue_turn_splitting: dialogueTurnSplittingToggle ? dialogueTurnSplittingToggle.checked : false,
            last_remove_punctuation: removePunctuationToggle ? removePunctuationToggle.checked : false,
            hide_generation_warning: hideGenerationWarning,
            theme: localStorage.getItem('uiTheme') || 'dark'
        };
        localStorage.setItem(UI_STATE_STORAGE_KEY, JSON.stringify(stateToSave));
    }

    function debouncedSaveState() {
        // Do not save anything until the entire UI has finished its initial setup.
        if (!uiReady || !listenersAttached) { return; }
        clearTimeout(saveStateTimeout);
        saveStateTimeout = setTimeout(saveCurrentUiState, DEBOUNCE_DELAY_MS);
    }

    // --- Initial Application Setup ---
    function initializeApplication() {
        const preferredTheme = localStorage.getItem('uiTheme') || currentUiState.theme || 'dark';
        applyTheme(preferredTheme);
        const pageTitle = currentConfig?.ui?.title || "Kitten TTS Server";
        document.title = pageTitle;
        if (appTitleLink) appTitleLink.textContent = pageTitle;
        if (ttsFormHeader) ttsFormHeader.textContent = `Generate Speech`;
        loadInitialUiState();
        populateVoices();
        populatePresets();
        if (languageSelectContainer && currentConfig?.ui?.show_language_select === false) {
            languageSelectContainer.classList.add('hidden');
        }
        const initialGenResult = currentConfig.initial_gen_result;
        if (initialGenResult && initialGenResult.outputUrl) {
            initializeWaveSurfer(initialGenResult.outputUrl, initialGenResult);
        }
    }

    async function fetchInitialData() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/ui/initial-data`);
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to fetch initial UI data: ${response.status} ${response.statusText}. Server response: ${errorText}`);
            }
            const data = await response.json();
            currentConfig = data.config || {};
            currentConfig.initial_gen_result = data.initial_gen_result || null;
            currentUiState = loadUiStateFromStorage();
            appPresets = data.presets || [];
            availableVoices = data.available_voices || KITTEN_TTS_VOICES;
            hideGenerationWarning = currentUiState.hide_generation_warning || false;
            currentVoice = currentUiState.last_voice || 'Jasper';

            // This now ONLY sets values. It does NOT attach state-saving listeners.
            initializeApplication();

        } catch (error) {
            console.error("Error fetching initial data:", error);
            showNotification(`Could not load essential application data: ${error.message}. Please try refreshing.`, 'error', 0);
            if (Object.keys(currentConfig).length === 0) {
                currentConfig = { ui: { title: "Kitten TTS Server (Error Mode)" }, generation_defaults: {} };
                availableVoices = KITTEN_TTS_VOICES;
            }
            initializeApplication(); // Attempt to init in a degraded state
        } finally {
            // --- PHASE 2: Attach listeners and enable UI readiness ---
            setTimeout(() => {
                attachStateSavingListeners();
                listenersAttached = true;
                uiReady = true;
            }, 50);
        }
    }

    function loadInitialUiState() {
        if (textArea && currentUiState.last_text) {
            textArea.value = currentUiState.last_text;
            if (charCount) charCount.textContent = textArea.value.length;
        }

        if (splitTextToggle) splitTextToggle.checked = currentUiState.last_split_text_enabled !== undefined ? currentUiState.last_split_text_enabled : true;
        if (chunkSizeSlider && currentUiState.last_chunk_size !== undefined) chunkSizeSlider.value = currentUiState.last_chunk_size;
        if (chunkSizeValue) chunkSizeValue.textContent = chunkSizeSlider ? chunkSizeSlider.value : '120';
        toggleChunkControlsVisibility();

        const genDefaults = currentConfig.generation_defaults || {};
        if (speedSlider) speedSlider.value = genDefaults.speed !== undefined ? genDefaults.speed : 1.0;
        if (speedValueDisplay) speedValueDisplay.textContent = speedSlider.value;
        if (languageSelect) languageSelect.value = genDefaults.language || 'en';
        if (outputFormatSelect) outputFormatSelect.value = currentConfig?.audio_output?.format || 'mp3';
        if (textProfileSelect) {
            const configuredProfile = currentConfig?.text_processing?.active_profile || 'balanced';
            const restoredProfile = currentUiState.last_text_profile || configuredProfile;
            const validProfiles = ['balanced', 'narration', 'dialogue'];
            textProfileSelect.value = validProfiles.includes(restoredProfile)
                ? restoredProfile
                : 'balanced';
        }
        const activeProfile = textProfileSelect ? textProfileSelect.value : 'balanced';
        const effectiveDefaults = getProfileDefaults(activeProfile);
        applyTextOptionsToControls({
            pause_strength: currentUiState.last_pause_strength ?? effectiveDefaults.pause_strength,
            speaker_label_mode: currentUiState.last_speaker_label_mode ?? effectiveDefaults.speaker_label_mode,
            max_punct_run: currentUiState.last_max_punct_run ?? effectiveDefaults.max_punct_run,
            normalize_pause_punctuation: currentUiState.last_normalize_pause_punctuation ?? effectiveDefaults.normalize_pause_punctuation,
            dialogue_turn_splitting: currentUiState.last_dialogue_turn_splitting ?? effectiveDefaults.dialogue_turn_splitting,
            remove_punctuation: currentUiState.last_remove_punctuation ?? effectiveDefaults.remove_punctuation,
        });
        if (hideGenerationWarningCheckbox) hideGenerationWarningCheckbox.checked = hideGenerationWarning;

        if (textArea && !textArea.value && appPresets && appPresets.length > 0) {
            const defaultPreset = appPresets.find(p => p.name === "Standard Narration") || appPresets;
            if (defaultPreset) applyPreset(defaultPreset, false);
        }
    }

    function attachStateSavingListeners() {
        if (textArea) textArea.addEventListener('input', () => { if (charCount) charCount.textContent = textArea.value.length; debouncedSaveState(); });
        if (voiceSelect) voiceSelect.addEventListener('change', () => { currentVoice = voiceSelect.value; debouncedSaveState(); });
        if (splitTextToggle) splitTextToggle.addEventListener('change', () => { toggleChunkControlsVisibility(); debouncedSaveState(); });
        if (chunkSizeSlider) {
            chunkSizeSlider.addEventListener('input', () => { if (chunkSizeValue) chunkSizeValue.textContent = chunkSizeSlider.value; });
            chunkSizeSlider.addEventListener('change', debouncedSaveState);
        }
        if (speedSlider) {
            speedSlider.addEventListener('input', () => {
                if (speedValueDisplay) speedValueDisplay.textContent = speedSlider.value;
            });
            speedSlider.addEventListener('change', debouncedSaveState);
        }
        if (languageSelect) languageSelect.addEventListener('change', debouncedSaveState);
        if (outputFormatSelect) outputFormatSelect.addEventListener('change', debouncedSaveState);
        if (textProfileSelect) {
            textProfileSelect.addEventListener('change', () => {
                const profileDefaults = getProfileDefaults(textProfileSelect.value);
                applyTextOptionsToControls(profileDefaults);
                debouncedSaveState();
            });
        }
        if (pauseStrengthSelect) pauseStrengthSelect.addEventListener('change', debouncedSaveState);
        if (speakerLabelModeSelect) speakerLabelModeSelect.addEventListener('change', debouncedSaveState);
        if (normalizePausePunctuationToggle) normalizePausePunctuationToggle.addEventListener('change', debouncedSaveState);
        if (dialogueTurnSplittingToggle) dialogueTurnSplittingToggle.addEventListener('change', debouncedSaveState);
        if (removePunctuationToggle) removePunctuationToggle.addEventListener('change', debouncedSaveState);
        if (maxPunctRunSlider) {
            maxPunctRunSlider.addEventListener('input', () => {
                if (maxPunctRunValueDisplay) maxPunctRunValueDisplay.textContent = maxPunctRunSlider.value;
            });
            maxPunctRunSlider.addEventListener('change', debouncedSaveState);
        }
    }

    // --- Dynamic UI Population ---
    function populateVoices() {
        if (!voiceSelect) return;
        const currentSelectedValue = voiceSelect.value;
        voiceSelect.innerHTML = '<option value="none">-- Select Voice --</option>';

        availableVoices.forEach(voice => {
            const option = document.createElement('option');
            option.value = voice;
            // Format display name
            option.textContent = voice;
            voiceSelect.appendChild(option);
        });

        const lastSelected = currentUiState.last_voice;
        if (currentSelectedValue !== 'none' && availableVoices.includes(currentSelectedValue)) {
            voiceSelect.value = currentSelectedValue;
            currentVoice = currentSelectedValue;
        } else if (lastSelected && availableVoices.includes(lastSelected)) {
            voiceSelect.value = lastSelected;
            currentVoice = lastSelected;
        } else {
            voiceSelect.value = availableVoices.length > 0 ? availableVoices[0] : 'Jasper';
            currentVoice = voiceSelect.value;
        }
    }

    function populatePresets() {
        if (!presetsContainer || !appPresets) return;
        if (appPresets.length === 0) {
            if (presetsPlaceholder) presetsPlaceholder.textContent = 'No presets available.';
            return;
        }
        if (presetsPlaceholder) presetsPlaceholder.remove();
        presetsContainer.innerHTML = '';
        appPresets.forEach((preset, index) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.id = `preset-btn-${index}`;
            button.className = 'preset-button';
            button.title = `Load '${preset.name}' text and settings`;
            button.textContent = preset.name;
            button.addEventListener('click', () => applyPreset(preset));
            presetsContainer.appendChild(button);
        });
    }

    function applyPreset(presetData, showNotif = true) {
        if (!presetData) return;
        if (textArea && presetData.text !== undefined) {
            textArea.value = presetData.text;
            if (charCount) charCount.textContent = textArea.value.length;
        }
        const genParams = presetData.params || presetData;
        if (speedSlider && genParams.speed !== undefined) speedSlider.value = genParams.speed;
        if (languageSelect && genParams.language !== undefined) languageSelect.value = genParams.language;
        if (outputFormatSelect && genParams.output_format !== undefined) outputFormatSelect.value = genParams.output_format;
        if (splitTextToggle && genParams.split_text !== undefined) splitTextToggle.checked = !!genParams.split_text;
        if (chunkSizeSlider && genParams.chunk_size !== undefined) {
            chunkSizeSlider.value = String(genParams.chunk_size);
            if (chunkSizeValue) chunkSizeValue.textContent = chunkSizeSlider.value;
        }
        if (speedValueDisplay && speedSlider) speedValueDisplay.textContent = speedSlider.value;

        if (genParams.voice && voiceSelect) {
            const voiceExists = Array.from(voiceSelect.options).some(opt => opt.value === genParams.voice);
            if (voiceExists) {
                voiceSelect.value = genParams.voice;
                currentVoice = genParams.voice;
            }
        }
        if (genParams.text_options && typeof genParams.text_options === 'object') {
            const textOptions = genParams.text_options;
            if (textProfileSelect && textOptions.profile) {
                textProfileSelect.value = textOptions.profile;
            }
            const profileDefaults = getProfileDefaults(textProfileSelect ? textProfileSelect.value : 'balanced');
            applyTextOptionsToControls({ ...profileDefaults, ...textOptions });
        } else if (textProfileSelect && genParams.text_profile) {
            textProfileSelect.value = genParams.text_profile;
            applyTextOptionsToControls(getProfileDefaults(textProfileSelect.value));
        }

        toggleChunkControlsVisibility();

        if (showNotif) showNotification(`Preset "${presetData.name}" loaded.`, 'info', 3000);
        debouncedSaveState();
    }

    function toggleChunkControlsVisibility() {
        const isChecked = splitTextToggle ? splitTextToggle.checked : false;
        if (chunkSizeControls) chunkSizeControls.classList.toggle('hidden', !isChecked);
        if (chunkExplanation) chunkExplanation.classList.toggle('hidden', !isChecked);
    }
    if (splitTextToggle) toggleChunkControlsVisibility();

    // --- Audio Player (WaveSurfer) ---
    function initializeWaveSurfer(audioUrl, resultDetails = {}) {
        if (wavesurfer) {
            wavesurfer.unAll();
            wavesurfer.destroy();
            wavesurfer = null;
        }
        if (currentAudioBlobUrl) {
            URL.revokeObjectURL(currentAudioBlobUrl);
            currentAudioBlobUrl = null;
        }
        currentAudioBlobUrl = audioUrl;

        // Ensure the container is clean or re-created
        audioPlayerContainer.innerHTML = `
            <div class="audio-player-card">
                <div class="p-6 sm:p-8">
                    <h2 class="card-header">Generated Audio</h2>
                    <div class="mb-5"><div id="waveform" class="waveform-container"></div></div>
                    <div class="audio-player-controls">
                        <div class="audio-player-buttons">
                            <button id="play-btn" class="btn-primary flex items-center" disabled>
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5 mr-1.5"><path fill-rule="evenodd" d="M2 10a8 8 0 1 1 16 0 8 8 0 0 1-16 0Zm6.39-2.908a.75.75 0 0 1 .766.027l3.5 2.25a.75.75 0 0 1 0 1.262l-3.5 2.25A.75.75 0 0 1 8 12.25v-4.5a.75.75 0 0 1 .39-.658Z" clip-rule="evenodd" /></svg>
                                <span>Play</span>
                            </button>
                            <a id="download-link" href="#" download="kitten_tts_output.wav" class="btn-secondary flex items-center opacity-50 pointer-events-none">
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5 mr-1.5">
                                  <path fill-rule="evenodd" d="M10 3a.75.75 0 01.75.75v6.638l1.96-2.158a.75.75 0 111.08 1.04l-3.25 3.5a.75.75 0 01-1.08 0l-3.25-3.5a.75.75 0 111.08-1.04l1.96 2.158V3.75A.75.75 0 0110 3zM3.75 13a.75.75 0 01.75.75v.008c0 .69.56 1.25 1.25 1.25h8.5c.69 0 1.25-.56 1.25-1.25V13.75a.75.75 0 011.5 0v.008c0 1.518-1.232 2.75-2.75 2.75h-8.5C4.232 16.5 3 15.268 3 13.75v-.008A.75.75 0 013.75 13z" clip-rule="evenodd" />
                                </svg>
                                <span>Download</span>
                            </a>
                        </div>
                        <div class="audio-player-info text-xs sm:text-sm">
                            Voice: <span id="player-voice" class="font-medium text-indigo-600 dark:text-indigo-400">--</span>
                            <span class="mx-1">•</span> Gen Time: <span id="player-gen-time" class="font-medium tabular-nums">--s</span>
                            <span class="mx-1">•</span> Duration: <span id="audio-duration" class="font-medium tabular-nums">--:--</span>
                        </div>
                    </div>
                </div>
            </div>`;

        // Re-select elements after recreating them
        const waveformDiv = audioPlayerContainer.querySelector('#waveform');
        const playBtn = audioPlayerContainer.querySelector('#play-btn');
        const downloadLink = audioPlayerContainer.querySelector('#download-link');
        const playerVoiceSpan = audioPlayerContainer.querySelector('#player-voice');
        const playerGenTimeSpan = audioPlayerContainer.querySelector('#player-gen-time');
        const audioDurationSpan = audioPlayerContainer.querySelector('#audio-duration');

        const audioFilename = resultDetails.filename || (typeof audioUrl === 'string' ? audioUrl.split('/').pop() : 'kitten_tts_output.wav');
        if (downloadLink) {
            downloadLink.href = audioUrl;
            downloadLink.download = audioFilename;
            const downloadTextSpan = downloadLink.querySelector('span');
            if (downloadTextSpan) {
                downloadTextSpan.textContent = `Download ${audioFilename.split('.').pop().toUpperCase()}`;
            }
        }
        if (playerVoiceSpan) {
            const displayVoice = resultDetails.submittedVoice || currentVoice || '--';
            playerVoiceSpan.textContent = displayVoice;
        }
        if (playerGenTimeSpan) playerGenTimeSpan.textContent = resultDetails.genTime ? `${resultDetails.genTime}s` : '--s';

        const playIconSVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5 mr-1.5"><path fill-rule="evenodd" d="M2 10a8 8 0 1 1 16 0 8 8 0 0 1-16 0Zm6.39-2.908a.75.75 0 0 1 .766.027l3.5 2.25a.75.75 0 0 1 0 1.262l-3.5 2.25A.75.75 0 0 1 8 12.25v-4.5a.75.75 0 0 1 .39-.658Z" clip-rule="evenodd" /></svg><span>Play</span>`;
        const pauseIconSVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5 mr-1.5"><path fill-rule="evenodd" d="M2 10a8 8 0 1 1 16 0 8 8 0 0 1-16 0Zm5-2.25A.75.75 0 0 1 7.75 7h4.5a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-.75.75h-4.5a.75.75 0 0 1-.75-.75v-4.5Z" clip-rule="evenodd" /></svg><span>Pause</span>`;
        const isDark = document.documentElement.classList.contains('dark');

        wavesurfer = WaveSurfer.create({
            container: waveformDiv, waveColor: isDark ? '#6366f1' : '#a5b4fc', progressColor: isDark ? '#4f46e5' : '#6366f1',
            cursorColor: isDark ? '#cbd5e1' : '#475569', barWidth: 3, barRadius: 3, cursorWidth: 1, height: 80, barGap: 2,
            responsive: true, url: audioUrl, mediaControls: false, normalize: true,
        });

        wavesurfer.on('ready', () => {
            const duration = wavesurfer.getDuration();
            if (audioDurationSpan) audioDurationSpan.textContent = formatTime(duration);
            if (playBtn) { playBtn.disabled = false; playBtn.innerHTML = playIconSVG; }
            if (downloadLink) { downloadLink.classList.remove('opacity-50', 'pointer-events-none'); downloadLink.setAttribute('aria-disabled', 'false'); }
        });
        wavesurfer.on('play', () => { if (playBtn) playBtn.innerHTML = pauseIconSVG; });
        wavesurfer.on('pause', () => { if (playBtn) playBtn.innerHTML = playIconSVG; });
        wavesurfer.on('finish', () => { if (playBtn) playBtn.innerHTML = playIconSVG; wavesurfer.seekTo(0); });
        wavesurfer.on('error', (err) => {
            console.error("WaveSurfer error:", err);
            showNotification(`Error loading audio waveform: ${err.message || err}`, 'error');
            if (waveformDiv) waveformDiv.innerHTML = `<p class="p-4 text-sm text-red-600 dark:text-red-400">Could not load waveform.</p>`;
            if (playBtn) playBtn.disabled = true;
        });

        if (playBtn) {
            playBtn.onclick = () => {
                if (wavesurfer) {
                    wavesurfer.playPause();
                }
            };
        }
        setTimeout(() => audioPlayerContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 150);
    }

    // --- TTS Generation Logic ---
    function getTTSFormData() {
        const jsonData = {
            text: textArea.value,
            voice: currentVoice,
            speed: parseFloat(speedSlider.value),
            language: languageSelect.value,
            split_text: splitTextToggle.checked,
            chunk_size: parseInt(chunkSizeSlider.value, 10),
            output_format: outputFormatSelect.value || 'mp3'
        };
        const textOptions = {};
        if (textProfileSelect && textProfileSelect.value) textOptions.profile = textProfileSelect.value;
        if (removePunctuationToggle) textOptions.remove_punctuation = removePunctuationToggle.checked;
        if (normalizePausePunctuationToggle) textOptions.normalize_pause_punctuation = normalizePausePunctuationToggle.checked;
        if (pauseStrengthSelect && pauseStrengthSelect.value) textOptions.pause_strength = pauseStrengthSelect.value;
        if (dialogueTurnSplittingToggle) textOptions.dialogue_turn_splitting = dialogueTurnSplittingToggle.checked;
        if (speakerLabelModeSelect && speakerLabelModeSelect.value) textOptions.speaker_label_mode = speakerLabelModeSelect.value;
        if (maxPunctRunSlider) textOptions.max_punct_run = parseInt(maxPunctRunSlider.value, 10);
        if (Object.keys(textOptions).length > 0) {
            jsonData.text_options = textOptions;
        }
        return jsonData;
    }

    async function submitTTSRequest() {
        isGenerating = true;
        showLoadingOverlay();
        const startTime = performance.now();
        const jsonData = getTTSFormData();
        try {
            const response = await fetch(`${API_BASE_URL}/tts`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(jsonData)
            });
            if (!response.ok) {
                const errorResult = await response.json().catch(() => ({ detail: `HTTP error ${response.status}` }));
                throw new Error(errorResult.detail || 'TTS generation failed.');
            }
            const audioBlob = await response.blob();
            const endTime = performance.now();
            const genTime = ((endTime - startTime) / 1000).toFixed(2);
            const contentDisposition = response.headers.get('Content-Disposition');
            const filenameFromServer = contentDisposition
                ? contentDisposition.split('filename=')[1]?.replace(/"/g, '')
                : 'kitten_tts_output.wav';
            const resultDetails = {
                outputUrl: URL.createObjectURL(audioBlob), filename: filenameFromServer, genTime: genTime,
                submittedVoice: jsonData.voice
            };
            initializeWaveSurfer(resultDetails.outputUrl, resultDetails);
            showNotification('Audio generated successfully!', 'success');
        } catch (error) {
            console.error('TTS Generation Error:', error);
            showNotification(error.message || 'An unknown error occurred during TTS generation.', 'error');
        } finally {
            isGenerating = false;
            hideLoadingOverlay();
        }
    }

    // --- Attach main generation event to the button's CLICK ---
    if (generateBtn) {
        generateBtn.addEventListener('click', function (event) {
            event.preventDefault();

            if (isGenerating) {
                showNotification("Generation is already in progress.", "warning");
                return;
            }
            const textContent = textArea.value.trim();
            if (!textContent) {
                showNotification("Please enter some text to generate speech.", 'error');
                return;
            }
            if (!currentVoice || currentVoice === 'none') {
                showNotification("Please select a voice.", 'error');
                return;
            }

            // Check for the generation quality warning.
            if (!hideGenerationWarning) {
                showGenerationWarningModal();
                return;
            }

            submitTTSRequest();
        });
    }

    // --- Modal Handling ---
    function showGenerationWarningModal() {
        if (generationWarningModal) {
            generationWarningModal.style.display = 'flex';
            generationWarningModal.classList.remove('hidden', 'opacity-0');
            generationWarningModal.dataset.state = 'open';
        }
    }
    function hideGenerationWarningModal() {
        if (generationWarningModal) {
            generationWarningModal.classList.add('opacity-0');
            setTimeout(() => {
                generationWarningModal.style.display = 'none';
                generationWarningModal.dataset.state = 'closed';
            }, 300);
        }
    }
    if (generationWarningAcknowledgeBtn) generationWarningAcknowledgeBtn.addEventListener('click', () => {
        if (hideGenerationWarningCheckbox && hideGenerationWarningCheckbox.checked) hideGenerationWarning = true;
        hideGenerationWarningModal(); debouncedSaveState(); submitTTSRequest();
    });
    if (loadingCancelBtn) loadingCancelBtn.addEventListener('click', () => {
        if (isGenerating) { isGenerating = false; hideLoadingOverlay(); showNotification("Generation UI cancelled by user.", "info"); }
    });
    function showLoadingOverlay() {
        if (loadingOverlay && generateBtn && loadingCancelBtn) {
            loadingMessage.textContent = 'Generating audio...';
            loadingStatusText.textContent = 'Please wait. This may take some time.';
            loadingOverlay.style.display = 'flex';
            loadingOverlay.classList.remove('hidden', 'opacity-0'); loadingOverlay.dataset.state = 'open';
            generateBtn.disabled = true; loadingCancelBtn.disabled = false;
        }
    }
    function hideLoadingOverlay() {
        if (loadingOverlay && generateBtn) {
            loadingOverlay.classList.add('opacity-0');
            setTimeout(() => {
                loadingOverlay.style.display = 'none';
                loadingOverlay.dataset.state = 'closed';
            }, 300);
            generateBtn.disabled = false;
        }
    }

    // --- Configuration Management ---
    if (resetSettingsBtn) {
        resetSettingsBtn.addEventListener('click', () => {
            if (!confirm("Clear UI state saved in this browser and reload?")) return;
            localStorage.removeItem(UI_STATE_STORAGE_KEY);
            localStorage.removeItem('uiTheme');
            showNotification("Local UI state cleared.", "success", 2000);
            setTimeout(() => window.location.reload(), 250);
        });
    }

    await fetchInitialData();
});
