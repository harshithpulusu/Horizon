/**
 * Voice Module - Handles speech recognition, wake word detection, and voice controls
 * Part of Horizon AI Assistant modular architecture
 */

class VoiceModule {
    constructor() {
        // Speech recognition variables
        this.recognition = null;
        this.isListening = false;
        
        // Wake words for detection
        this.wakeWords = ['horizon', 'hello horizon', 'hey horizon', 'hi horizon'];
        
        // DOM element references (will be set by main app)
        this.statusIndicator = null;
        this.startButton = null;
        this.stopButton = null;
        this.wakeWordIndicator = null;
        this.userInput = null;
    }

    /**
     * Initialize the voice module with DOM references
     */
    init(elements) {
        this.statusIndicator = elements.statusIndicator;
        this.startButton = elements.startButton;
        this.stopButton = elements.stopButton;
        this.wakeWordIndicator = elements.wakeWordIndicator;
        this.userInput = elements.userInput;
        
        this.setupEventListeners();
        this.requestNotificationPermission();
    }

    /**
     * Setup event listeners for voice controls
     */
    setupEventListeners() {
        if (this.startButton) {
            this.startButton.addEventListener('click', () => this.startListening());
        }
        
        if (this.stopButton) {
            this.stopButton.addEventListener('click', () => this.stopListening());
        }
    }

    /**
     * Request notification permission for wake word alerts
     */
    requestNotificationPermission() {
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission();
        }
    }

    /**
     * Check if message contains wake words
     */
    checkWakeWord(message) {
        const lowerMessage = message.toLowerCase();
        
        for (let wakeWord of this.wakeWords) {
            if (lowerMessage.includes(wakeWord)) {
                this.showWakeWordDetected();
                return true;
            }
        }
        return false;
    }

    /**
     * Show wake word detection animation
     */
    showWakeWordDetected() {
        if (this.wakeWordIndicator) {
            this.wakeWordIndicator.classList.add('wake-word-listening');
            setTimeout(() => {
                this.wakeWordIndicator.classList.remove('wake-word-listening');
            }, 3000);
        }
    }

    /**
     * Start speech recognition
     */
    startListening() {
        // Check browser support
        if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
            if (window.chatModule) {
                window.chatModule.addMessage('⚠️ Speech recognition not supported in this browser.', 'ai');
            }
            return;
        }

        // Update UI
        this.statusIndicator.textContent = 'Listening...';
        if (this.startButton) this.startButton.disabled = true;
        if (this.stopButton) this.stopButton.disabled = false;
        
        // Add listening message to chat
        if (window.chatModule) {
            window.chatModule.addMessage('🎤 Voice listening activated! Say "Horizon" followed by your request...', 'ai');
        }

        // Initialize speech recognition
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        this.recognition.lang = 'en-US';
        this.recognition.interimResults = false;
        this.recognition.maxAlternatives = 1;
        this.recognition.continuous = false;
        this.isListening = true;

        // Handle recognition result
        this.recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            this.userInput.value = transcript;
            
            // Send message using chat module
            if (window.chatModule) {
                window.chatModule.sendMessage();
            }
        };

        // Handle recognition errors
        this.recognition.onerror = (event) => {
            if (window.chatModule) {
                window.chatModule.addMessage('⚠️ Microphone error: ' + event.error, 'ai');
            }
            this.stopListening();
        };

        // Handle recognition end
        this.recognition.onend = () => {
            this.isListening = false;
            if (this.startButton) this.startButton.disabled = false;
            if (this.stopButton) this.stopButton.disabled = true;
            this.statusIndicator.textContent = 'Ready';
        };

        // Start recognition
        this.recognition.start();
    }

    /**
     * Stop speech recognition
     */
    stopListening() {
        if (this.recognition && this.isListening) {
            this.recognition.stop();
        }
        
        this.statusIndicator.textContent = 'Ready';
        if (this.startButton) this.startButton.disabled = false;
        if (this.stopButton) this.stopButton.disabled = true;
        this.isListening = false;
    }

    /**
     * Toggle microphone (for keyboard shortcut)
     */
    toggleMicrophone() {
        if (this.isListening) {
            this.stopListening();
        } else {
            this.startListening();
        }
    }

    /**
     * Check if currently listening
     */
    getListeningState() {
        return this.isListening;
    }

    /**
     * Get available wake words
     */
    getWakeWords() {
        return [...this.wakeWords];
    }

    /**
     * Add new wake word
     */
    addWakeWord(word) {
        if (word && !this.wakeWords.includes(word.toLowerCase())) {
            this.wakeWords.push(word.toLowerCase());
        }
    }

    /**
     * Remove wake word
     */
    removeWakeWord(word) {
        const index = this.wakeWords.indexOf(word.toLowerCase());
        if (index > -1) {
            this.wakeWords.splice(index, 1);
        }
    }
}

// Export for use in main app
window.VoiceModule = VoiceModule;