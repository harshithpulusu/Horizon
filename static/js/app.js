/**
 * Main App Coordinator - Initializes and coordinates all modules
 * Part of Horizon AI Assistant modular architecture
 */

class HorizonApp {
    constructor() {
        // Module instances
        this.chatModule = null;
        this.timerModule = null;
        this.voiceModule = null;
        this.uiModule = null;
        
        // DOM element references
        this.elements = {};
        
        // App state
        this.isInitialized = false;
    }

    /**
     * Initialize the application
     */
    async init() {
        try {
            console.log('🚀 Initializing Horizon AI Assistant...');
            
            // Wait for DOM to be ready
            if (document.readyState === 'loading') {
                await new Promise(resolve => {
                    document.addEventListener('DOMContentLoaded', resolve);
                });
            }
            
            // Gather DOM element references
            this.gatherDOMReferences();
            
            // Initialize modules
            this.initializeModules();
            
            // Setup module coordination
            this.setupModuleCoordination();
            
            // Setup global app features
            this.setupGlobalFeatures();
            
            // Mark as initialized
            this.isInitialized = true;
            
            console.log('✅ Horizon AI Assistant initialized successfully!');
            this.showWelcomeMessage();
            
        } catch (error) {
            console.error('❌ Failed to initialize Horizon AI Assistant:', error);
        }
    }

    /**
     * Gather references to DOM elements
     */
    gatherDOMReferences() {
        this.elements = {
            // Chat elements
            userInput: document.getElementById('userInput'),
            chatMessages: document.getElementById('chatMessages'),
            sendButton: document.getElementById('sendButton'),
            statusIndicator: document.getElementById('statusIndicator'),
            conversationCount: document.getElementById('conversationCount'),
            personalitySelect: document.getElementById('personalitySelect'),
            
            // Voice elements
            startButton: document.getElementById('startListening'),
            stopButton: document.getElementById('stopListening'),
            wakeWordIndicator: document.getElementById('wakeWordIndicator'),
            
            // UI elements
            shortcutsModal: document.getElementById('shortcutsModal'),
            
            // Timer/reminder containers
            activeTimers: document.getElementById('activeTimers'),
            activeReminders: document.getElementById('activeReminders')
        };

        // Validate critical elements
        const criticalElements = ['userInput', 'chatMessages', 'sendButton'];
        for (const elementName of criticalElements) {
            if (!this.elements[elementName]) {
                throw new Error(`Critical element not found: ${elementName}`);
            }
        }
    }

    /**
     * Initialize all modules
     */
    initializeModules() {
        console.log('📦 Initializing modules...');
        
        // Initialize Chat Module
        if (window.ChatModule) {
            this.chatModule = new window.ChatModule();
            this.chatModule.init(this.elements);
            window.chatModule = this.chatModule; // Global reference for compatibility
            console.log('✅ Chat module initialized');
        }
        
        // Initialize Timer Module
        if (window.TimerModule) {
            this.timerModule = new window.TimerModule();
            this.timerModule.init(this.elements);
            window.timerModule = this.timerModule; // Global reference for compatibility
            console.log('✅ Timer module initialized');
        }
        
        // Initialize Voice Module
        if (window.VoiceModule) {
            this.voiceModule = new window.VoiceModule();
            this.voiceModule.init(this.elements);
            window.voiceModule = this.voiceModule; // Global reference for compatibility
            console.log('✅ Voice module initialized');
        }
        
        // Initialize UI Module
        if (window.UIModule) {
            this.uiModule = new window.UIModule();
            this.uiModule.init(this.elements);
            window.uiModule = this.uiModule; // Global reference for compatibility
            console.log('✅ UI module initialized');
        }
    }

    /**
     * Setup coordination between modules
     */
    setupModuleCoordination() {
        console.log('🔗 Setting up module coordination...');
        
        // Setup send button
        if (this.uiModule) {
            this.uiModule.setupSendButton();
            this.uiModule.initPersonalitySelector(this.elements.personalitySelect);
            this.uiModule.initInteractiveElements();
        }
    }

    /**
     * Setup global app features
     */
    setupGlobalFeatures() {
        console.log('🌐 Setting up global features...');
        
        // Request notification permission
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission().then(permission => {
                console.log(`📱 Notification permission: ${permission}`);
            });
        }
        
        // Setup error handling
        this.setupErrorHandling();
        
        // Setup performance monitoring
        this.setupPerformanceMonitoring();
    }

    /**
     * Setup global error handling
     */
    setupErrorHandling() {
        window.addEventListener('error', (event) => {
            console.error('🚨 Global error:', event.error);
            if (this.uiModule) {
                this.uiModule.showNotification('An error occurred. Please refresh the page if problems persist.', 'error');
            }
        });

        window.addEventListener('unhandledrejection', (event) => {
            console.error('🚨 Unhandled promise rejection:', event.reason);
            if (this.uiModule) {
                this.uiModule.showNotification('A network error occurred. Please check your connection.', 'error');
            }
        });
    }

    /**
     * Setup performance monitoring
     */
    setupPerformanceMonitoring() {
        if (window.performance && window.performance.mark) {
            window.performance.mark('horizon-app-initialized');
        }
    }

    /**
     * Show welcome message
     */
    showWelcomeMessage() {
        if (this.chatModule) {
            const welcomeMessages = [
                "👋 Welcome to Horizon AI Assistant! I'm here to help you with anything you need.",
                "🎯 Try asking me to create images, set timers, or just have a conversation!",
                "🎤 You can use voice commands by saying 'Horizon' followed by your request.",
                "⌨️ Press Ctrl+Shift+K to see all keyboard shortcuts!"
            ];
            
            welcomeMessages.forEach((message, index) => {
                setTimeout(() => {
                    this.chatModule.addMessage(message, 'ai');
                }, index * 1500);
            });
        }
    }

    /**
     * Get app status information
     */
    getStatus() {
        return {
            isInitialized: this.isInitialized,
            modules: {
                chat: !!this.chatModule,
                timer: !!this.timerModule,
                voice: !!this.voiceModule,
                ui: !!this.uiModule
            },
            features: {
                speechRecognition: 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window,
                notifications: 'Notification' in window,
                audioContext: 'AudioContext' in window || 'webkitAudioContext' in window
            }
        };
    }

    /**
     * Cleanup method for app shutdown
     */
    cleanup() {
        console.log('🧹 Cleaning up Horizon AI Assistant...');
        
        // Stop any active timers
        if (this.timerModule) {
            this.timerModule.activeTimers.forEach(timer => {
                if (timer.intervalId) {
                    clearInterval(timer.intervalId);
                }
            });
        }
        
        // Stop voice recognition
        if (this.voiceModule && this.voiceModule.isListening) {
            this.voiceModule.stopListening();
        }
        
        console.log('✅ Cleanup complete');
    }
}

// Global functions for backward compatibility
window.sendQuickCommand = function(command) {
    if (window.chatModule) {
        window.chatModule.sendQuickCommand(command);
    }
};

window.toggleShortcutsModal = function() {
    if (window.uiModule) {
        window.uiModule.toggleShortcutsModal();
    }
};

// Initialize the app
const horizonApp = new HorizonApp();

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        horizonApp.init();
    });
} else {
    horizonApp.init();
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    horizonApp.cleanup();
});

// Export for external access
window.horizonApp = horizonApp;