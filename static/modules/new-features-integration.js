/**
 * New Features Integration Layer
 * Safe wrapper that connects smart suggestions with existing system.
 * Uses feature detection and automatic fallbacks to prevent breaking changes.
 */

class NewFeaturesIntegration {
    constructor() {
        this.features = {
            smartSuggestions: {
                enabled: false,
                manager: null,
                available: false
            }
        };
        
        this.initialized = false;
        this.fallbackMode = false;
        
        console.log('🔧 New Features Integration Layer initialized');
    }
    
    /**
     * Initialize all new features safely
     */
    async init() {
        try {
            console.log('🚀 Initializing new features integration...');
            
            // Feature detection
            await this.detectAvailableFeatures();
            
            // Initialize available features
            await this.initializeFeatures();
            
            // Setup integration points
            this.setupIntegrationPoints();
            
            // Setup fallback mechanisms
            this.setupFallbacks();
            
            this.initialized = true;
            console.log('✅ New features integration completed successfully');
            
            return true;
            
        } catch (error) {
            console.error('❌ New features integration failed:', error);
            this.enableFallbackMode();
            return false;
        }
    }
    
    /**
     * Detect which features are available
     */
    async detectAvailableFeatures() {
        console.log('🔍 Detecting available features...');
        
        // Check Smart Suggestions availability
        try {
            const suggestionsResponse = await fetch('/api/suggestions/health', {
                method: 'GET',
                timeout: 5000
            });
            
            if (suggestionsResponse.ok) {
                this.features.smartSuggestions.available = true;
                console.log('✅ Smart Suggestions API available');
            }
        } catch (error) {
            console.warn('⚠️ Smart Suggestions API not available:', error.message);
        }
        
        // Check if required DOM elements exist
        const hasInput = document.querySelector('#userInput, #user-input, .chat-input, input[type=\"text\"], textarea');
        if (!hasInput) {
            console.warn('⚠️ No suitable input element found for Smart Suggestions');
            this.features.smartSuggestions.available = false;
        }
        
        console.log('Feature availability:', {
            smartSuggestions: this.features.smartSuggestions.available
        });
    }
    
    /**
     * Initialize available features
     */
    async initializeFeatures() {
        console.log('🔧 Initializing available features...');
        
        // Initialize Smart Suggestions
        if (this.features.smartSuggestions.available) {
            try {
                // Check if manager already exists (from separate script)
                if (window.smartSuggestionsManager) {
                    this.features.smartSuggestions.manager = window.smartSuggestionsManager;
                    this.features.smartSuggestions.enabled = true;
                    console.log('✅ Smart Suggestions manager found and connected');
                } else {
                    console.log('ℹ️ Smart Suggestions manager will be loaded separately');
                }
            } catch (error) {
                console.warn('⚠️ Smart Suggestions initialization failed:', error);
                this.features.smartSuggestions.available = false;
            }
        }
    }
    
    /**
     * Setup integration points with existing system
     */
    setupIntegrationPoints() {
        console.log('🔗 Setting up integration points...');
        
        // Enhance existing sendMessage function (if it exists)
        this.enhanceSendMessage();
        
        // Setup event listeners for existing chat functionality
        this.setupChatIntegration();
        
        // Monitor for dynamic content changes
        this.setupDynamicMonitoring();
        
        console.log('✅ Integration points established');
    }
    
    /**
     * Safely enhance the existing sendMessage function
     */
    enhanceSendMessage() {
        try {
            // Check if sendMessage exists in global scope
            if (typeof window.sendMessage === 'function') {
                const originalSendMessage = window.sendMessage;
                
                // Create enhanced version with fallback
                window.sendMessage = (message, options = {}) => {
                    try {
                        // Learn from message if Smart Suggestions is available
                        if (this.features.smartSuggestions.enabled && 
                            this.features.smartSuggestions.manager &&
                            message && message.trim().length > 5) {
                            
                            this.features.smartSuggestions.manager.learnFromSelection(message);
                        }
                        
                        // Call original function
                        return originalSendMessage.call(this, message, options);
                        
                    } catch (error) {
                        console.error('Enhanced sendMessage error:', error);
                        // Fallback to original function
                        return originalSendMessage.call(this, message, options);
                    }
                };
                
                // Preserve original function properties
                Object.setPrototypeOf(window.sendMessage, originalSendMessage);
                
                console.log('✅ sendMessage function enhanced safely');
                
            } else {
                console.log('ℹ️ sendMessage function not found, skipping enhancement');
            }
            
        } catch (error) {
            console.warn('⚠️ sendMessage enhancement failed:', error);
        }
    }
    
    /**
     * Setup integration with existing chat functionality
     */
    setupChatIntegration() {
        try {
            // Listen for chat events
            document.addEventListener('chatMessage', (event) => {
                try {
                    if (this.features.smartSuggestions.enabled && event.detail && event.detail.message) {
                        // Learn from chat messages
                        this.features.smartSuggestions.manager?.learnFromSelection(event.detail.message);
                    }
                } catch (error) {
                    console.warn('Chat integration error:', error);
                }
            });
            
            // Listen for form submissions
            const forms = document.querySelectorAll('form');
            forms.forEach(form => {
                form.addEventListener('submit', (event) => {
                    try {
                        const inputs = form.querySelectorAll('input[type="text"], textarea');
                        inputs.forEach(input => {
                            if (this.features.smartSuggestions.enabled && input.value && input.value.trim().length > 5) {
                                this.features.smartSuggestions.manager?.learnFromSelection(input.value);
                            }
                        });
                    } catch (error) {
                        console.warn('Form integration error:', error);
                    }
                });
            });
            
            console.log('✅ Chat integration established');
            
        } catch (error) {
            console.warn('⚠️ Chat integration setup failed:', error);
        }
    }
    
    /**
     * Setup monitoring for dynamic content changes
     */
    setupDynamicMonitoring() {
        try {
            // Monitor for new elements that might need feature integration
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    if (mutation.type === 'childList') {
                        mutation.addedNodes.forEach((node) => {
                            if (node.nodeType === Node.ELEMENT_NODE) {
                                this.integrateNewElements(node);
                            }
                        });
                    }
                });
            });
            
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
            
            console.log('✅ Dynamic monitoring established');
            
        } catch (error) {
            console.warn('⚠️ Dynamic monitoring setup failed:', error);
        }
    }
    
    /**
     * Integrate new elements with features
     */
    integrateNewElements(element) {
        try {
            // Check for new input elements that might benefit from smart suggestions
            const inputs = element.querySelectorAll('input[type=\"text\"], textarea');
            if (inputs.length > 0 && this.features.smartSuggestions.available) {
                // Reinitialize or expand smart suggestions if needed
                setTimeout(() => {
                    if (this.features.smartSuggestions.manager?.expandToNewInputs) {
                        this.features.smartSuggestions.manager.expandToNewInputs(inputs);
                    }
                }, 100);
            }
            
        } catch (error) {
            console.warn('Element integration error:', error);
        }
    }
    
    /**
     * Setup fallback mechanisms
     */
    setupFallbacks() {
        try {
            // Global error handler for features
            window.addEventListener('error', (event) => {
                if (event.filename && (
                    event.filename.includes('smart-suggestions') ||
                    event.filename.includes('batch-operations')
                )) {
                    console.warn('Feature error detected, checking fallbacks:', event.error);
                    this.handleFeatureError(event);
                }
            });
            
            // Feature health monitoring
            setInterval(() => {
                this.monitorFeatureHealth();
            }, 30000); // Check every 30 seconds
            
            console.log('✅ Fallback mechanisms established');
            
        } catch (error) {
            console.warn('⚠️ Fallback setup failed:', error);
        }
    }
    
    /**
     * Handle feature errors gracefully
     */
    handleFeatureError(errorEvent) {
        try {
            const errorSource = errorEvent.filename;
            
            if (errorSource.includes('smart-suggestions')) {
                console.warn('Smart Suggestions error detected, disabling feature');
                this.features.smartSuggestions.enabled = false;
                this.features.smartSuggestions.manager = null;
                
                // Hide suggestion UI if visible
                const suggestionContainer = document.querySelector('.smart-suggestions-container');
                if (suggestionContainer) {
                    suggestionContainer.style.display = 'none';
                }
            }
            
        } catch (error) {
            console.error('Error in error handler:', error);
        }
    }
    
    /**
     * Monitor feature health
     */
    async monitorFeatureHealth() {
        try {
            // Check Smart Suggestions health
            if (this.features.smartSuggestions.enabled) {
                try {
                    const response = await fetch('/api/suggestions/health', { timeout: 5000 });
                    if (!response.ok) {
                        throw new Error(`Health check failed: ${response.status}`);
                    }
                } catch (error) {
                    console.warn('Smart Suggestions health check failed:', error);
                    this.features.smartSuggestions.enabled = false;
                }
            }
            
        } catch (error) {
            console.warn('Feature health monitoring error:', error);
        }
    }
    
    /**
     * Enable fallback mode (all features disabled)
     */
    enableFallbackMode() {
        console.warn('🚨 Enabling fallback mode - all new features disabled');
        
        this.fallbackMode = true;
        
        // Disable all features
        Object.keys(this.features).forEach(featureName => {
            this.features[featureName].enabled = false;
            this.features[featureName].available = false;
            this.features[featureName].manager = null;
        });
        
        // Hide all feature UI elements
        try {
            const suggestionContainer = document.querySelector('.smart-suggestions-container');
            const suggestionToggle = document.querySelector('.smart-suggestions-toggle');
            
            [suggestionContainer, suggestionToggle].forEach(element => {
                if (element) {
                    element.style.display = 'none';
                }
            });
        } catch (error) {
            console.warn('Error hiding feature UI:', error);
        }
        
        console.log('✅ Fallback mode enabled - existing chat functionality preserved');
    }
    
    /**
     * Get current feature status
     */
    getStatus() {
        return {
            initialized: this.initialized,
            fallbackMode: this.fallbackMode,
            features: {
                smartSuggestions: {
                    available: this.features.smartSuggestions.available,
                    enabled: this.features.smartSuggestions.enabled,
                    hasManager: !!this.features.smartSuggestions.manager
                }
            }
        };
    }
    
    /**
     * Manually enable/disable features
     */
    toggleFeature(featureName, enabled) {
        if (this.features[featureName]) {
            this.features[featureName].enabled = enabled && this.features[featureName].available;
            console.log(`${featureName} ${enabled ? 'enabled' : 'disabled'}`);
            return true;
        }
        return false;
    }
    
    /**
     * Cleanup resources
     */
    destroy() {
        try {
            // Cleanup feature managers
            Object.values(this.features).forEach(feature => {
                if (feature.manager && typeof feature.manager.destroy === 'function') {
                    feature.manager.destroy();
                }
            });
            
            console.log('✅ New Features Integration Layer destroyed');
            
        } catch (error) {
            console.error('Error during cleanup:', error);
        }
    }
}

// Auto-initialize integration layer
function initNewFeaturesIntegration() {
    try {
        const integration = new NewFeaturesIntegration();
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', async () => {
                await integration.init();
            });
        } else {
            // DOM already ready
            setTimeout(async () => {
                await integration.init();
            }, 1000); // Small delay to ensure other scripts load first
        }
        
        // Store global reference
        window.newFeaturesIntegration = integration;
        
        return integration;
        
    } catch (error) {
        console.error('New Features Integration initialization error:', error);
        return null;
    }
}

// Initialize integration layer
initNewFeaturesIntegration();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NewFeaturesIntegration;
}