/**
 * Multimodal Integration Layer - Safe wrapper for existing functionality
 * This module DOES NOT modify existing functions - it creates optional enhancements
 * If any component fails, existing functionality remains 100% intact
 */

class MultiModalIntegration {
    constructor() {
        this.isActive = false;
        this.originalSendMessage = null;
        this.multiModalManager = null;
        this.contextManager = null;
        this.enhancedChatEndpoint = '/api/multimodal/enhanced-chat';
        
        // Wait for DOM and other modules to be ready
        this.initializeWhenReady();
    }

    initializeWhenReady() {
        /**
         * Initialize integration when all components are ready
         */
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            // Wait a bit for other modules to initialize
            setTimeout(() => this.init(), 100);
        }
    }

    init() {
        /**
         * Initialize the integration layer safely
         */
        try {
            // Check if required components are available
            this.multiModalManager = window.MultiModalManager;
            this.contextManager = window.ContextManager;
            
            // Verify existing sendMessage function exists
            if (typeof window.sendMessage !== 'function') {
                console.warn('⚠️ MultiModalIntegration: sendMessage function not found, integration disabled');
                return;
            }

            // Check if at least one multimodal component is available
            const hasMultiModal = this.multiModalManager && this.multiModalManager.isEnabled;
            const hasContext = this.contextManager && this.contextManager.isEnabled;
            
            if (!hasMultiModal && !hasContext) {
                console.log('ℹ️ MultiModalIntegration: No multimodal components available, staying in compatibility mode');
                return;
            }

            // Store reference to original function
            this.originalSendMessage = window.sendMessage;
            
            // Create enhanced wrapper
            this.createEnhancedSendMessage();
            
            this.isActive = true;
            console.log('✅ MultiModalIntegration: Enhanced functionality active');
            
        } catch (error) {
            console.error('❌ MultiModalIntegration initialization failed:', error);
            this.fallbackToOriginal();
        }
    }

    createEnhancedSendMessage() {
        /**
         * Create enhanced sendMessage that adds multimodal functionality
         * while preserving all original behavior
         */
        const self = this;
        
        // Replace the global sendMessage with our enhanced version
        window.sendMessage = function() {
            try {
                // Get the original message from the input
                const userInput = document.getElementById('userInput');
                if (!userInput) {
                    // If we can't find the input, fall back to original
                    return self.originalSendMessage.apply(this, arguments);
                }
                
                const originalMessage = userInput.value.trim();
                
                // Check if we should use enhanced processing
                const hasImages = self.multiModalManager && self.multiModalManager.hasImages();
                const hasContext = self.contextManager && self.contextManager.contextAware;
                
                if (hasImages || hasContext) {
                    // Use enhanced multimodal processing
                    return self.processMultiModalMessage(originalMessage);
                } else {
                    // Use original function for regular messages
                    return self.originalSendMessage.apply(this, arguments);
                }
                
            } catch (error) {
                console.error('❌ Enhanced sendMessage error:', error);
                // On any error, fall back to original function
                return self.originalSendMessage.apply(this, arguments);
            }
        };
    }

    async processMultiModalMessage(originalMessage) {
        /**
         * Process message with multimodal enhancements
         */
        try {
            if (!originalMessage) {
                console.warn('Empty message provided to multimodal processing');
                return;
            }

            // Get multimodal components
            const attachedImages = this.multiModalManager ? this.multiModalManager.getAttachedImages() : [];
            const contextInfo = this.contextManager ? this.contextManager.getContextForMessage() : { hasContext: false };
            
            // Enhance the message with context if available
            let enhancedMessage = originalMessage;
            if (this.contextManager && this.contextManager.contextAware) {
                enhancedMessage = this.contextManager.enhanceMessage(originalMessage);
            }

            // Show that we're processing
            this.showProcessingState();

            // Upload images if any are attached
            let uploadedImages = [];
            if (attachedImages.length > 0 && this.multiModalManager) {
                console.log(`🖼️ Uploading ${attachedImages.length} image(s)...`);
                const uploadResult = await this.multiModalManager.uploadImages();
                
                if (uploadResult.success) {
                    uploadedImages = uploadResult.images || [];
                    console.log('✅ Images uploaded successfully');
                } else {
                    console.error('❌ Image upload failed:', uploadResult.error);
                    this.showError('Image upload failed. Sending message without images.');
                }
            }

            // Prepare enhanced chat request
            const enhancedChatData = {
                message: enhancedMessage,
                original_message: originalMessage,
                images: uploadedImages,
                context: contextInfo,
                personality: this.getPersonality(),
                session_id: this.contextManager ? this.contextManager.getSessionId() : undefined
            };

            // Send to enhanced chat endpoint
            const response = await fetch(this.enhancedChatEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(enhancedChatData)
            });

            const result = await response.json();
            
            if (result.success) {
                // Use the enhanced prompt with the original /chat endpoint
                await this.sendEnhancedToOriginalChat(result.enhanced_prompt, originalMessage, attachedImages, contextInfo);
            } else {
                throw new Error(result.error || 'Enhanced chat processing failed');
            }

        } catch (error) {
            console.error('❌ Multimodal processing error:', error);
            this.showError('Multimodal processing failed. Using standard chat.');
            
            // Fall back to original function
            return this.originalSendMessage();
        }
    }

    async sendEnhancedToOriginalChat(enhancedPrompt, originalMessage, attachedImages, contextInfo) {
        /**
         * Send enhanced prompt through the original /chat endpoint
         */
        try {
            const personality = this.getPersonality();
            const sessionId = this.contextManager ? this.contextManager.getSessionId() : undefined;
            
            // Update the input field temporarily with enhanced prompt
            const userInput = document.getElementById('userInput');
            const originalValue = userInput.value;
            userInput.value = enhancedPrompt;
            
            // Call original sendMessage function
            await this.originalSendMessage();
            
            // Clear attached images after successful send
            if (this.multiModalManager) {
                this.multiModalManager.clearAttachedImages();
            }
            
            // Add to conversation history
            if (this.contextManager) {
                // We'll need to get the AI response, but for now we'll add after the response comes back
                this.waitForResponseAndLog(originalMessage, attachedImages, contextInfo);
            }
            
        } catch (error) {
            console.error('❌ Error sending enhanced message:', error);
            throw error;
        }
    }

    waitForResponseAndLog(originalMessage, attachedImages, contextInfo) {
        /**
         * Wait for the AI response and log it to conversation history
         */
        // Simple observer to detect when a new AI message is added
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE && 
                        node.classList && 
                        node.classList.contains('message') && 
                        node.classList.contains('ai')) {
                        
                        // Get the AI response text
                        const messageContent = node.querySelector('.message-content');
                        if (messageContent) {
                            const aiResponse = messageContent.textContent;
                            
                            // Add to conversation history
                            if (this.contextManager) {
                                this.contextManager.addConversation(originalMessage, aiResponse, {
                                    hasImages: attachedImages.length > 0,
                                    imageCount: attachedImages.length,
                                    hasContext: contextInfo.hasContext,
                                    personality: this.getPersonality()
                                });
                            }
                            
                            // Stop observing
                            observer.disconnect();
                        }
                    }
                });
            });
        });

        // Start observing
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            observer.observe(chatMessages, { childList: true });
            
            // Stop observing after 30 seconds to prevent memory leaks
            setTimeout(() => observer.disconnect(), 30000);
        }
    }

    showProcessingState() {
        /**
         * Show that multimodal processing is happening
         */
        const statusIndicator = document.getElementById('statusIndicator');
        if (statusIndicator) {
            statusIndicator.textContent = 'Processing multimodal...';
        }
    }

    showError(message) {
        /**
         * Show error notification
         */
        if (this.multiModalManager && this.multiModalManager.showNotification) {
            this.multiModalManager.showNotification(message, 'error');
        } else {
            console.error('MultiModal Error:', message);
        }
    }

    getPersonality() {
        /**
         * Get current personality setting
         */
        const personalitySelect = document.getElementById('personalitySelect');
        return personalitySelect ? personalitySelect.value : 'friendly';
    }

    fallbackToOriginal() {
        /**
         * Restore original functionality if integration fails
         */
        if (this.originalSendMessage) {
            window.sendMessage = this.originalSendMessage;
            console.log('🔄 Fallback: Restored original sendMessage function');
        }
        this.isActive = false;
    }

    // PUBLIC API METHODS

    isIntegrationActive() {
        /**
         * Check if integration is currently active
         */
        return this.isActive;
    }

    getStatus() {
        /**
         * Get integration status for debugging
         */
        return {
            isActive: this.isActive,
            hasMultiModalManager: !!(this.multiModalManager && this.multiModalManager.isEnabled),
            hasContextManager: !!(this.contextManager && this.contextManager.isEnabled),
            originalFunctionPreserved: !!this.originalSendMessage,
            enhancedChatEndpoint: this.enhancedChatEndpoint
        };
    }

    disableIntegration() {
        /**
         * Disable integration and restore original functionality
         */
        this.fallbackToOriginal();
        console.log('🔌 MultiModal integration disabled by user');
    }

    enableIntegration() {
        /**
         * Re-enable integration if components are available
         */
        if (!this.isActive) {
            this.init();
        }
    }
}

// Initialize integration when modules are ready
window.MultiModalIntegration = new MultiModalIntegration();

// Add global status function for debugging
window.getMultiModalStatus = function() {
    const status = {
        integration: window.MultiModalIntegration ? window.MultiModalIntegration.getStatus() : 'Not available',
        multiModalManager: window.MultiModalManager ? window.MultiModalManager.getStatus() : 'Not available',
        contextManager: window.ContextManager ? window.ContextManager.getStatus() : 'Not available'
    };
    
    console.table(status);
    return status;
};