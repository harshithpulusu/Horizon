/**
 * UI Module - Handles keyboard shortcuts, modal controls, and UI interactions
 * Part of Horizon AI Assistant modular architecture
 */

class UIModule {
    constructor() {
        // DOM element references (will be set by main app)
        this.userInput = null;
        this.shortcutsModal = null;
        this.sendButton = null;
    }

    /**
     * Initialize the UI module with DOM references
     */
    init(elements) {
        this.userInput = elements.userInput;
        this.shortcutsModal = elements.shortcutsModal;
        this.sendButton = elements.sendButton;
        
        this.setupKeyboardShortcuts();
        this.setupModalControls();
    }

    /**
     * Setup global keyboard shortcuts
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl+Enter: Send message (global)
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                if (window.chatModule) {
                    window.chatModule.sendMessage();
                }
            }
            // Ctrl+M: Toggle microphone
            else if (e.ctrlKey && e.key === 'm') {
                e.preventDefault();
                if (window.voiceModule) {
                    window.voiceModule.toggleMicrophone();
                }
            }
            // Ctrl+T: Quick timer setup
            else if (e.ctrlKey && e.key === 't') {
                e.preventDefault();
                this.userInput.value = 'Set a timer for ';
                this.userInput.focus();
                this.userInput.setSelectionRange(this.userInput.value.length, this.userInput.value.length);
            }
            // Ctrl+Shift+K: Toggle shortcuts modal
            else if (e.ctrlKey && e.shiftKey && e.key === 'K') {
                e.preventDefault();
                this.toggleShortcutsModal();
            }
            // Escape: Close modals
            else if (e.key === 'Escape') {
                this.closeAllModals();
            }
        });
    }

    /**
     * Setup modal controls (click outside to close, etc.)
     */
    setupModalControls() {
        // Close modal when clicking outside
        if (this.shortcutsModal) {
            this.shortcutsModal.addEventListener('click', (e) => {
                if (e.target === this.shortcutsModal) {
                    this.toggleShortcutsModal();
                }
            });
        }
    }

    /**
     * Toggle keyboard shortcuts modal
     */
    toggleShortcutsModal() {
        if (!this.shortcutsModal) return;
        
        if (this.shortcutsModal.style.display === 'none' || this.shortcutsModal.style.display === '') {
            this.shortcutsModal.style.display = 'flex';
            // Focus trap for accessibility
            this.focusModal();
        } else {
            this.shortcutsModal.style.display = 'none';
        }
    }

    /**
     * Close all open modals
     */
    closeAllModals() {
        if (this.shortcutsModal && this.shortcutsModal.style.display === 'flex') {
            this.shortcutsModal.style.display = 'none';
        }
    }

    /**
     * Focus the modal for accessibility
     */
    focusModal() {
        if (this.shortcutsModal) {
            const focusableElements = this.shortcutsModal.querySelectorAll(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            if (focusableElements.length > 0) {
                focusableElements[0].focus();
            }
        }
    }

    /**
     * Show notification toast
     */
    showNotification(message, type = 'info', duration = 3000) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        // Style the notification
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(20, 20, 20, 0.95);
            backdrop-filter: blur(20px);
            color: #e5e5e5;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid rgba(139, 92, 246, 0.3);
            z-index: 1001;
            font-size: 14px;
            max-width: 300px;
            word-wrap: break-word;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        `;
        
        // Add type-specific styling
        if (type === 'error') {
            notification.style.borderColor = 'rgba(239, 68, 68, 0.3)';
            notification.style.background = 'rgba(239, 68, 68, 0.1)';
        } else if (type === 'success') {
            notification.style.borderColor = 'rgba(34, 197, 94, 0.3)';
            notification.style.background = 'rgba(34, 197, 94, 0.1)';
        }
        
        // Add to DOM
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        // Remove after duration
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, duration);
    }

    /**
     * Setup send button click handler
     */
    setupSendButton() {
        if (this.sendButton) {
            this.sendButton.addEventListener('click', () => {
                if (window.chatModule) {
                    window.chatModule.sendMessage();
                }
            });
        }
    }

    /**
     * Update UI based on application state
     */
    updateUIState(state) {
        // Update various UI elements based on app state
        if (state.isProcessing) {
            this.setProcessingState(true);
        } else {
            this.setProcessingState(false);
        }
    }

    /**
     * Set processing state UI
     */
    setProcessingState(isProcessing) {
        if (this.sendButton) {
            this.sendButton.disabled = isProcessing;
            this.sendButton.textContent = isProcessing ? '...' : 'Send';
        }
    }

    /**
     * Initialize personality selector
     */
    initPersonalitySelector(personalitySelect) {
        if (personalitySelect) {
            personalitySelect.addEventListener('change', (e) => {
                this.showNotification(`Personality changed to: ${e.target.selectedOptions[0].text}`, 'success');
            });
        }
    }

    /**
     * Add ripple effect to buttons
     */
    addRippleEffect(element) {
        element.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                left: ${x}px;
                top: ${y}px;
                background: rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                transform: scale(0);
                animation: ripple 0.6s linear;
                pointer-events: none;
            `;
            
            this.style.position = 'relative';
            this.style.overflow = 'hidden';
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    }

    /**
     * Initialize all interactive elements
     */
    initInteractiveElements() {
        // Add ripple effects to buttons
        const buttons = document.querySelectorAll('.quick-btn, .timer-btn, button');
        buttons.forEach(button => this.addRippleEffect(button));
    }

    /**
     * Get keyboard shortcuts help text
     */
    getKeyboardShortcuts() {
        return [
            { key: 'Ctrl + Enter', description: 'Send message' },
            { key: 'Ctrl + M', description: 'Toggle microphone' },
            { key: 'Ctrl + T', description: 'Set new timer' },
            { key: '↑ / ↓', description: 'Navigate message history' },
            { key: 'Escape', description: 'Close modals' },
            { key: 'Ctrl + Shift + K', description: 'Toggle shortcuts help' }
        ];
    }
}

// Add CSS for ripple animation
const style = document.createElement('style');
style.textContent = `
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Export for use in main app
window.UIModule = UIModule;