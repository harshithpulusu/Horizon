/**
 * Smart Suggestions Manager
 * Self-contained module for auto-complete suggestions that enhances input without breaking existing functionality.
 */

class SmartSuggestionsManager {
    constructor() {
        this.isEnabled = false;
        this.currentSuggestions = [];
        this.selectedIndex = -1;
        this.debounceTimer = null;
        this.debounceDelay = 300; // 300ms debounce
        
        // DOM elements
        this.inputElement = null;
        this.suggestionsContainer = null;
        this.toggleButton = null;
        
        // Configuration
        this.config = {
            maxSuggestions: 5,
            minInputLength: 2,
            apiEndpoint: '/api/suggestions/generate',
            learnEndpoint: '/api/suggestions/learn',
            enableLearning: true,
            showShortcuts: true
        };
        
        // Event handlers (bound to preserve 'this' context)
        this.handleInput = this.handleInput.bind(this);
        this.handleKeyDown = this.handleKeyDown.bind(this);
        this.handleClickOutside = this.handleClickOutside.bind(this);
        this.handleSuggestionClick = this.handleSuggestionClick.bind(this);
        
        console.log('🤖 Smart Suggestions Manager initialized');
    }
    
    /**
     * Initialize suggestions system - safe, non-invasive approach
     */
    init() {
        try {
            // Find input element (safely)
            this.findInputElement();
            
            if (!this.inputElement) {
                console.warn('Smart Suggestions: Input element not found, skipping initialization');
                return false;
            }
            
            // Create suggestions UI
            this.createSuggestionsUI();
            
            // Create toggle button
            this.createToggleButton();
            
            // Load settings
            this.loadSettings();
            
            // Attach event listeners (non-invasively)
            this.attachEventListeners();
            
            console.log('✅ Smart Suggestions initialized successfully');
            return true;
            
        } catch (error) {
            console.error('Smart Suggestions initialization error:', error);
            return false;
        }
    }
    
    /**
     * Safely find the input element without breaking existing functionality
     */
    findInputElement() {
        // Try multiple selectors to find the chat input
        const selectors = [
            '#userInput',
            '#user-input',
            '.chat-input',
            'input[type="text"]',
            'textarea'
        ];
        
        for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element && element.type !== 'hidden') {
                this.inputElement = element;
                console.log(`Found input element: ${selector}`);
                break;
            }
        }
    }
    
    /**
     * Create suggestions UI container
     */
    createSuggestionsUI() {
        // Create container
        this.suggestionsContainer = document.createElement('div');
        this.suggestionsContainer.className = 'smart-suggestions-container';
        this.suggestionsContainer.style.display = 'none';
        
        // Position relative to input
        const inputRect = this.inputElement.getBoundingClientRect();
        const inputParent = this.inputElement.parentElement;
        
        // Insert container after input element
        if (inputParent) {
            inputParent.insertBefore(this.suggestionsContainer, this.inputElement.nextSibling);
        } else {
            document.body.appendChild(this.suggestionsContainer);
        }
        
        console.log('Smart suggestions UI created');
    }
    
    /**
     * Create toggle button for enabling/disabling suggestions
     */
    createToggleButton() {
        this.toggleButton = document.createElement('button');
        this.toggleButton.className = 'smart-suggestions-toggle';
        this.toggleButton.innerHTML = '💡';
        this.toggleButton.title = 'Toggle Smart Suggestions';
        this.toggleButton.type = 'button'; // Prevent form submission
        
        // Add click handler
        this.toggleButton.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.toggleSuggestions();
        });
        
        // Find a good place to insert the button
        const inputContainer = this.inputElement.closest('.input-container, .form-group, .chat-input-container');
        if (inputContainer) {
            inputContainer.appendChild(this.toggleButton);
        } else {
            // Fallback: insert after input
            this.inputElement.parentElement.insertBefore(this.toggleButton, this.inputElement.nextSibling);
        }
        
        console.log('Toggle button created');
    }
    
    /**
     * Attach event listeners safely (non-invasive)
     */
    attachEventListeners() {
        // Input event for suggestion generation
        this.inputElement.addEventListener('input', this.handleInput);
        
        // Keyboard navigation
        this.inputElement.addEventListener('keydown', this.handleKeyDown);
        
        // Click outside to hide suggestions
        document.addEventListener('click', this.handleClickOutside);
        
        // Window resize to reposition suggestions
        window.addEventListener('resize', () => {
            if (this.suggestionsContainer && this.suggestionsContainer.style.display !== 'none') {
                this.positionSuggestions();
            }
        });
        
        console.log('Event listeners attached');
    }
    
    /**
     * Handle input changes with debouncing
     */
    handleInput(event) {
        if (!this.isEnabled) return;
        
        const inputValue = event.target.value;
        
        // Clear previous timer
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        
        // Debounce suggestion generation
        this.debounceTimer = setTimeout(() => {
            this.generateSuggestions(inputValue);
        }, this.debounceDelay);
    }
    
    /**
     * Handle keyboard navigation
     */
    handleKeyDown(event) {
        if (!this.isEnabled || !this.suggestionsContainer || this.suggestionsContainer.style.display === 'none') {
            return;
        }
        
        const suggestions = this.suggestionsContainer.querySelectorAll('.suggestion-item');
        
        switch (event.key) {
            case 'ArrowDown':
                event.preventDefault();
                this.selectedIndex = Math.min(this.selectedIndex + 1, suggestions.length - 1);
                this.updateSelection();
                break;
                
            case 'ArrowUp':
                event.preventDefault();
                this.selectedIndex = Math.max(this.selectedIndex - 1, -1);
                this.updateSelection();
                break;
                
            case 'Enter':
                if (this.selectedIndex >= 0 && suggestions[this.selectedIndex]) {
                    event.preventDefault();
                    this.applySuggestion(suggestions[this.selectedIndex].textContent);
                }
                break;
                
            case 'Escape':
                this.hideSuggestions();
                break;
                
            case 'Tab':
                if (this.selectedIndex >= 0 && suggestions[this.selectedIndex]) {
                    event.preventDefault();
                    this.applySuggestion(suggestions[this.selectedIndex].textContent);
                }
                break;
        }
    }
    
    /**
     * Handle clicks outside suggestions to hide them
     */
    handleClickOutside(event) {
        if (!this.suggestionsContainer) return;
        
        if (!this.suggestionsContainer.contains(event.target) && 
            !this.inputElement.contains(event.target)) {
            this.hideSuggestions();
        }
    }
    
    /**
     * Handle suggestion item clicks
     */
    handleSuggestionClick(event) {
        const suggestionText = event.target.textContent;
        this.applySuggestion(suggestionText);
    }
    
    /**
     * Generate suggestions via API
     */
    async generateSuggestions(inputValue) {
        try {
            const trimmedInput = inputValue.trim();
            
            // Don't generate suggestions for very short input
            if (trimmedInput.length < this.config.minInputLength) {
                this.hideSuggestions();
                return;
            }
            
            // API call
            const response = await fetch(this.config.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    input: trimmedInput,
                    limit: this.config.maxSuggestions
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success && data.suggestions) {
                this.displaySuggestions(data.suggestions, trimmedInput);
            } else {
                this.hideSuggestions();
            }
            
        } catch (error) {
            console.error('Error generating suggestions:', error);
            this.hideSuggestions();
        }
    }
    
    /**
     * Display suggestions in the UI
     */
    displaySuggestions(suggestions, currentInput) {
        if (!suggestions || suggestions.length === 0) {
            this.hideSuggestions();
            return;
        }
        
        this.currentSuggestions = suggestions;
        this.selectedIndex = -1;
        
        // Clear container
        this.suggestionsContainer.innerHTML = '';
        
        // Create suggestion items
        suggestions.forEach((suggestion, index) => {
            const item = document.createElement('div');
            item.className = 'suggestion-item';
            item.textContent = suggestion;
            
            // Highlight matching part
            if (currentInput && suggestion.toLowerCase().startsWith(currentInput.toLowerCase())) {
                const matchPart = suggestion.substring(0, currentInput.length);
                const restPart = suggestion.substring(currentInput.length);
                item.innerHTML = `<strong>${matchPart}</strong>${restPart}`;
            }
            
            // Add click handler
            item.addEventListener('click', this.handleSuggestionClick);
            
            this.suggestionsContainer.appendChild(item);
        });
        
        // Add keyboard shortcut hint
        if (this.config.showShortcuts && suggestions.length > 0) {
            const hint = document.createElement('div');
            hint.className = 'suggestion-hint';
            hint.innerHTML = '<small>↑↓ navigate • Enter/Tab to select • Esc to close</small>';
            this.suggestionsContainer.appendChild(hint);
        }
        
        // Position and show
        this.positionSuggestions();
        this.suggestionsContainer.style.display = 'block';
    }
    
    /**
     * Position suggestions container relative to input
     */
    positionSuggestions() {
        if (!this.inputElement || !this.suggestionsContainer) return;
        
        const inputRect = this.inputElement.getBoundingClientRect();
        const containerRect = this.suggestionsContainer.offsetParent.getBoundingClientRect();
        
        this.suggestionsContainer.style.position = 'absolute';
        this.suggestionsContainer.style.top = `${inputRect.bottom - containerRect.top + 5}px`;
        this.suggestionsContainer.style.left = `${inputRect.left - containerRect.left}px`;
        this.suggestionsContainer.style.width = `${inputRect.width}px`;
        this.suggestionsContainer.style.zIndex = '1000';
    }
    
    /**
     * Update visual selection
     */
    updateSelection() {
        const items = this.suggestionsContainer.querySelectorAll('.suggestion-item');
        
        items.forEach((item, index) => {
            if (index === this.selectedIndex) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        });
    }
    
    /**
     * Apply selected suggestion
     */
    applySuggestion(suggestionText) {
        // Set input value
        this.inputElement.value = suggestionText;
        
        // Trigger input event to notify other components
        const event = new Event('input', { bubbles: true });
        this.inputElement.dispatchEvent(event);
        
        // Learn from this selection if enabled
        if (this.config.enableLearning) {
            this.learnFromSelection(suggestionText);
        }
        
        // Hide suggestions
        this.hideSuggestions();
        
        // Focus input
        this.inputElement.focus();
        
        // Position cursor at end
        this.inputElement.setSelectionRange(suggestionText.length, suggestionText.length);
    }
    
    /**
     * Hide suggestions
     */
    hideSuggestions() {
        if (this.suggestionsContainer) {
            this.suggestionsContainer.style.display = 'none';
            this.currentSuggestions = [];
            this.selectedIndex = -1;
        }
    }
    
    /**
     * Toggle suggestions on/off
     */
    toggleSuggestions() {
        this.isEnabled = !this.isEnabled;
        
        // Update button appearance
        if (this.toggleButton) {
            this.toggleButton.innerHTML = this.isEnabled ? '💡' : '💡';
            this.toggleButton.style.opacity = this.isEnabled ? '1' : '0.5';
            this.toggleButton.title = this.isEnabled ? 'Disable Smart Suggestions' : 'Enable Smart Suggestions';
        }
        
        // Hide suggestions if disabled
        if (!this.isEnabled) {
            this.hideSuggestions();
        }
        
        // Save setting
        this.saveSettings();
        
        console.log(`Smart suggestions ${this.isEnabled ? 'enabled' : 'disabled'}`);
    }
    
    /**
     * Learn from user selections
     */
    async learnFromSelection(selectedText) {
        try {
            await fetch(this.config.learnEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    input: selectedText
                })
            });
        } catch (error) {
            console.warn('Learning API error:', error);
        }
    }
    
    /**
     * Save settings to localStorage
     */
    saveSettings() {
        try {
            const settings = {
                enabled: this.isEnabled,
                config: this.config
            };
            localStorage.setItem('smartSuggestionsSettings', JSON.stringify(settings));
        } catch (error) {
            console.warn('Failed to save suggestions settings:', error);
        }
    }
    
    /**
     * Load settings from localStorage
     */
    loadSettings() {
        try {
            const saved = localStorage.getItem('smartSuggestionsSettings');
            if (saved) {
                const settings = JSON.parse(saved);
                this.isEnabled = settings.enabled !== false; // Default to true
                
                // Merge config
                if (settings.config) {
                    Object.assign(this.config, settings.config);
                }
            } else {
                this.isEnabled = true; // Default enabled
            }
            
            // Update UI
            if (this.toggleButton) {
                this.toggleButton.style.opacity = this.isEnabled ? '1' : '0.5';
                this.toggleButton.title = this.isEnabled ? 'Disable Smart Suggestions' : 'Enable Smart Suggestions';
            }
            
        } catch (error) {
            console.warn('Failed to load suggestions settings:', error);
            this.isEnabled = true; // Fallback to enabled
        }
    }
    
    /**
     * Get current status
     */
    getStatus() {
        return {
            enabled: this.isEnabled,
            currentSuggestions: this.currentSuggestions.length,
            selectedIndex: this.selectedIndex,
            hasInput: !!this.inputElement,
            hasContainer: !!this.suggestionsContainer
        };
    }
    
    /**
     * Cleanup resources
     */
    destroy() {
        // Remove event listeners
        if (this.inputElement) {
            this.inputElement.removeEventListener('input', this.handleInput);
            this.inputElement.removeEventListener('keydown', this.handleKeyDown);
        }
        
        document.removeEventListener('click', this.handleClickOutside);
        
        // Clear timers
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        
        // Remove UI elements
        if (this.suggestionsContainer) {
            this.suggestionsContainer.remove();
        }
        
        if (this.toggleButton) {
            this.toggleButton.remove();
        }
        
        console.log('Smart Suggestions Manager destroyed');
    }
}

// Auto-initialize when DOM is ready (safe, non-invasive)
function initSmartSuggestions() {
    // Check if feature is available
    if (typeof fetch === 'undefined') {
        console.warn('Smart Suggestions: fetch API not available');
        return null;
    }
    
    try {
        const manager = new SmartSuggestionsManager();
        const initialized = manager.init();
        
        if (initialized) {
            // Store global reference for debugging
            window.smartSuggestionsManager = manager;
            return manager;
        } else {
            console.warn('Smart Suggestions: initialization failed');
            return null;
        }
        
    } catch (error) {
        console.error('Smart Suggestions initialization error:', error);
        return null;
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSmartSuggestions);
} else {
    initSmartSuggestions();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SmartSuggestionsManager;
}