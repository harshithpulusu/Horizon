/**
 * Horizon AI - Advanced Personality Blending & Mood-Based Switching System
 * Enables dynamic personality combinations and mood-responsive AI behavior
 */

class PersonalityBlendingSystem {
    constructor() {
        this.currentBlend = null;
        this.moodState = 'neutral';
        this.personalityWeights = {};
        this.moodHistory = [];
        this.blendingRules = {};
        
        this.initializeSystem();
        this.setupEventListeners();
        this.createBlendingUI();
        this.startMoodMonitoring();
        
        console.log('🎭 Personality Blending System initialized');
    }

    initializeSystem() {
        // Define personality traits and their blend compatibility
        this.personalityTraits = {
            'friendly': {
                traits: { empathy: 0.9, warmth: 0.9, optimism: 0.8, energy: 0.7 },
                blendableWith: ['enthusiastic', 'creative', 'zen'],
                incompatibleWith: ['analytical'],
                dominantIn: 'social_interactions'
            },
            'professional': {
                traits: { formality: 0.9, precision: 0.9, efficiency: 0.8, structure: 0.9 },
                blendableWith: ['analytical', 'engineer'],
                incompatibleWith: ['casual', 'witty'],
                dominantIn: 'business_tasks'
            },
            'creative': {
                traits: { creativity: 0.95, imagination: 0.9, expressiveness: 0.8, flexibility: 0.8 },
                blendableWith: ['friendly', 'enthusiastic', 'artistic'],
                incompatibleWith: ['analytical', 'professional'],
                dominantIn: 'creative_tasks'
            },
            'analytical': {
                traits: { logic: 0.95, precision: 0.9, objectivity: 0.9, thoroughness: 0.8 },
                blendableWith: ['professional', 'engineer', 'scientist'],
                incompatibleWith: ['creative', 'casual'],
                dominantIn: 'problem_solving'
            },
            'enthusiastic': {
                traits: { energy: 0.95, motivation: 0.9, positivity: 0.9, passion: 0.8 },
                blendableWith: ['friendly', 'creative', 'motivational'],
                incompatibleWith: ['zen', 'philosophical'],
                dominantIn: 'motivation'
            },
            'zen': {
                traits: { calmness: 0.95, mindfulness: 0.9, wisdom: 0.8, balance: 0.9 },
                blendableWith: ['philosophical', 'wise'],
                incompatibleWith: ['enthusiastic', 'energetic'],
                dominantIn: 'stress_relief'
            }
        };

        // Define mood-personality mappings
        this.moodPersonalityMap = {
            'excited': ['enthusiastic', 'creative', 'friendly'],
            'stressed': ['zen', 'supportive', 'calm'],
            'focused': ['analytical', 'professional', 'systematic'],
            'playful': ['witty', 'creative', 'casual'],
            'contemplative': ['philosophical', 'zen', 'wise'],
            'energetic': ['enthusiastic', 'motivational', 'dynamic'],
            'tired': ['gentle', 'supportive', 'understanding'],
            'curious': ['analytical', 'explorer', 'inquisitive'],
            'creative': ['artistic', 'imaginative', 'innovative'],
            'social': ['friendly', 'charismatic', 'engaging']
        };

        // Initialize blending rules
        this.blendingRules = {
            'creative_professional': {
                personalities: ['creative', 'professional'],
                weights: [0.6, 0.4],
                description: 'Innovative yet structured approach',
                triggers: ['design project', 'creative brief', 'innovative solution']
            },
            'friendly_analytical': {
                personalities: ['friendly', 'analytical'],
                weights: [0.5, 0.5],
                description: 'Warm but logical communication',
                triggers: ['explain analysis', 'data presentation', 'technical help']
            },
            'zen_enthusiastic': {
                personalities: ['zen', 'enthusiastic'],
                weights: [0.7, 0.3],
                description: 'Calm energy and mindful motivation',
                triggers: ['motivation', 'encouragement', 'mindful guidance']
            }
        };
    }

    setupEventListeners() {
        // Listen for personality blend requests
        document.addEventListener('personalityBlendRequest', (event) => {
            this.createPersonalityBlend(event.detail.personalities, event.detail.context);
        });

        // Listen for mood detection updates
        document.addEventListener('moodDetected', (event) => {
            this.handleMoodChange(event.detail.mood, event.detail.confidence);
        });

        // Listen for user interaction patterns
        document.addEventListener('userInteractionPattern', (event) => {
            this.analyzeBehaviorPattern(event.detail);
        });
    }

    createBlendingUI() {
        // Create personality blending control panel
        const blendingPanel = document.createElement('div');
        blendingPanel.className = 'personality-blending-panel collapsed';
        blendingPanel.innerHTML = `
            <div class="blending-header" id="personality-blending-header">
                <h3>🎭 Personality Blending</h3>
                <span class="collapse-indicator">▼</span>
            </div>
            
            <div class="blending-content">
                <div class="current-blend-display">
                    <div class="blend-visualization" id="blend-viz">
                        <div class="primary-personality">
                            <span class="personality-name">Friendly</span>
                            <div class="personality-strength" style="width: 70%"></div>
                        </div>
                    </div>
                </div>

                <div class="mood-indicator">
                    <div class="mood-display">
                        <span class="mood-icon">😊</span>
                        <span class="mood-text">Neutral</span>
                        <div class="mood-confidence">85%</div>
                    </div>
                    <div class="mood-history" id="mood-history"></div>
                </div>

                <div class="blending-controls">
                    <div class="personality-selector">
                        <h4>Select Personalities to Blend</h4>
                        <div class="personality-grid" id="personality-grid"></div>
                    </div>
                    
                    <div class="blend-weights">
                        <h4>Blend Weights</h4>
                        <div class="weight-sliders" id="weight-sliders"></div>
                    </div>
                    
                    <div class="context-selector">
                        <h4>Context Awareness</h4>
                        <select id="context-selector">
                            <option value="general">General Conversation</option>
                            <option value="creative_work">Creative Work</option>
                            <option value="problem_solving">Problem Solving</option>
                            <option value="social_interaction">Social Interaction</option>
                            <option value="learning">Learning & Teaching</option>
                            <option value="emotional_support">Emotional Support</option>
                        </select>
                    </div>
                    
                    <div class="blend-actions">
                        <button id="create-blend-btn" class="create-blend-btn">
                            Create Blend
                        </button>
                        <button id="save-preset-btn" class="save-preset-btn">
                            Save Preset
                        </button>
                        <button id="reset-btn" class="reset-btn">
                            Reset
                        </button>
                    </div>
                </div>

                <div class="mood-based-switching">
                    <h4>🧠 Mood-Based Auto-Switching</h4>
                    <div class="auto-switch-toggle">
                        <label class="switch">
                            <input type="checkbox" id="mood-auto-switch" checked>
                            <span class="slider round"></span>
                        </label>
                        <span>Enable Automatic Mood-Based Personality Switching</span>
                    </div>
                    
                    <div class="mood-sensitivity">
                        <label>Mood Sensitivity: <span id="sensitivity-value">70%</span></label>
                        <input type="range" id="mood-sensitivity" min="10" max="100" value="70">
                    </div>
                </div>

                <div class="blend-presets">
                    <h4>🎨 Saved Blends</h4>
                    <div class="preset-grid" id="preset-grid"></div>
                </div>
            </div>
        `;

        // Add to sidebar or create floating panel
        this.insertBlendingPanel(blendingPanel);
        this.setupBlendingEventListeners();
        this.populatePersonalityGrid();
        this.loadSavedPresets();
    }

    setupBlendingEventListeners() {
        // Add click listener for the header to toggle panel
        const header = document.getElementById('personality-blending-header');
        if (header) {
            header.addEventListener('click', () => this.togglePanel());
        }
        
        // Add event listeners for action buttons
        const createBlendBtn = document.getElementById('create-blend-btn');
        if (createBlendBtn) {
            createBlendBtn.addEventListener('click', () => this.createCustomBlend());
        }
        
        const savePresetBtn = document.getElementById('save-preset-btn');
        if (savePresetBtn) {
            savePresetBtn.addEventListener('click', () => this.saveBlendPreset());
        }
        
        const resetBtn = document.getElementById('reset-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetToDefault());
        }
    }

    togglePanel() {
        const panel = document.querySelector('.personality-blending-panel');
        if (panel) {
            panel.classList.toggle('collapsed');
            console.log('🎭 Personality blending panel toggled:', panel.classList.contains('collapsed') ? 'collapsed' : 'expanded');
        } else {
            console.error('❌ Personality blending panel not found');
        }
    }

    insertBlendingPanel(panel) {
        // Try to find sidebar first
        let sidebar = document.querySelector('.sidebar');
        if (sidebar) {
            sidebar.insertBefore(panel, sidebar.firstChild);
        } else {
            // Create floating panel
            panel.style.position = 'fixed';
            panel.style.top = '20px';
            panel.style.right = '20px';
            panel.style.width = '350px';
            panel.style.maxHeight = '80vh';
            panel.style.overflowY = 'auto';
            panel.style.backgroundColor = 'rgba(30, 30, 30, 0.95)';
            panel.style.borderRadius = '15px';
            panel.style.border = '1px solid rgba(100, 255, 200, 0.3)';
            panel.style.padding = '20px';
            panel.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.5)';
            panel.style.backdropFilter = 'blur(20px)';
            panel.style.zIndex = '1000';
            document.body.appendChild(panel);
        }
    }

    populatePersonalityGrid() {
        const grid = document.getElementById('personality-grid');
        if (!grid) {
            console.error('❌ Personality grid element not found');
            return;
        }
        
        const personalities = Object.keys(this.personalityTraits);
        console.log('🎭 Populating personality grid with:', personalities);
        
        if (personalities.length === 0) {
            console.error('❌ No personalities found in personalityTraits');
            return;
        }
        
        grid.innerHTML = personalities.map(personality => `
            <div class="personality-card" data-personality="${personality}">
                <div class="personality-icon">${this.getPersonalityIcon(personality)}</div>
                <div class="personality-name">${personality}</div>
                <div class="personality-traits">
                    ${this.formatTraits(this.personalityTraits[personality].traits)}
                </div>
                <div class="selection-checkbox">
                    <input type="checkbox" id="select-${personality}" value="${personality}">
                </div>
            </div>
        `).join('');
        
        console.log('✅ Personality grid populated with', personalities.length, 'personalities');
    }

    createPersonalityBlend(personalities, context = 'general') {
        if (!personalities || personalities.length < 2) {
            console.warn('Need at least 2 personalities to create a blend');
            return null;
        }

        // Calculate optimal weights based on context and compatibility
        const weights = this.calculateOptimalWeights(personalities, context);
        
        const blend = {
            id: `blend_${Date.now()}`,
            personalities: personalities,
            weights: weights,
            context: context,
            traits: this.blendTraits(personalities, weights),
            description: this.generateBlendDescription(personalities, weights),
            createdAt: new Date(),
            effectiveness: this.calculateBlendEffectiveness(personalities, context)
        };

        this.currentBlend = blend;
        this.updateBlendVisualization(blend);
        this.applyPersonalityBlend(blend);
        
        console.log('🎭 Created personality blend:', blend);
        return blend;
    }

    calculateOptimalWeights(personalities, context) {
        const weights = new Array(personalities.length).fill(1 / personalities.length);
        
        // Adjust weights based on context relevance
        personalities.forEach((personality, index) => {
            const traits = this.personalityTraits[personality];
            if (traits.dominantIn === context) {
                weights[index] *= 1.5; // Boost dominant personality
            }
        });

        // Normalize weights
        const sum = weights.reduce((a, b) => a + b, 0);
        return weights.map(w => w / sum);
    }

    blendTraits(personalities, weights) {
        const blendedTraits = {};
        const allTraitNames = new Set();
        
        // Collect all trait names
        personalities.forEach(personality => {
            const traits = this.personalityTraits[personality].traits;
            Object.keys(traits).forEach(trait => allTraitNames.add(trait));
        });

        // Blend traits using weighted average
        allTraitNames.forEach(traitName => {
            let weightedSum = 0;
            let totalWeight = 0;
            
            personalities.forEach((personality, index) => {
                const traits = this.personalityTraits[personality].traits;
                if (traits[traitName]) {
                    weightedSum += traits[traitName] * weights[index];
                    totalWeight += weights[index];
                }
            });
            
            if (totalWeight > 0) {
                blendedTraits[traitName] = Math.min(1.0, weightedSum / totalWeight);
            }
        });

        return blendedTraits;
    }

    handleMoodChange(newMood, confidence) {
        console.log(`🧠 Mood detected: ${newMood} (${confidence}% confidence)`);
        
        // Update mood state
        this.moodState = newMood;
        this.moodHistory.push({
            mood: newMood,
            confidence: confidence,
            timestamp: Date.now()
        });

        // Keep only last 10 mood changes
        if (this.moodHistory.length > 10) {
            this.moodHistory.shift();
        }

        // Update UI
        this.updateMoodDisplay(newMood, confidence);

        // Check if auto-switching is enabled
        const autoSwitch = document.getElementById('mood-auto-switch');
        if (autoSwitch && autoSwitch.checked) {
            const sensitivity = document.getElementById('mood-sensitivity').value;
            if (confidence >= sensitivity) {
                this.switchToMoodBasedPersonality(newMood);
            }
        }
    }

    switchToMoodBasedPersonality(mood) {
        const recommendedPersonalities = this.moodPersonalityMap[mood];
        if (!recommendedPersonalities || recommendedPersonalities.length === 0) {
            return;
        }

        // Select the best personality for this mood
        const selectedPersonality = recommendedPersonalities[0];
        
        // Create a mood-enhanced blend
        const moodBlend = this.createMoodEnhancedBlend(selectedPersonality, mood);
        
        // Notify the main AI system
        this.notifyPersonalityChange(moodBlend);
    }

    createMoodEnhancedBlend(basePersonality, mood) {
        const moodModifiers = {
            'excited': { energy: 1.2, enthusiasm: 1.3, expressiveness: 1.2 },
            'stressed': { calmness: 1.4, supportiveness: 1.3, patience: 1.2 },
            'focused': { precision: 1.3, efficiency: 1.2, structure: 1.2 },
            'playful': { humor: 1.4, creativity: 1.2, flexibility: 1.3 },
            'contemplative': { wisdom: 1.3, thoughtfulness: 1.4, depth: 1.2 }
        };

        const modifier = moodModifiers[mood] || {};
        const baseTraits = this.personalityTraits[basePersonality]?.traits || {};
        const enhancedTraits = { ...baseTraits };

        // Apply mood modifiers
        Object.keys(modifier).forEach(trait => {
            if (enhancedTraits[trait]) {
                enhancedTraits[trait] = Math.min(1.0, enhancedTraits[trait] * modifier[trait]);
            }
        });

        return {
            id: `mood_${mood}_${Date.now()}`,
            personalities: [basePersonality],
            weights: [1.0],
            mood: mood,
            traits: enhancedTraits,
            description: `${basePersonality} enhanced for ${mood} mood`,
            isMoodBased: true
        };
    }

    updateBlendVisualization(blend) {
        const vizElement = document.getElementById('blend-viz');
        if (!vizElement) return;

        const visualizationHTML = blend.personalities.map((personality, index) => `
            <div class="blend-component">
                <div class="personality-info">
                    <span class="personality-icon">${this.getPersonalityIcon(personality)}</span>
                    <span class="personality-name">${personality}</span>
                </div>
                <div class="personality-strength" 
                     style="width: ${blend.weights[index] * 100}%; 
                            background: ${this.getPersonalityColor(personality)}">
                </div>
                <span class="weight-percentage">${Math.round(blend.weights[index] * 100)}%</span>
            </div>
        `).join('');

        vizElement.innerHTML = `
            <div class="blend-header">
                <h4>Current Blend: ${blend.description}</h4>
                <div class="effectiveness-score">
                    Effectiveness: ${Math.round(blend.effectiveness * 100)}%
                </div>
            </div>
            <div class="blend-components">
                ${visualizationHTML}
            </div>
        `;
    }

    updateMoodDisplay(mood, confidence) {
        const moodIcon = document.querySelector('.mood-icon');
        const moodText = document.querySelector('.mood-text');
        const moodConfidence = document.querySelector('.mood-confidence');

        if (moodIcon) moodIcon.textContent = this.getMoodIcon(mood);
        if (moodText) moodText.textContent = mood.charAt(0).toUpperCase() + mood.slice(1);
        if (moodConfidence) moodConfidence.textContent = `${Math.round(confidence)}%`;

        // Update mood history visualization
        this.updateMoodHistory();
    }

    updateMoodHistory() {
        const historyElement = document.getElementById('mood-history');
        if (!historyElement) return;

        const historyHTML = this.moodHistory.slice(-5).map(entry => `
            <div class="mood-entry" title="${new Date(entry.timestamp).toLocaleTimeString()}">
                <span class="mood-icon">${this.getMoodIcon(entry.mood)}</span>
                <span class="confidence">${Math.round(entry.confidence)}%</span>
            </div>
        `).join('');

        historyElement.innerHTML = historyHTML;
    }

    applyPersonalityBlend(blend) {
        // Send blend configuration to the main AI system
        if (window.aiAssistant && typeof window.aiAssistant.updatePersonality === 'function') {
            window.aiAssistant.updatePersonality(blend);
        }

        // Store current blend for persistence
        localStorage.setItem('currentPersonalityBlend', JSON.stringify(blend));

        // Trigger event for other systems
        document.dispatchEvent(new CustomEvent('personalityBlendApplied', {
            detail: { blend }
        }));
    }

    startMoodMonitoring() {
        // Monitor user input patterns for mood detection
        setInterval(() => {
            this.analyzeMoodFromRecentInteractions();
        }, 30000); // Check every 30 seconds
    }

    analyzeMoodFromRecentInteractions() {
        // Simple mood analysis based on recent messages
        const recentMessages = this.getRecentMessages();
        if (recentMessages.length === 0) return;

        const moodIndicators = this.extractMoodIndicators(recentMessages);
        const detectedMood = this.classifyMood(moodIndicators);
        
        if (detectedMood.confidence > 0.6) {
            document.dispatchEvent(new CustomEvent('moodDetected', {
                detail: {
                    mood: detectedMood.mood,
                    confidence: detectedMood.confidence * 100
                }
            }));
        }
    }

    // Utility methods
    getPersonalityIcon(personality) {
        const icons = {
            friendly: '😊', professional: '💼', creative: '🎨', analytical: '📊',
            enthusiastic: '⚡', zen: '🧘', witty: '😄', casual: '😎',
            scientist: '🔬', philosopher: '🤔', engineer: '⚙️', writer: '✍️'
        };
        return icons[personality] || '🤖';
    }

    formatTraits(traits) {
        return Object.keys(traits).slice(0, 3).map(trait => 
            trait.charAt(0).toUpperCase() + trait.slice(1)
        ).join(', ');
    }

    getMoodIcon(mood) {
        const icons = {
            excited: '🎉', stressed: '😰', focused: '🎯', playful: '🎪',
            contemplative: '🤔', energetic: '⚡', tired: '😴', curious: '🔍',
            creative: '🎨', social: '🤝', neutral: '😊'
        };
        return icons[mood] || '😊';
    }

    getPersonalityColor(personality) {
        const colors = {
            friendly: '#4CAF50', professional: '#2196F3', creative: '#FF9800',
            analytical: '#9C27B0', enthusiastic: '#FF5722', zen: '#009688',
            witty: '#FFEB3B', casual: '#795548'
        };
        return colors[personality] || '#64FFCC';
    }

    // Additional methods would be implemented here...
    
    createCustomBlend() {
        const selectedPersonalities = Array.from(document.querySelectorAll('#personality-grid input:checked'))
            .map(checkbox => checkbox.value);
        
        if (selectedPersonalities.length < 2) {
            alert('Please select at least 2 personalities to blend');
            return;
        }

        const context = document.getElementById('context-selector').value;
        this.createPersonalityBlend(selectedPersonalities, context);
    }

    saveBlendPreset() {
        if (!this.currentBlend) {
            alert('No active blend to save');
            return;
        }

        const name = prompt('Enter a name for this blend preset:');
        if (name) {
            const presets = JSON.parse(localStorage.getItem('personalityBlendPresets') || '{}');
            presets[name] = this.currentBlend;
            localStorage.setItem('personalityBlendPresets', JSON.stringify(presets));
            this.loadSavedPresets();
        }
    }

    resetToDefault() {
        // Clear all selections
        const checkboxes = document.querySelectorAll('#personality-grid input[type="checkbox"]');
        checkboxes.forEach(checkbox => checkbox.checked = false);
        
        // Reset to default friendly personality
        this.currentBlend = null;
        this.personalityWeights = { friendly: 1.0 };
        
        // Update visualization
        this.updateBlendVisualization({
            personalities: ['friendly'],
            weights: [1.0],
            description: 'Default Friendly Assistant',
            effectiveness: 1.0
        });
        
        console.log('🔄 Reset to default personality configuration');
    }

    loadSavedPresets() {
        const presets = JSON.parse(localStorage.getItem('personalityBlendPresets') || '{}');
        const presetGrid = document.getElementById('preset-grid');
        
        if (presetGrid) {
            presetGrid.innerHTML = Object.keys(presets).map(name => `
                <div class="preset-card" onclick="personalityBlending.loadPreset('${name}')">
                    <div class="preset-name">${name}</div>
                    <div class="preset-description">${presets[name].description}</div>
                </div>
            `).join('');
        }
    }

    loadPreset(name) {
        const presets = JSON.parse(localStorage.getItem('personalityBlendPresets') || '{}');
        const preset = presets[name];
        
        if (preset) {
            this.currentBlend = preset;
            this.updateBlendVisualization(preset);
            console.log('✅ Loaded preset:', name);
        }
    }
}

// Initialize the personality blending system
window.personalityBlending = new PersonalityBlendingSystem();

// Expose for global access
window.PersonalityBlendingSystem = PersonalityBlendingSystem;