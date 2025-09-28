// Contextual Intelligence System
// Location-based responses and Time-aware suggestions

class ContextualIntelligenceSystem {
    constructor(aiAssistant) {
        this.aiAssistant = aiAssistant;
        this.locationData = null;
        this.timeContext = null;
        this.weatherData = null;
        this.userTimezone = null;
        this.locationPermissionGranted = false;
        this.contextualSuggestions = [];
        this.lastLocationUpdate = null;
        this.locationUpdateInterval = null;
        
        this.init();
    }
    
    init() {
        console.log('🌍 Initializing Contextual Intelligence System...');
        
        this.detectTimezone();
        this.requestLocationPermission();
        this.setupTimeContextMonitoring();
        this.addContextualUI();
        this.startPeriodicUpdates();
        
        console.log('🧠 Contextual Intelligence System activated');
    }
    
    // ===== LOCATION-BASED RESPONSES =====
    
    async requestLocationPermission() {
        if (!navigator.geolocation) {
            console.log('❌ Geolocation not supported by browser');
            return;
        }
        
        try {
            // First try to get position to trigger permission request
            await this.getCurrentPosition();
            this.locationPermissionGranted = true;
            console.log('✅ Location permission granted');
            
            this.setupLocationTracking();
            
        } catch (error) {
            console.log('⚠️ Location permission denied or unavailable:', error.message);
            this.showLocationPermissionPrompt();
        }
    }
    
    getCurrentPosition() {
        return new Promise((resolve, reject) => {
            navigator.geolocation.getCurrentPosition(
                position => resolve(position),
                error => reject(error),
                {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 300000 // 5 minutes
                }
            );
        });
    }
    
    async setupLocationTracking() {
        try {
            const position = await this.getCurrentPosition();
            await this.updateLocationData(position);
            
            // Set up location monitoring (update every 5 minutes if user moves)
            this.locationUpdateInterval = setInterval(async () => {
                try {
                    const newPosition = await this.getCurrentPosition();
                    const distance = this.calculateDistance(
                        this.locationData.coordinates.latitude,
                        this.locationData.coordinates.longitude,
                        newPosition.coords.latitude,
                        newPosition.coords.longitude
                    );
                    
                    // Update if moved more than 100 meters
                    if (distance > 0.1) {
                        await this.updateLocationData(newPosition);
                    }
                } catch (error) {
                    console.log('Location update failed:', error.message);
                }
            }, 300000); // 5 minutes
            
        } catch (error) {
            console.error('Failed to setup location tracking:', error);
        }
    }
    
    async updateLocationData(position) {
        this.locationData = {
            coordinates: {
                latitude: position.coords.latitude,
                longitude: position.coords.longitude,
                accuracy: position.coords.accuracy
            },
            timestamp: new Date().toISOString(),
            address: await this.reverseGeocode(position.coords.latitude, position.coords.longitude),
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
        };
        
        this.lastLocationUpdate = new Date();
        
        // Get weather data for the location
        await this.updateWeatherData();
        
        // Generate location-based suggestions
        this.generateLocationBasedSuggestions();
        
        console.log('📍 Location updated:', this.locationData.address);
        
        // Store location context for memory system
        if (this.aiAssistant && this.aiAssistant.memorySystem) {
            this.aiAssistant.memorySystem.storePersistentContext({
                context_type: 'location_context',
                context_key: 'current_location',
                context_value: this.locationData,
                importance_score: 0.6,
                decay_rate: 0.1
            });
        }
    }
    
    async reverseGeocode(lat, lon) {
        try {
            // Using a free geocoding service (you may want to use a more robust service)
            const response = await fetch(`https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lon}&localityLanguage=en`);
            const data = await response.json();
            
            return {
                city: data.city || data.locality || 'Unknown',
                region: data.principalSubdivision || data.region || 'Unknown',
                country: data.countryName || 'Unknown',
                full_address: data.localityInfo?.informative || `${data.city || 'Unknown'}, ${data.countryName || 'Unknown'}`
            };
        } catch (error) {
            console.error('Reverse geocoding failed:', error);
            return {
                city: 'Unknown',
                region: 'Unknown', 
                country: 'Unknown',
                full_address: 'Location unavailable'
            };
        }
    }
    
    async updateWeatherData() {
        if (!this.locationData) return;
        
        try {
            // Note: In production, you'd use a proper weather API like OpenWeatherMap
            // For demo, we'll simulate weather data
            this.weatherData = {
                temperature: Math.round(15 + Math.random() * 20), // 15-35°C
                condition: this.getRandomWeatherCondition(),
                humidity: Math.round(40 + Math.random() * 40),
                wind_speed: Math.round(Math.random() * 20),
                timestamp: new Date().toISOString(),
                location: this.locationData.address.city
            };
            
            console.log('🌤️ Weather data updated:', this.weatherData);
            
        } catch (error) {
            console.error('Weather update failed:', error);
        }
    }
    
    getRandomWeatherCondition() {
        const conditions = ['sunny', 'cloudy', 'partly-cloudy', 'rainy', 'drizzle'];
        return conditions[Math.floor(Math.random() * conditions.length)];
    }
    
    calculateDistance(lat1, lon1, lat2, lon2) {
        const R = 6371; // Earth's radius in kilometers
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                  Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                  Math.sin(dLon / 2) * Math.sin(dLon / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c; // Distance in kilometers
    }
    
    generateLocationBasedSuggestions() {
        if (!this.locationData) return;
        
        const suggestions = [];
        const city = this.locationData.address.city;
        const country = this.locationData.address.country;
        
        // Location-specific suggestions
        suggestions.push({
            type: 'location_info',
            title: `About ${city}`,
            suggestion: `Tell me interesting facts about ${city}`,
            priority: 0.7,
            category: 'local_knowledge'
        });
        
        suggestions.push({
            type: 'local_search',
            title: 'Find Nearby',
            suggestion: `Find restaurants near me in ${city}`,
            priority: 0.8,
            category: 'local_services'
        });
        
        if (this.weatherData) {
            suggestions.push({
                type: 'weather_advice',
                title: 'Weather-Based Advice',
                suggestion: `Give me outfit suggestions for ${this.weatherData.condition} weather`,
                priority: 0.6,
                category: 'weather_context'
            });
        }
        
        // Country-specific suggestions
        if (country !== 'Unknown') {
            suggestions.push({
                type: 'cultural_info',
                title: `${country} Culture`,
                suggestion: `Tell me about cultural customs in ${country}`,
                priority: 0.5,
                category: 'cultural_context'
            });
        }
        
        this.contextualSuggestions = [...this.contextualSuggestions, ...suggestions];
        this.updateSuggestionsUI();
    }
    
    // ===== TIME-AWARE SUGGESTIONS =====
    
    detectTimezone() {
        this.userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
        console.log('🕐 User timezone detected:', this.userTimezone);
    }
    
    setupTimeContextMonitoring() {
        this.updateTimeContext();
        
        // Update time context every minute
        setInterval(() => {
            this.updateTimeContext();
        }, 60000);
    }
    
    updateTimeContext() {
        const now = new Date();
        const hour = now.getHours();
        const dayOfWeek = now.getDay();
        const dayOfMonth = now.getDate();
        const month = now.getMonth();
        
        this.timeContext = {
            current_time: now.toISOString(),
            local_time: now.toLocaleTimeString(),
            hour: hour,
            minute: now.getMinutes(),
            day_of_week: dayOfWeek,
            day_name: now.toLocaleDateString('en-US', { weekday: 'long' }),
            day_of_month: dayOfMonth,
            month: month,
            month_name: now.toLocaleDateString('en-US', { month: 'long' }),
            year: now.getFullYear(),
            timezone: this.userTimezone,
            period: this.getTimeOfDay(hour),
            is_weekend: dayOfWeek === 0 || dayOfWeek === 6,
            is_work_hours: hour >= 9 && hour <= 17 && dayOfWeek >= 1 && dayOfWeek <= 5
        };
        
        this.generateTimeBasedSuggestions();
    }
    
    getTimeOfDay(hour) {
        if (hour >= 5 && hour < 12) return 'morning';
        if (hour >= 12 && hour < 17) return 'afternoon';
        if (hour >= 17 && hour < 21) return 'evening';
        return 'night';
    }
    
    generateTimeBasedSuggestions() {
        if (!this.timeContext) return;
        
        const suggestions = [];
        const { hour, period, is_weekend, is_work_hours, day_name } = this.timeContext;
        
        // Time-of-day specific suggestions
        switch (period) {
            case 'morning':
                suggestions.push({
                    type: 'morning_routine',
                    title: 'Morning Boost',
                    suggestion: 'Give me a motivational quote to start my day',
                    priority: 0.8,
                    category: 'time_context'
                });
                
                if (!is_weekend) {
                    suggestions.push({
                        type: 'work_prep',
                        title: 'Work Preparation',
                        suggestion: 'Help me plan my work priorities for today',
                        priority: 0.9,
                        category: 'productivity'
                    });
                }
                break;
                
            case 'afternoon':
                if (is_work_hours) {
                    suggestions.push({
                        type: 'productivity_break',
                        title: 'Productivity Break',
                        suggestion: 'Suggest a quick mental break activity',
                        priority: 0.7,
                        category: 'wellness'
                    });
                } else {
                    suggestions.push({
                        type: 'afternoon_activity',
                        title: 'Afternoon Plans',
                        suggestion: 'Suggest activities for this afternoon',
                        priority: 0.6,
                        category: 'leisure'
                    });
                }
                break;
                
            case 'evening':
                suggestions.push({
                    type: 'evening_wind_down',
                    title: 'Evening Relaxation',
                    suggestion: 'Recommend relaxation techniques for tonight',
                    priority: 0.7,
                    category: 'wellness'
                });
                
                suggestions.push({
                    type: 'dinner_ideas',
                    title: 'Dinner Inspiration',
                    suggestion: 'Suggest healthy dinner ideas',
                    priority: 0.8,
                    category: 'food'
                });
                break;
                
            case 'night':
                suggestions.push({
                    type: 'sleep_preparation',
                    title: 'Better Sleep',
                    suggestion: 'Give me tips for better sleep tonight',
                    priority: 0.9,
                    category: 'health'
                });
                break;
        }
        
        // Day-specific suggestions
        if (day_name === 'Monday') {
            suggestions.push({
                type: 'monday_motivation',
                title: 'Monday Motivation',
                suggestion: 'Help me tackle this Monday with energy',
                priority: 0.8,
                category: 'motivation'
            });
        }
        
        if (is_weekend) {
            suggestions.push({
                type: 'weekend_activities',
                title: 'Weekend Fun',
                suggestion: `Suggest fun activities for ${day_name}`,
                priority: 0.7,
                category: 'leisure'
            });
        }
        
        // Season-based suggestions (based on month)
        const season = this.getSeason(this.timeContext.month);
        suggestions.push({
            type: 'seasonal_activity',
            title: `${season} Activities`,
            suggestion: `Recommend ${season.toLowerCase()} activities`,
            priority: 0.5,
            category: 'seasonal'
        });
        
        this.contextualSuggestions = [...this.contextualSuggestions, ...suggestions];
        this.updateSuggestionsUI();
    }
    
    getSeason(month) {
        if (month >= 2 && month <= 4) return 'Spring';
        if (month >= 5 && month <= 7) return 'Summer';
        if (month >= 8 && month <= 10) return 'Fall';
        return 'Winter';
    }
    
    // ===== CONTEXTUAL RESPONSE ENHANCEMENT =====
    
    enhanceResponseWithContext(userInput, aiResponse) {
        let enhancedResponse = aiResponse;
        
        // Add location context if relevant
        if (this.shouldAddLocationContext(userInput)) {
            enhancedResponse = this.addLocationContext(enhancedResponse);
        }
        
        // Add time context if relevant
        if (this.shouldAddTimeContext(userInput)) {
            enhancedResponse = this.addTimeContext(enhancedResponse);
        }
        
        // Add weather context if relevant
        if (this.shouldAddWeatherContext(userInput)) {
            enhancedResponse = this.addWeatherContext(enhancedResponse);
        }
        
        return enhancedResponse;
    }
    
    shouldAddLocationContext(userInput) {
        const locationKeywords = ['near me', 'local', 'where', 'nearby', 'around here', 'in my area'];
        return locationKeywords.some(keyword => userInput.toLowerCase().includes(keyword));
    }
    
    shouldAddTimeContext(userInput) {
        const timeKeywords = ['today', 'tonight', 'tomorrow', 'this morning', 'schedule', 'plan', 'when'];
        return timeKeywords.some(keyword => userInput.toLowerCase().includes(keyword));
    }
    
    shouldAddWeatherContext(userInput) {
        const weatherKeywords = ['weather', 'outside', 'rain', 'sunny', 'cold', 'hot', 'outfit', 'clothes'];
        return weatherKeywords.some(keyword => userInput.toLowerCase().includes(keyword));
    }
    
    addLocationContext(response) {
        if (!this.locationData) return response;
        
        const locationInfo = ` (Based on your location in ${this.locationData.address.city}, ${this.locationData.address.country})`;
        return response + locationInfo;
    }
    
    addTimeContext(response) {
        if (!this.timeContext) return response;
        
        const timeInfo = ` (Current time: ${this.timeContext.local_time}, ${this.timeContext.day_name})`;
        return response + timeInfo;
    }
    
    addWeatherContext(response) {
        if (!this.weatherData) return response;
        
        const weatherInfo = ` (Current weather in ${this.weatherData.location}: ${this.weatherData.condition}, ${this.weatherData.temperature}°C)`;
        return response + weatherInfo;
    }
    
    // ===== UI AND INTERACTION =====
    
    addContextualUI() {
        this.addContextIndicators();
        this.addSuggestionsPanel();
        this.addLocationPrompt();
    }
    
    addContextIndicators() {
        const indicator = document.createElement('div');
        indicator.id = 'contextualIndicator';
        indicator.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            z-index: 1000;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            cursor: pointer;
            transition: all 0.3s ease;
        `;
        indicator.innerHTML = '🌍⏰ Context Aware';
        indicator.title = 'Location & Time Intelligence Active';
        
        indicator.onclick = () => this.toggleContextPanel();
        
        document.body.appendChild(indicator);
    }
    
    addSuggestionsPanel() {
        const panel = document.createElement('div');
        panel.id = 'contextualSuggestions';
        panel.className = 'contextual-panel';
        panel.style.cssText = `
            position: fixed;
            top: 70px;
            right: 20px;
            width: 300px;
            max-height: 400px;
            background: rgba(26, 26, 46, 0.95);
            border: 2px solid #667eea;
            border-radius: 12px;
            padding: 20px;
            color: white;
            font-size: 0.9em;
            z-index: 10000;
            overflow-y: auto;
            backdrop-filter: blur(10px);
            display: none;
        `;
        
        panel.innerHTML = `
            <h4 style="color: #667eea; margin: 0 0 15px 0; text-align: center;">🌍 Contextual Suggestions</h4>
            <div id="locationContext" style="margin-bottom: 15px;">
                <h5 style="color: #64ffda; margin-bottom: 8px;">📍 Location</h5>
                <div id="locationInfo">Detecting location...</div>
            </div>
            <div id="timeContext" style="margin-bottom: 15px;">
                <h5 style="color: #ffd54f; margin-bottom: 8px;">⏰ Time Context</h5>
                <div id="timeInfo">Loading time context...</div>
            </div>
            <div id="suggestedActions">
                <h5 style="color: #ff8a65; margin-bottom: 8px;">💡 Smart Suggestions</h5>
                <div id="suggestionsList"></div>
            </div>
        `;
        
        document.body.appendChild(panel);
        
        // Add styles
        const style = document.createElement('style');
        style.textContent = `
            .contextual-suggestion {
                background: rgba(102, 126, 234, 0.1);
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 8px;
                padding: 10px;
                margin: 8px 0;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            
            .contextual-suggestion:hover {
                background: rgba(102, 126, 234, 0.2);
                transform: translateX(5px);
            }
            
            .suggestion-title {
                font-weight: bold;
                color: #667eea;
                margin-bottom: 4px;
            }
            
            .suggestion-text {
                font-size: 0.85em;
                opacity: 0.9;
            }
        `;
        document.head.appendChild(style);
    }
    
    addLocationPrompt() {
        if (this.locationPermissionGranted) return;
        
        const prompt = document.createElement('div');
        prompt.id = 'locationPrompt';
        prompt.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(102, 126, 234, 0.95);
            color: white;
            padding: 15px 20px;
            border-radius: 12px;
            max-width: 300px;
            z-index: 10000;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        `;
        
        prompt.innerHTML = `
            <h5 style="margin: 0 0 8px 0;">🌍 Enable Location Intelligence</h5>
            <p style="margin: 0 0 12px 0; font-size: 0.9em;">Allow location access for personalized suggestions and local context.</p>
            <div style="text-align: right;">
                <button onclick="this.parentElement.parentElement.remove()" 
                        style="background: rgba(255,255,255,0.2); border: none; color: white; padding: 6px 12px; border-radius: 4px; margin-right: 8px; cursor: pointer;">
                    Later
                </button>
                <button onclick="window.contextualIntelligence.requestLocationPermission(); this.parentElement.parentElement.remove()" 
                        style="background: #4CAF50; border: none; color: white; padding: 6px 12px; border-radius: 4px; cursor: pointer;">
                    Enable
                </button>
            </div>
        `;
        
        document.body.appendChild(prompt);
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            if (prompt.parentNode) {
                prompt.remove();
            }
        }, 10000);
    }
    
    showLocationPermissionPrompt() {
        this.addLocationPrompt();
    }
    
    toggleContextPanel() {
        const panel = document.getElementById('contextualSuggestions');
        if (panel) {
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
            if (panel.style.display === 'block') {
                this.updateContextUI();
            }
        }
    }
    
    updateContextUI() {
        this.updateLocationUI();
        this.updateTimeUI();
        this.updateSuggestionsUI();
    }
    
    updateLocationUI() {
        const locationInfo = document.getElementById('locationInfo');
        if (!locationInfo) return;
        
        if (this.locationData) {
            locationInfo.innerHTML = `
                <div style="font-size: 0.85em;">
                    <div>📍 ${this.locationData.address.full_address}</div>
                    ${this.weatherData ? `<div style="margin-top: 4px;">🌤️ ${this.weatherData.temperature}°C, ${this.weatherData.condition}</div>` : ''}
                </div>
            `;
        } else {
            locationInfo.innerHTML = '<div style="font-size: 0.85em; opacity: 0.7;">Location not available</div>';
        }
    }
    
    updateTimeUI() {
        const timeInfo = document.getElementById('timeInfo');
        if (!timeInfo || !this.timeContext) return;
        
        timeInfo.innerHTML = `
            <div style="font-size: 0.85em;">
                <div>🕐 ${this.timeContext.local_time}</div>
                <div style="margin-top: 4px;">📅 ${this.timeContext.day_name}, ${this.timeContext.period}</div>
                ${this.timeContext.is_weekend ? '<div style="margin-top: 4px; color: #4CAF50;">🎉 Weekend!</div>' : ''}
            </div>
        `;
    }
    
    updateSuggestionsUI() {
        const suggestionsList = document.getElementById('suggestionsList');
        if (!suggestionsList) return;
        
        suggestionsList.innerHTML = '';
        
        // Get top 4 suggestions by priority
        const topSuggestions = [...this.contextualSuggestions]
            .sort((a, b) => b.priority - a.priority)
            .slice(0, 4);
        
        topSuggestions.forEach(suggestion => {
            const suggestionElement = document.createElement('div');
            suggestionElement.className = 'contextual-suggestion';
            suggestionElement.innerHTML = `
                <div class="suggestion-title">${suggestion.title}</div>
                <div class="suggestion-text">${suggestion.suggestion}</div>
            `;
            
            suggestionElement.onclick = () => {
                if (this.aiAssistant && this.aiAssistant.voiceInput) {
                    this.aiAssistant.voiceInput.value = suggestion.suggestion;
                    this.aiAssistant.sendMessage();
                }
                this.toggleContextPanel();
            };
            
            suggestionsList.appendChild(suggestionElement);
        });
        
        // Clear old suggestions periodically
        if (this.contextualSuggestions.length > 20) {
            this.contextualSuggestions = this.contextualSuggestions.slice(-10);
        }
    }
    
    startPeriodicUpdates() {
        // Update suggestions every 5 minutes
        setInterval(() => {
            this.contextualSuggestions = []; // Clear old suggestions
            this.generateTimeBasedSuggestions();
            
            if (this.locationData) {
                this.generateLocationBasedSuggestions();
            }
        }, 300000); // 5 minutes
    }
    
    // ===== PUBLIC API =====
    
    getLocationContext() {
        return this.locationData;
    }
    
    getTimeContext() {
        return this.timeContext;
    }
    
    getWeatherContext() {
        return this.weatherData;
    }
    
    getContextualSuggestions(category = null) {
        if (!category) return this.contextualSuggestions;
        return this.contextualSuggestions.filter(s => s.category === category);
    }
    
    // Integration with AI assistant
    enhanceUserMessage(message) {
        let enhancedMessage = message;
        
        // Add implicit context if user asks about location/time sensitive queries
        if (this.shouldAddLocationContext(message) && this.locationData) {
            enhancedMessage += ` [User location: ${this.locationData.address.city}, ${this.locationData.address.country}]`;
        }
        
        if (this.shouldAddTimeContext(message) && this.timeContext) {
            enhancedMessage += ` [Current time: ${this.timeContext.period} on ${this.timeContext.day_name}]`;
        }
        
        if (this.shouldAddWeatherContext(message) && this.weatherData) {
            enhancedMessage += ` [Weather: ${this.weatherData.condition}, ${this.weatherData.temperature}°C]`;
        }
        
        return enhancedMessage;
    }
    
    cleanup() {
        if (this.locationUpdateInterval) {
            clearInterval(this.locationUpdateInterval);
        }
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        if (window.aiAssistant) {
            window.contextualIntelligence = new ContextualIntelligenceSystem(window.aiAssistant);
            console.log('🚀 Contextual Intelligence system loaded successfully!');
        }
    }, 2500);
});