/**
 * Horizon Advanced Analytics & Session Management System
 * 
 * This module provides:
 * - Session persistence and management
 * - Chat history tracking
 * - Usage analytics and tracking
 * - Performance monitoring
 * - Heatmap tracking
 * - A/B testing framework
 */

class HorizonAnalyticsManager {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.userId = this.getUserId();
        this.sessionStartTime = Date.now();
        this.lastActivity = Date.now();
        this.chatHistory = this.loadChatHistory();
        this.analyticsData = this.loadAnalyticsData();
        this.performanceMetrics = {};
        this.heatmapData = [];
        this.abTestVariant = this.determineABTestVariant();
        
        this.init();
    }

    init() {
        this.initializeSession();
        this.startPerformanceMonitoring();
        this.initializeHeatmapTracking();
        this.setupAnalyticsTracking();
        this.updateSessionUI();
        this.loadChatHistoryUI();
        
        console.log('🔬 Analytics & Session Manager initialized');
        console.log(`📊 Session ID: ${this.sessionId}`);
        console.log(`👤 User ID: ${this.userId}`);
        console.log(`🧪 A/B Test Variant: ${this.abTestVariant}`);
    }

    // ===== SESSION MANAGEMENT =====

    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now().toString(36);
    }

    getUserId() {
        let userId = localStorage.getItem('horizon_user_id');
        if (!userId) {
            userId = 'user_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now().toString(36);
            localStorage.setItem('horizon_user_id', userId);
        }
        return userId;
    }

    initializeSession() {
        // Update session metadata
        const sessionData = {
            sessionId: this.sessionId,
            userId: this.userId,
            startTime: this.sessionStartTime,
            lastActivity: this.lastActivity,
            userAgent: navigator.userAgent,
            referrer: document.referrer,
            abTestVariant: this.abTestVariant
        };

        // Store session in localStorage for persistence
        localStorage.setItem('horizon_current_session', JSON.stringify(sessionData));
        
        // Add to session history
        this.addToSessionHistory(sessionData);
        
        this.trackEvent('session_started', sessionData);
    }

    updateLastActivity() {
        this.lastActivity = Date.now();
        this.updateSessionUI();
    }

    updateSessionUI() {
        const sessionStatus = document.getElementById('sessionStatus');
        const sessionIdEl = document.getElementById('sessionId');
        const lastActivityEl = document.getElementById('lastActivity');

        if (sessionStatus) {
            sessionStatus.textContent = 'Connected';
            sessionStatus.className = 'session-connected';
        }

        if (sessionIdEl) {
            sessionIdEl.textContent = this.sessionId.substr(-8);
            sessionIdEl.title = this.sessionId;
        }

        if (lastActivityEl) {
            const timeSince = this.formatTimeSince(this.lastActivity);
            lastActivityEl.textContent = timeSince;
        }
    }

    formatTimeSince(timestamp) {
        const seconds = Math.floor((Date.now() - timestamp) / 1000);
        if (seconds < 60) return 'Just now';
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        return `${Math.floor(seconds / 86400)}d ago`;
    }

    // ===== CHAT HISTORY MANAGEMENT =====

    loadChatHistory() {
        return JSON.parse(localStorage.getItem('horizon_chat_history') || '[]');
    }

    saveChatHistory() {
        localStorage.setItem('horizon_chat_history', JSON.stringify(this.chatHistory));
    }

    addToChatHistory(message, response, metadata = {}) {
        const chatEntry = {
            id: 'chat_' + Date.now(),
            sessionId: this.sessionId,
            userId: this.userId,
            timestamp: Date.now(),
            message: message,
            response: response,
            metadata: {
                personality: metadata.personality || 'friendly',
                responseTime: metadata.responseTime || 0,
                intent: metadata.intent || 'general',
                ...metadata
            }
        };

        this.chatHistory.unshift(chatEntry); // Add to beginning
        
        // Keep only last 100 conversations
        if (this.chatHistory.length > 100) {
            this.chatHistory = this.chatHistory.slice(0, 100);
        }

        this.saveChatHistory();
        this.updateChatHistoryUI();
        this.trackEvent('message_sent', {
            message_length: message.length,
            response_length: response.length,
            ...metadata
        });
    }

    loadChatHistoryUI() {
        const historyList = document.getElementById('chatHistoryList');
        if (!historyList) return;

        if (this.chatHistory.length === 0) {
            historyList.innerHTML = '<div class="empty-state">No previous conversations</div>';
            return;
        }

        const historyHTML = this.chatHistory.slice(0, 10).map(chat => `
            <div class="history-item" onclick="loadChatFromHistory('${chat.id}')" data-id="${chat.id}">
                <div class="history-preview">${this.truncateText(chat.message, 40)}</div>
                <div class="history-meta">
                    <span class="history-time">${new Date(chat.timestamp).toLocaleDateString()}</span>
                    <span class="history-count">${chat.metadata.personality || 'N/A'}</span>
                </div>
            </div>
        `).join('');

        historyList.innerHTML = historyHTML;
    }

    updateChatHistoryUI() {
        this.loadChatHistoryUI();
        
        // Update message count
        const messagesToday = document.getElementById('messagesToday');
        if (messagesToday) {
            const today = new Date().toDateString();
            const todayCount = this.chatHistory.filter(chat => 
                new Date(chat.timestamp).toDateString() === today
            ).length;
            messagesToday.textContent = todayCount;
        }
    }

    filterChatHistory() {
        const searchTerm = document.getElementById('historySearch').value.toLowerCase();
        const historyItems = document.querySelectorAll('.history-item');
        
        historyItems.forEach(item => {
            const preview = item.querySelector('.history-preview').textContent.toLowerCase();
            item.style.display = preview.includes(searchTerm) ? 'block' : 'none';
        });
    }

    truncateText(text, maxLength) {
        return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    }

    // ===== ANALYTICS TRACKING =====

    setupAnalyticsTracking() {
        // Track page interactions
        document.addEventListener('click', (e) => this.trackClick(e));
        document.addEventListener('keydown', (e) => this.trackKeypress(e));
        document.addEventListener('scroll', () => this.trackScroll());
        
        // Track form interactions
        const userInput = document.getElementById('userInput');
        if (userInput) {
            userInput.addEventListener('focus', () => this.trackEvent('input_focus'));
            userInput.addEventListener('blur', () => this.trackEvent('input_blur'));
        }

        // Track feature usage
        this.trackFeatureUsage();
        
        // Periodic analytics save
        setInterval(() => this.saveAnalyticsData(), 30000); // Every 30 seconds
    }

    trackEvent(eventType, data = {}) {
        const event = {
            id: 'event_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
            type: eventType,
            timestamp: Date.now(),
            sessionId: this.sessionId,
            userId: this.userId,
            data: data,
            url: window.location.href,
            userAgent: navigator.userAgent
        };

        this.analyticsData.events = this.analyticsData.events || [];
        this.analyticsData.events.push(event);

        // Keep only last 1000 events in memory
        if (this.analyticsData.events.length > 1000) {
            this.analyticsData.events = this.analyticsData.events.slice(-1000);
        }

        this.updateLastActivity();
    }

    trackClick(event) {
        const element = event.target;
        const clickData = {
            elementTag: element.tagName,
            elementId: element.id,
            elementClass: element.className,
            elementText: element.textContent.substring(0, 50),
            x: event.clientX,
            y: event.clientY,
            timestamp: Date.now()
        };

        this.trackEvent('click', clickData);
        this.addToHeatmap(event.clientX, event.clientY, 'click');
    }

    trackKeypress(event) {
        this.trackEvent('keypress', {
            key: event.key,
            ctrlKey: event.ctrlKey,
            altKey: event.altKey,
            shiftKey: event.shiftKey
        });
    }

    trackScroll() {
        this.trackEvent('scroll', {
            scrollY: window.scrollY,
            scrollX: window.scrollX,
            scrollHeight: document.documentElement.scrollHeight,
            clientHeight: document.documentElement.clientHeight
        });
    }

    trackFeatureUsage() {
        // Track specific Horizon features
        const features = {
            'timer_usage': () => document.querySelectorAll('.timer-item').length,
            'reminder_usage': () => document.querySelectorAll('.reminder-item').length,
            'voice_usage': () => document.getElementById('startListening').disabled ? 1 : 0,
            'personality_changes': () => document.getElementById('personalitySelect').value,
            'quick_commands': () => document.querySelectorAll('.quick-btn').length
        };

        Object.entries(features).forEach(([feature, valueGetter]) => {
            try {
                const value = valueGetter();
                this.trackEvent('feature_usage', { feature, value });
            } catch (e) {
                console.warn(`Failed to track feature: ${feature}`, e);
            }
        });
    }

    loadAnalyticsData() {
        return JSON.parse(localStorage.getItem('horizon_analytics') || '{"events": [], "sessions": [], "performance": {}}');
    }

    saveAnalyticsData() {
        localStorage.setItem('horizon_analytics', JSON.stringify(this.analyticsData));
    }

    // ===== PERFORMANCE MONITORING =====

    startPerformanceMonitoring() {
        // Monitor page load performance
        window.addEventListener('load', () => {
            this.measurePageLoadPerformance();
        });

        // Monitor API response times
        this.monitorAPIPerformance();
        
        // Monitor memory usage (if available)
        this.monitorMemoryUsage();
    }

    measurePageLoadPerformance() {
        if ('performance' in window) {
            const perfData = performance.getEntriesByType('navigation')[0];
            
            this.performanceMetrics.pageLoad = {
                loadTime: perfData.loadEventEnd - perfData.loadEventStart,
                domContentLoaded: perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
                responseTime: perfData.responseEnd - perfData.responseStart,
                transferSize: perfData.transferSize,
                timestamp: Date.now()
            };

            this.trackEvent('performance_page_load', this.performanceMetrics.pageLoad);
            this.updatePerformanceUI();
        }
    }

    monitorAPIPerformance() {
        // Intercept fetch requests to monitor API performance
        const originalFetch = window.fetch;
        window.fetch = async (...args) => {
            const startTime = performance.now();
            try {
                const response = await originalFetch(...args);
                const endTime = performance.now();
                const responseTime = endTime - startTime;

                this.trackEvent('api_request', {
                    url: args[0],
                    method: args[1]?.method || 'GET',
                    responseTime: responseTime,
                    status: response.status,
                    success: response.ok
                });

                // Update UI response time
                const responseTimeEl = document.getElementById('responseTime');
                if (responseTimeEl && args[0].includes('/chat')) {
                    responseTimeEl.textContent = Math.round(responseTime) + 'ms';
                }

                return response;
            } catch (error) {
                const endTime = performance.now();
                this.trackEvent('api_error', {
                    url: args[0],
                    responseTime: endTime - startTime,
                    error: error.message
                });
                throw error;
            }
        };
    }

    monitorMemoryUsage() {
        if ('memory' in performance) {
            setInterval(() => {
                const memInfo = performance.memory;
                this.performanceMetrics.memory = {
                    usedJSHeapSize: memInfo.usedJSHeapSize,
                    totalJSHeapSize: memInfo.totalJSHeapSize,
                    jsHeapSizeLimit: memInfo.jsHeapSizeLimit,
                    timestamp: Date.now()
                };

                this.trackEvent('performance_memory', this.performanceMetrics.memory);
            }, 60000); // Every minute
        }
    }

    updatePerformanceUI() {
        // Update performance indicators in the UI
        const statusIndicator = document.getElementById('statusIndicator');
        if (statusIndicator && this.performanceMetrics.pageLoad) {
            const loadTime = this.performanceMetrics.pageLoad.loadTime;
            if (loadTime < 1000) {
                statusIndicator.style.background = '#10b981'; // Green for fast
            } else if (loadTime < 3000) {
                statusIndicator.style.background = '#f59e0b'; // Yellow for medium
            } else {
                statusIndicator.style.background = '#ef4444'; // Red for slow
            }
        }
    }

    // ===== HEATMAP TRACKING =====

    initializeHeatmapTracking() {
        this.heatmapData = JSON.parse(localStorage.getItem('horizon_heatmap') || '[]');
        
        // Track mouse movements periodically
        let mouseTimer;
        document.addEventListener('mousemove', (e) => {
            clearTimeout(mouseTimer);
            mouseTimer = setTimeout(() => {
                this.addToHeatmap(e.clientX, e.clientY, 'move');
            }, 100); // Throttle mouse tracking
        });

        // Save heatmap data periodically
        setInterval(() => this.saveHeatmapData(), 60000); // Every minute
    }

    addToHeatmap(x, y, type) {
        const heatPoint = {
            x: x,
            y: y,
            type: type,
            timestamp: Date.now(),
            sessionId: this.sessionId,
            url: window.location.href
        };

        this.heatmapData.push(heatPoint);

        // Keep only last 10000 points
        if (this.heatmapData.length > 10000) {
            this.heatmapData = this.heatmapData.slice(-10000);
        }
    }

    saveHeatmapData() {
        localStorage.setItem('horizon_heatmap', JSON.stringify(this.heatmapData));
    }

    // ===== A/B TESTING FRAMEWORK =====

    determineABTestVariant() {
        const variants = ['A', 'B'];
        const userHash = this.hashCode(this.userId);
        return variants[Math.abs(userHash) % variants.length];
    }

    hashCode(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return hash;
    }

    applyABTestVariant() {
        // Apply A/B test variations based on variant
        if (this.abTestVariant === 'B') {
            // Example: Change button colors for variant B
            const quickBtns = document.querySelectorAll('.quick-btn');
            quickBtns.forEach(btn => {
                btn.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
            });

            // Track that variant B was applied
            this.trackEvent('ab_test_variant_applied', { variant: 'B', feature: 'button_colors' });
        }
    }

    // ===== SESSION HISTORY MANAGEMENT =====

    addToSessionHistory(sessionData) {
        let sessionHistory = JSON.parse(localStorage.getItem('horizon_session_history') || '[]');
        sessionHistory.unshift(sessionData);
        
        // Keep only last 50 sessions
        if (sessionHistory.length > 50) {
            sessionHistory = sessionHistory.slice(0, 50);
        }
        
        localStorage.setItem('horizon_session_history', JSON.stringify(sessionHistory));
        
        // Update total sessions counter
        const totalSessions = document.getElementById('totalSessions');
        if (totalSessions) {
            totalSessions.textContent = sessionHistory.length;
        }
    }

    // ===== PUBLIC METHODS =====

    getAnalyticsSummary() {
        return {
            sessionId: this.sessionId,
            userId: this.userId,
            sessionDuration: Date.now() - this.sessionStartTime,
            totalEvents: this.analyticsData.events?.length || 0,
            chatHistory: this.chatHistory.length,
            performanceMetrics: this.performanceMetrics,
            abTestVariant: this.abTestVariant,
            heatmapPoints: this.heatmapData.length
        };
    }

    exportAnalyticsData() {
        const exportData = {
            analytics: this.analyticsData,
            chatHistory: this.chatHistory,
            heatmapData: this.heatmapData,
            performanceMetrics: this.performanceMetrics,
            sessionInfo: this.getAnalyticsSummary()
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `horizon_analytics_${this.userId}_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    clearAnalyticsData() {
        if (confirm('Are you sure you want to clear all analytics data?')) {
            localStorage.removeItem('horizon_analytics');
            localStorage.removeItem('horizon_chat_history');
            localStorage.removeItem('horizon_heatmap');
            localStorage.removeItem('horizon_session_history');
            
            this.analyticsData = { events: [], sessions: [], performance: {} };
            this.chatHistory = [];
            this.heatmapData = [];
            
            this.updateChatHistoryUI();
            console.log('✅ Analytics data cleared');
        }
    }
}

// Global functions for UI interaction
function toggleChatHistory() {
    const content = document.getElementById('chatHistoryContent');
    const toggle = document.getElementById('historyToggle');
    
    if (content.style.display === 'none') {
        content.style.display = 'block';
        toggle.textContent = '▼';
    } else {
        content.style.display = 'none';
        toggle.textContent = '▶';
    }
}

function filterChatHistory() {
    if (window.horizonAnalytics) {
        window.horizonAnalytics.filterChatHistory();
    }
}

function loadChatFromHistory(chatId) {
    if (window.horizonAnalytics) {
        const chat = window.horizonAnalytics.chatHistory.find(c => c.id === chatId);
        if (chat) {
            // Highlight the selected history item
            document.querySelectorAll('.history-item').forEach(item => {
                item.classList.remove('active');
            });
            document.querySelector(`[data-id="${chatId}"]`).classList.add('active');
            
            // Load the conversation into the input
            const userInput = document.getElementById('userInput');
            if (userInput) {
                userInput.value = chat.message;
                userInput.focus();
            }
            
            // Track the history load event
            window.horizonAnalytics.trackEvent('chat_history_loaded', {
                chatId: chatId,
                messageLength: chat.message.length
            });
        }
    }
}

// Initialize the analytics manager when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.horizonAnalytics = new HorizonAnalyticsManager();
    
    // Apply A/B test variants
    setTimeout(() => {
        window.horizonAnalytics.applyABTestVariant();
    }, 100);
    
    console.log('🚀 Horizon Analytics & Session Manager ready');
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HorizonAnalyticsManager;
}