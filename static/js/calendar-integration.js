/**
 * Calendar Integration Module
 * Self-contained calendar sync and timer integration functionality.
 * Safe implementation that works alongside existing chat interface.
 */

class CalendarIntegration {
    constructor() {
        this.baseUrl = '/api/calendar';
        this.isAuthenticated = false;
        this.authWindow = null;
        this.syncInProgress = false;
        this.config = {
            autoSyncEnabled: false,
            syncInterval: 300000, // 5 minutes
            defaultCalendar: 'primary',
            timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            reminderMinutes: 10
        };
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }
    
    async init() {
        try {
            console.log('🗓️ Initializing Calendar Integration...');
            
            // Create calendar UI elements
            this.createCalendarUI();
            
            // Check authentication status
            await this.checkAuthStatus();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Start auto-sync if enabled
            if (this.config.autoSyncEnabled && this.isAuthenticated) {
                this.startAutoSync();
            }
            
            console.log('✅ Calendar Integration initialized');
        } catch (error) {
            console.error('Calendar initialization error:', error);
        }
    }
    
    createCalendarUI() {
        try {
            // Create small calendar toggle button only
            this.createCalendarToggle();
            
            // Create popup modal (hidden by default)
            this.createCalendarModal();
            
            // Add calendar-specific styles
            this.addCalendarStyles();
            
        } catch (error) {
            console.error('Calendar UI creation error:', error);
        }
    }
    
    createCalendarToggle() {
        try {
            // Find the voice controls area (where microphone buttons are)
            const voiceControls = document.querySelector('.voice-controls');
            
            if (!voiceControls) {
                console.warn('Voice controls not found');
                return;
            }
            
            // Create small calendar sync button for the input area
            const syncBtn = document.createElement('button');
            syncBtn.className = 'voice-btn secondary cal-input-toggle';
            syncBtn.id = 'cal-input-toggle';
            syncBtn.innerHTML = '📅';
            syncBtn.title = 'Quick Calendar Sync';
            
            syncBtn.addEventListener('click', () => {
                if (this.isAuthenticated) {
                    this.syncCurrentTimer();
                } else {
                    this.showCalendarModal();
                }
            });
            
            // Add to voice controls area
            voiceControls.appendChild(syncBtn);
            
        } catch (error) {
            console.error('Calendar toggle creation error:', error);
        }
    }
    
    createCalendarModal() {
        try {
            // Create modal overlay
            const modal = document.createElement('div');
            modal.className = 'cal-modal-overlay';
            modal.id = 'cal-modal';
            modal.style.display = 'none';
            modal.innerHTML = `
                <div class="cal-modal-content">
                    <div class="cal-modal-header">
                        <h3 class="cal-modal-title">
                            <i class="cal-icon">📅</i>
                            Calendar Integration
                        </h3>
                        <button class="cal-close-btn" id="cal-close-modal">✕</button>
                    </div>
                    
                    <div class="cal-modal-body">
                        <div class="cal-status-section">
                            <div class="cal-status" id="cal-status">
                                <span class="cal-status-text">Not Connected</span>
                            </div>
                        </div>
                        
                        <div class="cal-auth-section" id="cal-auth-section">
                            <p class="cal-description">
                                Connect your Google Calendar to sync timers and reminders automatically.
                            </p>
                            <button class="cal-btn cal-btn-primary" id="cal-auth-btn">
                                Connect Google Calendar
                            </button>
                        </div>
                        
                        <div class="cal-connected-section" id="cal-connected-section" style="display: none;">
                            <div class="cal-actions">
                                <button class="cal-btn cal-btn-secondary" id="cal-sync-timer-btn">
                                    Sync Current Timer
                                </button>
                                <button class="cal-btn cal-btn-outline" id="cal-view-events-btn">
                                    View Recent Events
                                </button>
                            </div>
                            
                            <div class="cal-settings">
                                <label class="cal-checkbox-label">
                                    <input type="checkbox" id="cal-auto-sync-toggle" class="cal-checkbox">
                                    <span class="cal-checkbox-text">Auto-sync timers to calendar</span>
                                </label>
                            </div>
                            
                            <div class="cal-disconnect">
                                <button class="cal-btn cal-btn-outline cal-btn-sm" id="cal-disconnect-btn">
                                    Disconnect
                                </button>
                            </div>
                        </div>
                        
                        <div class="cal-timer-info" id="cal-timer-info" style="display: none;">
                            <div class="cal-timer-details">
                                <span class="cal-timer-text">Timer synced to calendar</span>
                                <div class="cal-timer-meta">
                                    <span class="cal-event-time"></span>
                                    <span class="cal-event-calendar"></span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="cal-recent-events" id="cal-recent-events" style="display: none;">
                            <h4 class="cal-events-title">Recent Events</h4>
                            <div class="cal-events-list"></div>
                        </div>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            
            // Add modal event listeners
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.hideCalendarModal();
                }
            });
            
            document.getElementById('cal-close-modal').addEventListener('click', () => {
                this.hideCalendarModal();
            });
            
        } catch (error) {
            console.error('Calendar modal creation error:', error);
        }
    }
    
    createStandaloneUI() {
        // This method is no longer needed since we use modal
        console.log('Using modal instead of standalone UI');
    }
    
    showCalendarModal() {
        const modal = document.getElementById('cal-modal');
        if (modal) {
            modal.style.display = 'flex';
            
            // Check auth status when opening
            this.checkAuthStatus();
        }
    }
    
    hideCalendarModal() {
        const modal = document.getElementById('cal-modal');
        if (modal) {
            modal.style.display = 'none';
        }
    }
    
    addCalendarStyles() {
        try {
            // Only add styles once
            if (document.querySelector('#cal-integration-styles')) {
                return;
            }
            
            const styles = document.createElement('style');
            styles.id = 'cal-integration-styles';
            styles.textContent = `
                /* Calendar Integration Styles - Modal Design */
                .cal-input-toggle {
                    /* Uses existing voice-btn styles */
                    background: rgba(102, 126, 234, 0.2) !important;
                    border: 1px solid rgba(102, 126, 234, 0.3) !important;
                    color: #667eea !important;
                }
                
                .cal-input-toggle:hover {
                    background: rgba(102, 126, 234, 0.3) !important;
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
                }
                
                .cal-modal-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.7);
                    backdrop-filter: blur(4px);
                    z-index: 10000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    animation: cal-fade-in 0.3s ease;
                }
                
                .cal-modal-content {
                    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                    border: 2px solid #444;
                    border-radius: 16px;
                    width: 90%;
                    max-width: 500px;
                    max-height: 80vh;
                    overflow: hidden;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
                    animation: cal-slide-up 0.3s ease;
                }
                
                .cal-modal-header {
                    padding: 20px 24px;
                    background: linear-gradient(135deg, #333 0%, #444 100%);
                    border-bottom: 2px solid #555;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                
                .cal-modal-title {
                    margin: 0;
                    color: #e5e5e5;
                    font-size: 1.3rem;
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                
                .cal-close-btn {
                    background: none;
                    border: none;
                    color: #adb5bd;
                    font-size: 1.5rem;
                    cursor: pointer;
                    width: 32px;
                    height: 32px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: all 0.3s ease;
                }
                
                .cal-close-btn:hover {
                    background: rgba(255, 255, 255, 0.1);
                    color: #e5e5e5;
                }
                
                .cal-modal-body {
                    padding: 24px;
                    max-height: 60vh;
                    overflow-y: auto;
                }
                
                .cal-status-section {
                    text-align: center;
                    margin-bottom: 24px;
                }
                
                .cal-status {
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                }
                
                .cal-status-text {
                    font-size: 1rem;
                    font-weight: 500;
                    padding: 8px 16px;
                    border-radius: 20px;
                    background: #6c757d;
                    color: white;
                }
                
                .cal-status.connected .cal-status-text {
                    background: #28a745;
                }
                
                .cal-status.syncing .cal-status-text {
                    background: #007bff;
                    animation: cal-pulse 1.5s infinite;
                }
                
                .cal-auth-section {
                    text-align: center;
                }
                
                .cal-description {
                    color: #ced4da;
                    margin-bottom: 20px;
                    line-height: 1.5;
                }
                
                .cal-connected-section {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }
                
                .cal-actions {
                    display: flex;
                    gap: 12px;
                    flex-wrap: wrap;
                }
                
                .cal-settings {
                    padding: 16px;
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 8px;
                    border: 1px solid #444;
                }
                
                .cal-disconnect {
                    text-align: center;
                    padding-top: 16px;
                    border-top: 1px solid #444;
                }
                
                .cal-btn {
                    padding: 12px 24px;
                    border: none;
                    border-radius: 8px;
                    font-size: 0.95rem;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    text-decoration: none;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    min-height: 44px;
                    flex: 1;
                }
                
                .cal-btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 6px 16px rgba(0,0,0,0.2);
                }
                
                .cal-btn-primary {
                    background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                    color: white;
                }
                
                .cal-btn-primary:hover {
                    background: linear-gradient(135deg, #0056b3 0%, #004085 100%);
                }
                
                .cal-btn-secondary {
                    background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
                    color: white;
                }
                
                .cal-btn-secondary:hover {
                    background: linear-gradient(135deg, #1e7e34 0%, #155724 100%);
                }
                
                .cal-btn-outline {
                    background: transparent;
                    color: #adb5bd;
                    border: 2px solid #6c757d;
                }
                
                .cal-btn-outline:hover {
                    background: rgba(108, 117, 125, 0.1);
                    border-color: #adb5bd;
                    color: #e5e5e5;
                }
                
                .cal-btn-sm {
                    padding: 8px 16px;
                    font-size: 0.85rem;
                    min-height: 36px;
                }
                
                .cal-checkbox-label {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    cursor: pointer;
                    font-size: 0.95rem;
                    color: #e5e5e5;
                }
                
                .cal-checkbox {
                    width: 18px;
                    height: 18px;
                    accent-color: #007bff;
                }
                
                .cal-timer-info {
                    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                    border: 1px solid #90caf9;
                    border-radius: 8px;
                    padding: 16px;
                    margin-top: 16px;
                }
                
                .cal-timer-details {
                    display: flex;
                    flex-direction: column;
                    gap: 6px;
                }
                
                .cal-timer-text {
                    font-weight: 500;
                    color: #1976d2;
                }
                
                .cal-timer-meta {
                    display: flex;
                    gap: 16px;
                    font-size: 0.9rem;
                    color: #424242;
                }
                
                .cal-recent-events {
                    margin-top: 20px;
                }
                
                .cal-events-title {
                    margin: 0 0 12px 0;
                    color: #e5e5e5;
                    font-size: 1.1rem;
                    font-weight: 600;
                }
                
                .cal-events-list {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                    max-height: 200px;
                    overflow-y: auto;
                }
                
                .cal-event-item {
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid #444;
                    border-radius: 8px;
                    padding: 12px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    transition: all 0.2s ease;
                }
                
                .cal-event-item:hover {
                    background: rgba(255, 255, 255, 0.1);
                    border-color: #555;
                }
                
                .cal-event-info {
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                }
                
                .cal-event-title {
                    font-weight: 500;
                    color: #e5e5e5;
                    font-size: 0.9rem;
                }
                
                .cal-event-time {
                    font-size: 0.8rem;
                    color: #adb5bd;
                }
                
                .cal-icon {
                    font-size: 1.3rem;
                }
                
                /* Animations */
                @keyframes cal-fade-in {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                
                @keyframes cal-slide-up {
                    from { 
                        opacity: 0; 
                        transform: translateY(30px) scale(0.95); 
                    }
                    to { 
                        opacity: 1; 
                        transform: translateY(0) scale(1); 
                    }
                }
                
                @keyframes cal-pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.6; }
                }
                
                /* Responsive design */
                @media (max-width: 768px) {
                    .cal-modal-content {
                        width: 95%;
                        margin: 20px;
                    }
                    
                    .cal-actions {
                        flex-direction: column;
                    }
                }
            `;
            
            document.head.appendChild(styles);
            
        } catch (error) {
            console.error('Calendar styles creation error:', error);
        }
    }
    
    setupEventListeners() {
        try {
            // Authentication button
            document.addEventListener('click', (e) => {
                if (e.target.id === 'cal-auth-btn') {
                    this.handleAuth();
                }
                if (e.target.id === 'cal-sync-timer-btn') {
                    this.syncCurrentTimer();
                }
                if (e.target.id === 'cal-view-events-btn') {
                    this.showRecentEvents();
                }
                if (e.target.id === 'cal-disconnect-btn') {
                    this.handleDisconnect();
                }
            });
            
            // Auto-sync toggle (use event delegation)
            document.addEventListener('change', (e) => {
                if (e.target.id === 'cal-auto-sync-toggle') {
                    this.toggleAutoSync(e.target.checked);
                }
            });
            
            // Listen for timer events from existing timer system
            document.addEventListener('timerStarted', (e) => this.onTimerStarted(e.detail));
            document.addEventListener('timerStopped', (e) => this.onTimerStopped(e.detail));
            
        } catch (error) {
            console.error('Event listeners setup error:', error);
        }
    }
    
    async checkAuthStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            const data = await response.json();
            
            if (data.success && data.calendar_connected) {
                this.setAuthenticatedState(true);
            } else {
                this.setAuthenticatedState(false);
            }
            
        } catch (error) {
            console.error('Auth status check error:', error);
            this.setAuthenticatedState(false);
        }
    }
    
    async handleAuth() {
        try {
            this.updateStatus('Connecting...', 'syncing');
            
            // Get auth URL
            const response = await fetch(`${this.baseUrl}/auth/url`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const data = await response.json();
            
            if (data.success && data.auth_url) {
                // Open auth window
                this.authWindow = window.open(
                    data.auth_url,
                    'calendar-auth',
                    'width=600,height=600,scrollbars=yes,resizable=yes'
                );
                
                // Monitor auth window
                this.monitorAuthWindow();
                
            } else {
                throw new Error(data.error || 'Failed to get auth URL');
            }
            
        } catch (error) {
            console.error('Authentication error:', error);
            this.showError('Authentication failed: ' + error.message);
            this.updateStatus('Not Connected', 'disconnected');
        }
    }
    
    monitorAuthWindow() {
        const checkClosed = setInterval(() => {
            if (this.authWindow && this.authWindow.closed) {
                clearInterval(checkClosed);
                // Check if auth was successful
                setTimeout(() => this.checkAuthStatus(), 1000);
            }
        }, 1000);
        
        // Timeout after 5 minutes
        setTimeout(() => {
            clearInterval(checkClosed);
            if (this.authWindow && !this.authWindow.closed) {
                this.authWindow.close();
                this.updateStatus('Authentication timeout', 'error');
            }
        }, 300000);
    }
    
    setAuthenticatedState(authenticated) {
        this.isAuthenticated = authenticated;
        
        const authSection = document.getElementById('cal-auth-section');
        const connectedSection = document.getElementById('cal-connected-section');
        
        if (authenticated) {
            this.updateStatus('Connected', 'connected');
            
            if (authSection) {
                authSection.style.display = 'none';
            }
            
            if (connectedSection) {
                connectedSection.style.display = 'block';
            }
            
            // Auto-hide modal after connection
            setTimeout(() => {
                this.hideCalendarModal();
                this.showSuccess('Google Calendar connected successfully!');
            }, 1500);
            
        } else {
            this.updateStatus('Not Connected', 'disconnected');
            
            if (authSection) {
                authSection.style.display = 'block';
            }
            
            if (connectedSection) {
                connectedSection.style.display = 'none';
            }
        }
    }
    
    updateStatus(text, type = '') {
        const statusEl = document.getElementById('cal-status');
        const statusTextEl = statusEl?.querySelector('.cal-status-text');
        
        if (statusTextEl) {
            statusTextEl.textContent = text;
        }
        
        if (statusEl) {
            statusEl.className = `cal-status ${type}`;
        }
    }
    
    async syncCurrentTimer() {
        try {
            if (!this.isAuthenticated) {
                this.showError('Please connect to Google Calendar first');
                return;
            }
            
            // Get current timer information
            const timerInfo = this.getCurrentTimerInfo();
            
            if (!timerInfo || !timerInfo.isRunning) {
                this.showError('No active timer found to sync');
                return;
            }
            
            this.updateStatus('Syncing...', 'syncing');
            
            // Sync to calendar
            const response = await fetch(`${this.baseUrl}/sync/timer`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    title: timerInfo.title || 'Focus Session',
                    duration_minutes: timerInfo.durationMinutes || 25,
                    description: timerInfo.description || 'Timer session from Horizon AI',
                    reminder_minutes: this.config.reminderMinutes
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showSuccess(`Timer synced to calendar: ${data.event_title}`);
                this.updateStatus('Connected', 'connected');
                this.showTimerInfo(data.event);
            } else {
                throw new Error(data.error || 'Sync failed');
            }
            
        } catch (error) {
            console.error('Timer sync error:', error);
            this.showError('Failed to sync timer: ' + error.message);
            this.updateStatus('Connected', 'connected');
        }
    }
    
    getCurrentTimerInfo() {
        try {
            // Try to find timer information from existing timer system
            const timerDisplay = document.querySelector('.timer-display') ||
                               document.querySelector('.timer-time') ||
                               document.querySelector('#timer');
            
            if (!timerDisplay) return null;
            
            // Extract timer state
            const isRunning = document.querySelector('.timer-running') !== null ||
                            document.querySelector('.start-btn')?.textContent?.includes('Stop') ||
                            timerDisplay.classList.contains('running');
            
            // Get duration from timer (try different selectors)
            let durationMinutes = 25; // default
            
            const durationInput = document.querySelector('#duration') ||
                                document.querySelector('.timer-duration') ||
                                document.querySelector('input[type="number"]');
            
            if (durationInput && durationInput.value) {
                durationMinutes = parseInt(durationInput.value) || 25;
            }
            
            // Get timer title/type
            let title = 'Focus Session';
            const titleInput = document.querySelector('#timer-title') ||
                             document.querySelector('.timer-title') ||
                             document.querySelector('.session-type');
            
            if (titleInput) {
                if (titleInput.value) {
                    title = titleInput.value;
                } else if (titleInput.textContent) {
                    title = titleInput.textContent.trim();
                }
            }
            
            return {
                isRunning,
                durationMinutes,
                title,
                description: `${durationMinutes}-minute focus session`
            };
            
        } catch (error) {
            console.error('Timer info extraction error:', error);
            return null;
        }
    }
    
    async showRecentEvents() {
        try {
            if (!this.isAuthenticated) {
                this.showError('Please connect to Google Calendar first');
                return;
            }
            
            const eventsContainer = document.getElementById('cal-recent-events') ||
                                  document.getElementById('cal-events-section');
            
            if (!eventsContainer) return;
            
            eventsContainer.style.display = 'block';
            
            const eventsList = eventsContainer.querySelector('.cal-events-list');
            if (eventsList) {
                eventsList.innerHTML = '<div class="cal-loading">Loading events...</div>';
            }
            
            // Fetch recent events
            const response = await fetch(`${this.baseUrl}/events/recent`);
            const data = await response.json();
            
            if (data.success && data.events) {
                this.displayEvents(data.events, eventsList);
            } else {
                eventsList.innerHTML = '<div class="cal-error">Failed to load events</div>';
            }
            
        } catch (error) {
            console.error('Events loading error:', error);
            const eventsList = document.querySelector('.cal-events-list');
            if (eventsList) {
                eventsList.innerHTML = '<div class="cal-error">Error loading events</div>';
            }
        }
    }
    
    displayEvents(events, container) {
        if (!container || !events.length) {
            container.innerHTML = '<div class="cal-loading">No recent events found</div>';
            return;
        }
        
        const eventsHtml = events.map(event => `
            <div class="cal-event-item">
                <div class="cal-event-info">
                    <div class="cal-event-title">${event.title || 'Untitled Event'}</div>
                    <div class="cal-event-time">${this.formatEventTime(event)}</div>
                </div>
                <div class="cal-event-actions">
                    <button class="cal-event-btn cal-event-btn-edit" onclick="calendarIntegration.editEvent('${event.id}')">
                        Edit
                    </button>
                </div>
            </div>
        `).join('');
        
        container.innerHTML = eventsHtml;
    }
    
    formatEventTime(event) {
        try {
            const start = new Date(event.start_time);
            const now = new Date();
            
            // If event is today, show time only
            if (start.toDateString() === now.toDateString()) {
                return start.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            }
            
            // If event is this week, show day and time
            const daysDiff = Math.floor((start - now) / (1000 * 60 * 60 * 24));
            if (daysDiff < 7 && daysDiff >= 0) {
                return start.toLocaleDateString([], { weekday: 'short', hour: '2-digit', minute: '2-digit' });
            }
            
            // Otherwise show full date
            return start.toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
            
        } catch (error) {
            return 'Invalid date';
        }
    }
    
    showTimerInfo(event) {
        const timerInfo = document.getElementById('cal-timer-info');
        if (!timerInfo || !event) return;
        
        const eventTime = timerInfo.querySelector('.cal-event-time');
        const eventCalendar = timerInfo.querySelector('.cal-event-calendar');
        
        if (eventTime) {
            eventTime.textContent = this.formatEventTime(event);
        }
        
        if (eventCalendar) {
            eventCalendar.textContent = event.calendar || 'Primary Calendar';
        }
        
        timerInfo.style.display = 'block';
        
        // Hide after 10 seconds
        setTimeout(() => {
            timerInfo.style.display = 'none';
        }, 10000);
    }
    
    toggleAutoSync(enabled) {
        this.config.autoSyncEnabled = enabled;
        
        if (enabled && this.isAuthenticated) {
            this.startAutoSync();
            this.showSuccess('Auto-sync enabled');
        } else {
            this.stopAutoSync();
            if (enabled) {
                this.showError('Connect to calendar to enable auto-sync');
            } else {
                this.showSuccess('Auto-sync disabled');
            }
        }
    }
    
    startAutoSync() {
        this.stopAutoSync(); // Clear any existing interval
        
        this.autoSyncInterval = setInterval(() => {
            if (this.isAuthenticated && !this.syncInProgress) {
                this.autoSyncCheck();
            }
        }, this.config.syncInterval);
    }
    
    stopAutoSync() {
        if (this.autoSyncInterval) {
            clearInterval(this.autoSyncInterval);
            this.autoSyncInterval = null;
        }
    }
    
    async autoSyncCheck() {
        try {
            const timerInfo = this.getCurrentTimerInfo();
            
            if (timerInfo && timerInfo.isRunning && !this.lastSyncedTimer) {
                console.log('🔄 Auto-syncing timer to calendar...');
                await this.syncCurrentTimer();
                this.lastSyncedTimer = Date.now();
            }
            
            // Reset sync flag when timer stops
            if (!timerInfo || !timerInfo.isRunning) {
                this.lastSyncedTimer = null;
            }
            
        } catch (error) {
            console.error('Auto-sync check error:', error);
        }
    }
    
    onTimerStarted(timerData) {
        if (this.config.autoSyncEnabled && this.isAuthenticated) {
            setTimeout(() => this.syncCurrentTimer(), 2000); // Delay to ensure timer is fully started
        }
    }
    
    onTimerStopped(timerData) {
        this.lastSyncedTimer = null;
        
        const timerInfo = document.getElementById('cal-timer-info');
        if (timerInfo) {
            timerInfo.style.display = 'none';
        }
    }
    
    async handleDisconnect() {
        try {
            // Note: Actual disconnect would require backend endpoint
            this.setAuthenticatedState(false);
            this.stopAutoSync();
            this.hideCalendarModal();
            this.showSuccess('Disconnected from Google Calendar');
        } catch (error) {
            console.error('Disconnect error:', error);
        }
    }
    
    showError(message) {
        this.showMessage(message, 'error');
    }
    
    showSuccess(message) {
        this.showMessage(message, 'success');
    }
    
    showMessage(message, type = 'info') {
        try {
            // Create toast notification instead of modal message
            const toast = document.createElement('div');
            toast.className = `cal-toast cal-toast-${type}`;
            toast.textContent = message;
            
            // Style the toast
            Object.assign(toast.style, {
                position: 'fixed',
                top: '20px',
                right: '20px',
                padding: '12px 20px',
                borderRadius: '8px',
                color: 'white',
                fontWeight: '500',
                fontSize: '0.9rem',
                zIndex: '10001',
                opacity: '0',
                transform: 'translateX(100%)',
                transition: 'all 0.3s ease',
                maxWidth: '300px',
                boxShadow: '0 4px 16px rgba(0,0,0,0.2)'
            });
            
            // Set background color based on type
            if (type === 'error') {
                toast.style.background = 'linear-gradient(135deg, #dc3545 0%, #c82333 100%)';
            } else if (type === 'success') {
                toast.style.background = 'linear-gradient(135deg, #28a745 0%, #1e7e34 100%)';
            } else {
                toast.style.background = 'linear-gradient(135deg, #007bff 0%, #0056b3 100%)';
            }
            
            document.body.appendChild(toast);
            
            // Animate in
            setTimeout(() => {
                toast.style.opacity = '1';
                toast.style.transform = 'translateX(0)';
            }, 10);
            
            // Auto-remove after 4 seconds
            setTimeout(() => {
                toast.style.opacity = '0';
                toast.style.transform = 'translateX(100%)';
                setTimeout(() => {
                    if (toast.parentNode) {
                        toast.parentNode.removeChild(toast);
                    }
                }, 300);
            }, 4000);
            
        } catch (error) {
            console.error('Message display error:', error);
            // Fallback to console
            console.log(`Calendar ${type}: ${message}`);
        }
    }
    
    destroy() {
        try {
            this.stopAutoSync();
            
            // Remove event listeners
            document.removeEventListener('timerStarted', this.onTimerStarted);
            document.removeEventListener('timerStopped', this.onTimerStopped);
            
            // Close auth window if open
            if (this.authWindow && !this.authWindow.closed) {
                this.authWindow.close();
            }
            
        } catch (error) {
            console.error('Calendar integration cleanup error:', error);
        }
    }
}

// Initialize calendar integration when script loads
const calendarIntegration = new CalendarIntegration();

// Export for global access
window.calendarIntegration = calendarIntegration;