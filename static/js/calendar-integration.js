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
            // Find timer container to add calendar integration
            const timerContainer = document.querySelector('.timer-container') || 
                                 document.querySelector('#timer-section') ||
                                 document.querySelector('.timer');
            
            if (!timerContainer) {
                console.warn('Timer container not found, creating standalone calendar section');
                this.createStandaloneUI();
                return;
            }
            
            // Create calendar integration section
            const calendarSection = document.createElement('div');
            calendarSection.className = 'cal-integration-section';
            calendarSection.innerHTML = `
                <div class="cal-header">
                    <h4 class="cal-title">
                        <i class="cal-icon">📅</i>
                        Calendar Sync
                    </h4>
                    <div class="cal-status" id="cal-status">
                        <span class="cal-status-text">Not Connected</span>
                    </div>
                </div>
                
                <div class="cal-controls">
                    <button class="cal-btn cal-btn-primary" id="cal-auth-btn">
                        Connect Google Calendar
                    </button>
                    
                    <div class="cal-sync-controls" id="cal-sync-controls" style="display: none;">
                        <button class="cal-btn cal-btn-secondary" id="cal-sync-timer-btn">
                            Sync Current Timer
                        </button>
                        
                        <button class="cal-btn cal-btn-outline" id="cal-view-events-btn">
                            View Events
                        </button>
                        
                        <div class="cal-auto-sync">
                            <label class="cal-checkbox-label">
                                <input type="checkbox" id="cal-auto-sync-toggle" class="cal-checkbox">
                                <span class="cal-checkbox-text">Auto-sync timers</span>
                            </label>
                        </div>
                    </div>
                </div>
                
                <div class="cal-timer-info" id="cal-timer-info" style="display: none;">
                    <div class="cal-timer-details">
                        <span class="cal-timer-text">Timer will sync to calendar</span>
                        <div class="cal-timer-meta">
                            <span class="cal-event-time"></span>
                            <span class="cal-event-calendar"></span>
                        </div>
                    </div>
                </div>
                
                <div class="cal-recent-events" id="cal-recent-events" style="display: none;">
                    <h5 class="cal-events-title">Recent Events</h5>
                    <div class="cal-events-list"></div>
                </div>
            `;
            
            // Insert after timer or at the end of timer container
            const timerDisplay = timerContainer.querySelector('.timer-display') ||
                               timerContainer.querySelector('.timer-controls');
            
            if (timerDisplay && timerDisplay.nextSibling) {
                timerContainer.insertBefore(calendarSection, timerDisplay.nextSibling);
            } else {
                timerContainer.appendChild(calendarSection);
            }
            
            // Add calendar-specific styles
            this.addCalendarStyles();
            
        } catch (error) {
            console.error('Calendar UI creation error:', error);
        }
    }
    
    createStandaloneUI() {
        try {
            // Create standalone calendar section in main content area
            const mainContent = document.querySelector('.main-content') ||
                              document.querySelector('.container') ||
                              document.body;
            
            const calendarContainer = document.createElement('div');
            calendarContainer.className = 'cal-standalone-container';
            calendarContainer.innerHTML = `
                <div class="cal-standalone-section">
                    <div class="cal-header">
                        <h3 class="cal-title">
                            <i class="cal-icon">📅</i>
                            Calendar Integration
                        </h3>
                        <div class="cal-status" id="cal-status">
                            <span class="cal-status-text">Not Connected</span>
                        </div>
                    </div>
                    
                    <div class="cal-content">
                        <div class="cal-auth-section">
                            <p class="cal-description">
                                Connect your Google Calendar to sync timers and reminders automatically.
                            </p>
                            <button class="cal-btn cal-btn-primary" id="cal-auth-btn">
                                Connect Google Calendar
                            </button>
                        </div>
                        
                        <div class="cal-main-controls" id="cal-main-controls" style="display: none;">
                            <div class="cal-actions">
                                <button class="cal-btn cal-btn-secondary" id="cal-sync-timer-btn">
                                    Sync Timer to Calendar
                                </button>
                                <button class="cal-btn cal-btn-outline" id="cal-view-events-btn">
                                    View My Events
                                </button>
                                <button class="cal-btn cal-btn-outline" id="cal-create-event-btn">
                                    Create Event
                                </button>
                            </div>
                            
                            <div class="cal-settings">
                                <h4>Settings</h4>
                                <label class="cal-checkbox-label">
                                    <input type="checkbox" id="cal-auto-sync-toggle" class="cal-checkbox">
                                    <span class="cal-checkbox-text">Auto-sync timers to calendar</span>
                                </label>
                            </div>
                        </div>
                        
                        <div class="cal-events-section" id="cal-events-section" style="display: none;">
                            <h4>Upcoming Events</h4>
                            <div class="cal-events-list"></div>
                        </div>
                    </div>
                </div>
            `;
            
            mainContent.appendChild(calendarContainer);
            this.addCalendarStyles();
            
        } catch (error) {
            console.error('Standalone calendar UI creation error:', error);
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
                /* Calendar Integration Styles - Unique prefixes to avoid conflicts */
                .cal-integration-section {
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    border: 1px solid #dee2e6;
                    border-radius: 12px;
                    padding: 16px;
                    margin-top: 16px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    transition: all 0.3s ease;
                }
                
                .cal-integration-section:hover {
                    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
                }
                
                .cal-standalone-container {
                    max-width: 800px;
                    margin: 20px auto;
                    padding: 0 20px;
                }
                
                .cal-standalone-section {
                    background: white;
                    border-radius: 16px;
                    padding: 24px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                }
                
                .cal-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 16px;
                    padding-bottom: 12px;
                    border-bottom: 2px solid #e9ecef;
                }
                
                .cal-title {
                    margin: 0;
                    color: #495057;
                    font-size: 1.1rem;
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                
                .cal-icon {
                    font-size: 1.2rem;
                }
                
                .cal-status {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                
                .cal-status-text {
                    font-size: 0.9rem;
                    font-weight: 500;
                    padding: 4px 12px;
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
                
                @keyframes cal-pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.6; }
                }
                
                .cal-controls, .cal-main-controls {
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }
                
                .cal-actions {
                    display: flex;
                    gap: 12px;
                    flex-wrap: wrap;
                }
                
                .cal-btn {
                    padding: 10px 20px;
                    border: none;
                    border-radius: 8px;
                    font-size: 0.9rem;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    text-decoration: none;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    gap: 6px;
                    min-height: 40px;
                }
                
                .cal-btn:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                }
                
                .cal-btn:active {
                    transform: translateY(0);
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
                    color: #6c757d;
                    border: 2px solid #dee2e6;
                }
                
                .cal-btn-outline:hover {
                    background: #f8f9fa;
                    border-color: #adb5bd;
                }
                
                .cal-btn:disabled {
                    opacity: 0.6;
                    cursor: not-allowed;
                    transform: none !important;
                }
                
                .cal-auto-sync, .cal-settings {
                    margin-top: 12px;
                    padding: 12px;
                    background: rgba(248, 249, 250, 0.8);
                    border-radius: 8px;
                    border: 1px solid #e9ecef;
                }
                
                .cal-checkbox-label {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    cursor: pointer;
                    font-size: 0.9rem;
                    color: #495057;
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
                    padding: 12px;
                    margin-top: 12px;
                }
                
                .cal-timer-details {
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                }
                
                .cal-timer-text {
                    font-weight: 500;
                    color: #1976d2;
                }
                
                .cal-timer-meta {
                    display: flex;
                    gap: 16px;
                    font-size: 0.85rem;
                    color: #424242;
                }
                
                .cal-recent-events, .cal-events-section {
                    margin-top: 16px;
                }
                
                .cal-events-title {
                    margin: 0 0 12px 0;
                    color: #495057;
                    font-size: 1rem;
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
                    background: white;
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                    padding: 12px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    transition: all 0.2s ease;
                }
                
                .cal-event-item:hover {
                    background: #f8f9fa;
                    border-color: #dee2e6;
                }
                
                .cal-event-info {
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                }
                
                .cal-event-title {
                    font-weight: 500;
                    color: #212529;
                    font-size: 0.9rem;
                }
                
                .cal-event-time {
                    font-size: 0.8rem;
                    color: #6c757d;
                }
                
                .cal-event-actions {
                    display: flex;
                    gap: 8px;
                }
                
                .cal-event-btn {
                    padding: 4px 8px;
                    border: none;
                    border-radius: 4px;
                    font-size: 0.75rem;
                    cursor: pointer;
                    transition: all 0.2s ease;
                }
                
                .cal-event-btn-edit {
                    background: #ffc107;
                    color: #212529;
                }
                
                .cal-event-btn-delete {
                    background: #dc3545;
                    color: white;
                }
                
                .cal-loading {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                    color: #6c757d;
                }
                
                .cal-error {
                    background: #f8d7da;
                    border: 1px solid #f5c6cb;
                    color: #721c24;
                    padding: 12px;
                    border-radius: 8px;
                    margin-top: 12px;
                    font-size: 0.9rem;
                }
                
                .cal-success {
                    background: #d4edda;
                    border: 1px solid #c3e6cb;
                    color: #155724;
                    padding: 12px;
                    border-radius: 8px;
                    margin-top: 12px;
                    font-size: 0.9rem;
                }
                
                /* Responsive design */
                @media (max-width: 768px) {
                    .cal-actions {
                        flex-direction: column;
                    }
                    
                    .cal-btn {
                        width: 100%;
                    }
                    
                    .cal-timer-meta {
                        flex-direction: column;
                        gap: 4px;
                    }
                    
                    .cal-event-item {
                        flex-direction: column;
                        align-items: flex-start;
                        gap: 8px;
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
            const authBtn = document.getElementById('cal-auth-btn');
            if (authBtn) {
                authBtn.addEventListener('click', () => this.handleAuth());
            }
            
            // Sync timer button
            const syncBtn = document.getElementById('cal-sync-timer-btn');
            if (syncBtn) {
                syncBtn.addEventListener('click', () => this.syncCurrentTimer());
            }
            
            // View events button
            const viewBtn = document.getElementById('cal-view-events-btn');
            if (viewBtn) {
                viewBtn.addEventListener('click', () => this.showRecentEvents());
            }
            
            // Create event button
            const createBtn = document.getElementById('cal-create-event-btn');
            if (createBtn) {
                createBtn.addEventListener('click', () => this.createQuickEvent());
            }
            
            // Auto-sync toggle
            const autoSyncToggle = document.getElementById('cal-auto-sync-toggle');
            if (autoSyncToggle) {
                autoSyncToggle.addEventListener('change', (e) => {
                    this.toggleAutoSync(e.target.checked);
                });
            }
            
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
        
        const authBtn = document.getElementById('cal-auth-btn');
        const syncControls = document.getElementById('cal-sync-controls') || 
                           document.getElementById('cal-main-controls');
        
        if (authenticated) {
            this.updateStatus('Connected', 'connected');
            
            if (authBtn) {
                authBtn.textContent = 'Disconnect';
                authBtn.onclick = () => this.handleDisconnect();
            }
            
            if (syncControls) {
                syncControls.style.display = 'block';
            }
            
        } else {
            this.updateStatus('Not Connected', 'disconnected');
            
            if (authBtn) {
                authBtn.textContent = 'Connect Google Calendar';
                authBtn.onclick = () => this.handleAuth();
            }
            
            if (syncControls) {
                syncControls.style.display = 'none';
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
    
    showError(message) {
        this.showMessage(message, 'error');
    }
    
    showSuccess(message) {
        this.showMessage(message, 'success');
    }
    
    showMessage(message, type = 'info') {
        try {
            // Remove existing messages
            const existingMessages = document.querySelectorAll('.cal-error, .cal-success');
            existingMessages.forEach(msg => msg.remove());
            
            // Create message element
            const messageEl = document.createElement('div');
            messageEl.className = `cal-${type}`;
            messageEl.textContent = message;
            
            // Find container to show message
            const container = document.querySelector('.cal-integration-section') ||
                            document.querySelector('.cal-standalone-section') ||
                            document.querySelector('.cal-controls');
            
            if (container) {
                container.appendChild(messageEl);
                
                // Auto-remove after 5 seconds
                setTimeout(() => {
                    if (messageEl.parentNode) {
                        messageEl.parentNode.removeChild(messageEl);
                    }
                }, 5000);
            }
            
        } catch (error) {
            console.error('Message display error:', error);
        }
    }
    
    async handleDisconnect() {
        try {
            // Note: Actual disconnect would require backend endpoint
            this.setAuthenticatedState(false);
            this.stopAutoSync();
            this.showSuccess('Disconnected from Google Calendar');
        } catch (error) {
            console.error('Disconnect error:', error);
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