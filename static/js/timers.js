/**
 * Timer Module - Handles timers, reminders, and audio notifications
 * Part of Horizon AI Assistant modular architecture
 */

class TimerModule {
    constructor() {
        // Active timers and reminders
        this.activeTimers = [];
        this.activeReminders = [];
    }

    /**
     * Initialize the timer module with DOM references
     */
    init(elements) {
        // DOM elements will be passed from main app
        this.updateTimersDisplay();
        this.updateRemindersDisplay();
    }

    /**
     * Add a new timer
     */
    addTimer(name, duration) {
        const timer = {
            id: Date.now(),
            name: name,
            duration: duration,
            remaining: duration,
            isRunning: false,
            startTime: null,
            intervalId: null
        };
        
        this.activeTimers.push(timer);
        this.updateTimersDisplay();
        return timer;
    }

    /**
     * Start a timer
     */
    startTimer(timerId) {
        const timer = this.activeTimers.find(t => t.id === timerId);
        if (timer && !timer.isRunning) {
            timer.isRunning = true;
            timer.startTime = Date.now();
            this.updateTimersDisplay();
            
            timer.intervalId = setInterval(() => {
                if (!timer.isRunning) {
                    clearInterval(timer.intervalId);
                    timer.intervalId = null;
                    return;
                }
                
                const elapsed = Date.now() - timer.startTime;
                timer.remaining = Math.max(0, timer.duration - elapsed);
                
                if (timer.remaining === 0) {
                    timer.isRunning = false;
                    clearInterval(timer.intervalId);
                    timer.intervalId = null;
                    this.showTimerComplete(timer);
                    this.removeTimer(timerId);
                }
                
                this.updateTimersDisplay();
            }, 50); // High frequency for better responsiveness
        }
    }

    /**
     * Pause a timer
     */
    pauseTimer(timerId) {
        const timer = this.activeTimers.find(t => t.id === timerId);
        if (timer && timer.isRunning) {
            timer.isRunning = false;
            
            // Clear the interval
            if (timer.intervalId) {
                clearInterval(timer.intervalId);
                timer.intervalId = null;
            }
            
            const elapsed = Date.now() - timer.startTime;
            timer.remaining = Math.max(0, timer.duration - elapsed);
            timer.duration = timer.remaining;
            this.updateTimersDisplay();
        }
    }

    /**
     * Remove a timer
     */
    removeTimer(timerId) {
        const timer = this.activeTimers.find(t => t.id === timerId);
        if (timer) {
            // Clear interval if timer is running
            if (timer.intervalId) {
                clearInterval(timer.intervalId);
            }
        }
        this.activeTimers = this.activeTimers.filter(t => t.id !== timerId);
        this.updateTimersDisplay();
    }

    /**
     * Update the timers display in the sidebar
     */
    updateTimersDisplay() {
        const container = document.getElementById('activeTimers');
        if (!container) return;
        
        if (this.activeTimers.length === 0) {
            container.innerHTML = '<div class="empty-state">No active timers</div>';
            return;
        }

        container.innerHTML = this.activeTimers.map(timer => `
            <div class="timer-item">
                <div class="timer-header">
                    <div class="timer-name">${timer.name}</div>
                    <button class="timer-btn" onclick="window.timerModule.removeTimer(${timer.id})">×</button>
                </div>
                <div class="timer-time">${this.formatTime(timer.remaining)}</div>
                <div class="timer-controls">
                    ${timer.isRunning ? 
                        `<button class="timer-btn" onclick="window.timerModule.pauseTimer(${timer.id})" style="background: rgba(239, 68, 68, 0.2); border-color: rgba(239, 68, 68, 0.3); color: #ef4444;">⏹️ Stop</button>` :
                        `<button class="timer-btn" onclick="window.timerModule.startTimer(${timer.id})">▶️ Start</button>`
                    }
                </div>
            </div>
        `).join('');
    }

    /**
     * Format time from milliseconds to MM:SS
     */
    formatTime(ms) {
        const totalSeconds = Math.ceil(ms / 1000);
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = totalSeconds % 60;
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }

    /**
     * Show timer completion notification
     */
    showTimerComplete(timer) {
        // Add message to chat (if chat module is available)
        if (window.chatModule) {
            window.chatModule.addMessage(`⏰ Timer "${timer.name}" has completed!`, 'ai');
        }
        
        // Play timer completion sound
        this.playTimerSound();
        
        // Browser notification if supported
        if (Notification.permission === 'granted') {
            new Notification(`Timer Complete: ${timer.name}`, {
                icon: '/static/favicon.ico',
                body: 'Your timer has finished!'
            });
        }
    }

    /**
     * Play timer completion sound using Web Audio API
     */
    playTimerSound() {
        try {
            // Create audio context for better browser support
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Create a simple chime sound using oscillators
            const oscillator1 = audioContext.createOscillator();
            const oscillator2 = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            // Set frequencies for a pleasant chime (C and E notes)
            oscillator1.frequency.setValueAtTime(523.25, audioContext.currentTime); // C5
            oscillator2.frequency.setValueAtTime(659.25, audioContext.currentTime); // E5
            
            // Set wave type
            oscillator1.type = 'sine';
            oscillator2.type = 'sine';
            
            // Connect oscillators to gain node
            oscillator1.connect(gainNode);
            oscillator2.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            // Set volume and envelope
            gainNode.gain.setValueAtTime(0, audioContext.currentTime);
            gainNode.gain.linearRampToValueAtTime(0.3, audioContext.currentTime + 0.1);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 1.5);
            
            // Start and stop the sound
            oscillator1.start(audioContext.currentTime);
            oscillator2.start(audioContext.currentTime);
            oscillator1.stop(audioContext.currentTime + 1.5);
            oscillator2.stop(audioContext.currentTime + 1.5);
            
        } catch (error) {
            console.log('Audio playback not supported or blocked');
        }
    }

    /**
     * Add a new reminder
     */
    addReminder(title, time) {
        const reminder = {
            id: Date.now(),
            title: title,
            time: time,
            timestamp: Date.now()
        };
        
        this.activeReminders.push(reminder);
        this.updateRemindersDisplay();
        
        // Set timeout for reminder
        const delay = time.getTime() - Date.now();
        if (delay > 0) {
            setTimeout(() => {
                this.showReminderNotification(reminder);
                this.removeReminder(reminder.id);
            }, delay);
        }
        
        return reminder;
    }

    /**
     * Remove a reminder
     */
    removeReminder(reminderId) {
        this.activeReminders = this.activeReminders.filter(r => r.id !== reminderId);
        this.updateRemindersDisplay();
    }

    /**
     * Update reminders display in sidebar
     */
    updateRemindersDisplay() {
        const container = document.getElementById('activeReminders');
        if (!container) return;
        
        if (this.activeReminders.length === 0) {
            container.innerHTML = '<div class="empty-state">No active reminders</div>';
            return;
        }

        container.innerHTML = this.activeReminders.map(reminder => `
            <div class="reminder-item">
                <div class="reminder-header">
                    <div class="reminder-title">${reminder.title}</div>
                    <button class="timer-btn" onclick="window.timerModule.removeReminder(${reminder.id})">×</button>
                </div>
                <div class="reminder-time">${reminder.time.toLocaleTimeString()}</div>
            </div>
        `).join('');
    }

    /**
     * Show reminder notification
     */
    showReminderNotification(reminder) {
        // Add message to chat (if chat module is available)
        if (window.chatModule) {
            window.chatModule.addMessage(`⏰ Reminder: ${reminder.title}`, 'ai');
        }
        
        // Browser notification if supported
        if (Notification.permission === 'granted') {
            new Notification(`Reminder: ${reminder.title}`, {
                icon: '/static/favicon.ico',
                body: 'Don\\'t forget!'
            });
        }
    }

    /**
     * Process timer and reminder commands from messages
     */
    processTimerReminders(message) {
        const lowerMessage = message.toLowerCase();
        
        // Timer patterns
        const timerMatch = lowerMessage.match(/set timer for (\\d+) (minute|minutes|min|second|seconds|sec)/);
        if (timerMatch) {
            const amount = parseInt(timerMatch[1]);
            const unit = timerMatch[2];
            const duration = unit.startsWith('min') ? amount * 60 * 1000 : amount * 1000;
            const timer = this.addTimer(`${amount} ${unit}`, duration);
            this.startTimer(timer.id);
            return true;
        }
        
        // Reminder patterns (simple 5-minute reminder for demo)
        if (lowerMessage.includes('remind me to')) {
            const reminderText = message.replace(/remind me to /i, '');
            const reminderTime = new Date(Date.now() + 5 * 60 * 1000); // 5 minutes from now
            this.addReminder(reminderText, reminderTime);
            return true;
        }
        
        return false;
    }
}

// Export for use in main app
window.TimerModule = TimerModule;