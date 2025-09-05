// Advanced UI Components for Horizon AI Assistant
// Toast notifications, Professional modals, Improved tooltips, Status indicators with animations

class AdvancedUIComponents {
    constructor() {
        this.init();
        this.toastQueue = [];
        this.activeToasts = new Set();
        this.maxToasts = 5;
    }

    init() {
        this.createToastContainer();
        this.createModalContainer();
        this.createTooltipContainer();
        this.enhanceStatusIndicators();
        this.initEventListeners();
        this.createProgressBars();
    }

    // ============ TOAST NOTIFICATIONS ============
    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    showToast(message, type = 'info', duration = 5000, options = {}) {
        const toast = {
            id: Date.now() + Math.random(),
            message,
            type,
            duration,
            options: {
                title: '',
                icon: '',
                actions: [],
                dismissible: true,
                sound: true,
                ...options
            }
        };

        if (this.activeToasts.size >= this.maxToasts) {
            this.toastQueue.push(toast);
            return;
        }

        this.createToast(toast);
    }

    createToast(toast) {
        const container = document.getElementById('toast-container');
        const toastElement = document.createElement('div');
        toastElement.className = `toast toast-${toast.type}`;
        toastElement.dataset.toastId = toast.id;

        const icon = toast.options.icon || this.getToastIcon(toast.type);
        const title = toast.options.title || this.getToastTitle(toast.type);

        toastElement.innerHTML = `
            <div class="toast-header">
                <div class="toast-icon">${icon}</div>
                <div class="toast-title">${title}</div>
                ${toast.options.dismissible ? '<button class="toast-close" aria-label="Close">×</button>' : ''}
            </div>
            <div class="toast-body">
                <div class="toast-message">${toast.message}</div>
                ${toast.options.actions.length > 0 ? this.createToastActions(toast.options.actions) : ''}
            </div>
            <div class="toast-progress"></div>
        `;

        // Add event listeners
        if (toast.options.dismissible) {
            const closeBtn = toastElement.querySelector('.toast-close');
            closeBtn.addEventListener('click', () => this.dismissToast(toast.id));
        }

        // Add action listeners
        toast.options.actions.forEach((action, index) => {
            const actionBtn = toastElement.querySelector(`[data-action-index="${index}"]`);
            if (actionBtn) {
                actionBtn.addEventListener('click', () => {
                    action.callback();
                    if (action.dismissOnClick !== false) {
                        this.dismissToast(toast.id);
                    }
                });
            }
        });

        // Play sound if enabled
        if (toast.options.sound) {
            this.playToastSound(toast.type);
        }

        container.appendChild(toastElement);
        this.activeToasts.add(toast.id);

        // Trigger animation
        requestAnimationFrame(() => {
            toastElement.classList.add('toast-show');
        });

        // Auto-dismiss
        if (toast.duration > 0) {
            const progressBar = toastElement.querySelector('.toast-progress');
            progressBar.style.animationDuration = `${toast.duration}ms`;
            progressBar.classList.add('toast-progress-animate');

            setTimeout(() => {
                this.dismissToast(toast.id);
            }, toast.duration);
        }

        // Handle click to pause/resume auto-dismiss
        toastElement.addEventListener('mouseenter', () => {
            const progressBar = toastElement.querySelector('.toast-progress');
            progressBar.style.animationPlayState = 'paused';
        });

        toastElement.addEventListener('mouseleave', () => {
            const progressBar = toastElement.querySelector('.toast-progress');
            progressBar.style.animationPlayState = 'running';
        });
    }

    createToastActions(actions) {
        return `
            <div class="toast-actions">
                ${actions.map((action, index) => `
                    <button class="toast-action-btn ${action.class || ''}" data-action-index="${index}">
                        ${action.text}
                    </button>
                `).join('')}
            </div>
        `;
    }

    dismissToast(toastId) {
        const toastElement = document.querySelector(`[data-toast-id="${toastId}"]`);
        if (!toastElement) return;

        toastElement.classList.add('toast-hide');
        
        setTimeout(() => {
            toastElement.remove();
            this.activeToasts.delete(toastId);
            
            // Process queue
            if (this.toastQueue.length > 0) {
                const nextToast = this.toastQueue.shift();
                this.createToast(nextToast);
            }
        }, 300);
    }

    getToastIcon(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️',
            loading: '⏳'
        };
        return icons[type] || '📢';
    }

    getToastTitle(type) {
        const titles = {
            success: 'Success',
            error: 'Error',
            warning: 'Warning',
            info: 'Information',
            loading: 'Processing'
        };
        return titles[type] || 'Notification';
    }

    playToastSound(type) {
        // Create audio context for modern sound synthesis
        if (window.AudioContext || window.webkitAudioContext) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            const audioContext = new AudioContext();
            
            const frequencies = {
                success: [523, 659, 784], // C5, E5, G5
                error: [392, 311], // G4, Eb4
                warning: [466, 622], // Bb4, Eb5
                info: [440], // A4
                loading: [329, 415] // E4, Ab4
            };

            const freq = frequencies[type] || [440];
            
            freq.forEach((frequency, index) => {
                setTimeout(() => {
                    const oscillator = audioContext.createOscillator();
                    const gainNode = audioContext.createGain();
                    
                    oscillator.connect(gainNode);
                    gainNode.connect(audioContext.destination);
                    
                    oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);
                    oscillator.type = 'sine';
                    
                    gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
                    
                    oscillator.start(audioContext.currentTime);
                    oscillator.stop(audioContext.currentTime + 0.2);
                }, index * 100);
            });
        }
    }

    // ============ PROFESSIONAL MODALS ============
    createModalContainer() {
        const container = document.createElement('div');
        container.id = 'modal-overlay';
        container.className = 'modal-overlay';
        document.body.appendChild(container);

        container.addEventListener('click', (e) => {
            if (e.target === container) {
                this.closeModal();
            }
        });

        // ESC key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && container.classList.contains('modal-show')) {
                this.closeModal();
            }
        });
    }

    showModal(options = {}) {
        const defaultOptions = {
            title: 'Modal Title',
            content: 'Modal content goes here',
            size: 'medium', // small, medium, large, fullscreen
            type: 'default', // default, confirm, alert, form
            buttons: [],
            closable: true,
            backdrop: true,
            animation: 'fade', // fade, slide, zoom, flip
            onShow: () => {},
            onHide: () => {},
            className: ''
        };

        const config = { ...defaultOptions, ...options };
        const overlay = document.getElementById('modal-overlay');
        
        overlay.innerHTML = `
            <div class="modal modal-${config.size} modal-${config.type} modal-${config.animation} ${config.className}">
                <div class="modal-header">
                    <h3 class="modal-title">${config.title}</h3>
                    ${config.closable ? '<button class="modal-close" aria-label="Close">×</button>' : ''}
                </div>
                <div class="modal-body">
                    ${config.content}
                </div>
                ${config.buttons.length > 0 ? this.createModalButtons(config.buttons) : ''}
            </div>
        `;

        // Add event listeners
        if (config.closable) {
            const closeBtn = overlay.querySelector('.modal-close');
            closeBtn.addEventListener('click', () => this.closeModal());
        }

        // Add button listeners
        config.buttons.forEach((button, index) => {
            const buttonElement = overlay.querySelector(`[data-button-index="${index}"]`);
            if (buttonElement) {
                buttonElement.addEventListener('click', () => {
                    const result = button.callback ? button.callback() : true;
                    if (result !== false && button.dismiss !== false) {
                        this.closeModal();
                    }
                });
            }
        });

        // Show modal
        overlay.classList.add('modal-show');
        document.body.classList.add('modal-open');
        
        // Focus management
        setTimeout(() => {
            const firstFocusable = overlay.querySelector('button, input, select, textarea, [tabindex]:not([tabindex="-1"])');
            if (firstFocusable) {
                firstFocusable.focus();
            }
        }, 100);

        config.onShow();
    }

    createModalButtons(buttons) {
        return `
            <div class="modal-footer">
                ${buttons.map((button, index) => `
                    <button class="modal-btn ${button.class || 'btn-secondary'}" data-button-index="${index}">
                        ${button.text}
                    </button>
                `).join('')}
            </div>
        `;
    }

    closeModal() {
        const overlay = document.getElementById('modal-overlay');
        const modal = overlay.querySelector('.modal');
        
        if (modal) {
            modal.classList.add('modal-hide');
            setTimeout(() => {
                overlay.classList.remove('modal-show');
                document.body.classList.remove('modal-open');
                overlay.innerHTML = '';
            }, 300);
        }
    }

    // Predefined modal types
    showAlert(title, message, callback) {
        this.showModal({
            title,
            content: `<div class="alert-content">${message}</div>`,
            type: 'alert',
            size: 'small',
            buttons: [
                {
                    text: 'OK',
                    class: 'btn-primary',
                    callback
                }
            ]
        });
    }

    showConfirm(title, message, onConfirm, onCancel) {
        this.showModal({
            title,
            content: `<div class="confirm-content">${message}</div>`,
            type: 'confirm',
            size: 'small',
            buttons: [
                {
                    text: 'Cancel',
                    class: 'btn-secondary',
                    callback: onCancel
                },
                {
                    text: 'Confirm',
                    class: 'btn-primary',
                    callback: onConfirm
                }
            ]
        });
    }

    showPrompt(title, message, defaultValue = '', onSubmit, onCancel) {
        this.showModal({
            title,
            content: `
                <div class="prompt-content">
                    <p>${message}</p>
                    <input type="text" class="prompt-input" value="${defaultValue}" placeholder="Enter value...">
                </div>
            `,
            type: 'form',
            size: 'medium',
            buttons: [
                {
                    text: 'Cancel',
                    class: 'btn-secondary',
                    callback: onCancel
                },
                {
                    text: 'Submit',
                    class: 'btn-primary',
                    callback: () => {
                        const input = document.querySelector('.prompt-input');
                        if (onSubmit) {
                            onSubmit(input.value);
                        }
                    }
                }
            ]
        });
    }

    // ============ IMPROVED TOOLTIPS ============
    createTooltipContainer() {
        const container = document.createElement('div');
        container.id = 'tooltip-container';
        container.className = 'tooltip-container';
        document.body.appendChild(container);
    }

    initTooltips() {
        // Find all elements with tooltip attributes
        const tooltipElements = document.querySelectorAll('[data-tooltip], [title]');
        
        tooltipElements.forEach(element => {
            const tooltipText = element.dataset.tooltip || element.title;
            const position = element.dataset.tooltipPosition || 'top';
            const theme = element.dataset.tooltipTheme || 'dark';
            const delay = parseInt(element.dataset.tooltipDelay) || 500;
            
            // Remove default title to prevent browser tooltip
            if (element.title) {
                element.dataset.tooltip = element.title;
                element.removeAttribute('title');
            }
            
            let tooltipTimeout;
            let tooltipElement;
            
            element.addEventListener('mouseenter', () => {
                tooltipTimeout = setTimeout(() => {
                    tooltipElement = this.showTooltip(element, tooltipText, position, theme);
                }, delay);
            });
            
            element.addEventListener('mouseleave', () => {
                clearTimeout(tooltipTimeout);
                if (tooltipElement) {
                    this.hideTooltip(tooltipElement);
                }
            });
            
            element.addEventListener('click', () => {
                clearTimeout(tooltipTimeout);
                if (tooltipElement) {
                    this.hideTooltip(tooltipElement);
                }
            });
        });
    }

    showTooltip(element, text, position = 'top', theme = 'dark') {
        const container = document.getElementById('tooltip-container');
        const tooltip = document.createElement('div');
        tooltip.className = `tooltip tooltip-${position} tooltip-${theme}`;
        tooltip.innerHTML = `<div class="tooltip-content">${text}</div>`;
        
        container.appendChild(tooltip);
        
        // Position the tooltip
        const elementRect = element.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        
        let top, left;
        
        switch (position) {
            case 'top':
                top = elementRect.top - tooltipRect.height - 10;
                left = elementRect.left + (elementRect.width - tooltipRect.width) / 2;
                break;
            case 'bottom':
                top = elementRect.bottom + 10;
                left = elementRect.left + (elementRect.width - tooltipRect.width) / 2;
                break;
            case 'left':
                top = elementRect.top + (elementRect.height - tooltipRect.height) / 2;
                left = elementRect.left - tooltipRect.width - 10;
                break;
            case 'right':
                top = elementRect.top + (elementRect.height - tooltipRect.height) / 2;
                left = elementRect.right + 10;
                break;
        }
        
        // Keep tooltip in viewport
        const viewport = {
            width: window.innerWidth,
            height: window.innerHeight
        };
        
        if (left < 0) left = 10;
        if (left + tooltipRect.width > viewport.width) left = viewport.width - tooltipRect.width - 10;
        if (top < 0) top = 10;
        if (top + tooltipRect.height > viewport.height) top = viewport.height - tooltipRect.height - 10;
        
        tooltip.style.top = `${top}px`;
        tooltip.style.left = `${left}px`;
        
        // Animate in
        requestAnimationFrame(() => {
            tooltip.classList.add('tooltip-show');
        });
        
        return tooltip;
    }

    hideTooltip(tooltip) {
        if (!tooltip) return;
        
        tooltip.classList.add('tooltip-hide');
        setTimeout(() => {
            tooltip.remove();
        }, 200);
    }

    // ============ STATUS INDICATORS WITH ANIMATIONS ============
    enhanceStatusIndicators() {
        this.createStatusIndicators();
        this.addConnectionStatus();
        this.addLoadingIndicators();
    }

    createStatusIndicators() {
        // Find status elements and enhance them
        const statusElements = document.querySelectorAll('.status-indicator, [data-status]');
        
        statusElements.forEach(element => {
            this.enhanceStatusElement(element);
        });
    }

    enhanceStatusElement(element) {
        const status = element.dataset.status || 'ready';
        const pulse = element.dataset.pulse === 'true';
        const glow = element.dataset.glow === 'true';
        
        element.classList.add('enhanced-status');
        
        if (pulse) {
            element.classList.add('status-pulse');
        }
        
        if (glow) {
            element.classList.add('status-glow');
        }
        
        // Add status dot
        if (!element.querySelector('.status-dot')) {
            const dot = document.createElement('span');
            dot.className = `status-dot status-${status}`;
            element.insertBefore(dot, element.firstChild);
        }
    }

    updateStatus(elementId, status, message, options = {}) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        const dot = element.querySelector('.status-dot');
        if (dot) {
            dot.className = `status-dot status-${status}`;
        }
        
        if (message) {
            const textNode = element.childNodes[element.childNodes.length - 1];
            if (textNode && textNode.nodeType === Node.TEXT_NODE) {
                textNode.textContent = message;
            } else {
                element.appendChild(document.createTextNode(message));
            }
        }
        
        // Add animation effects
        if (options.animate) {
            element.classList.add('status-change');
            setTimeout(() => {
                element.classList.remove('status-change');
            }, 600);
        }
        
        if (options.pulse) {
            element.classList.add('status-pulse');
        } else {
            element.classList.remove('status-pulse');
        }
        
        if (options.glow) {
            element.classList.add('status-glow');
        } else {
            element.classList.remove('status-glow');
        }
    }

    addConnectionStatus() {
        // Monitor connection status
        const updateConnectionStatus = () => {
            const status = navigator.onLine ? 'online' : 'offline';
            const message = navigator.onLine ? 'Connected' : 'Offline';
            
            this.showToast(
                navigator.onLine ? 'Connection restored' : 'Connection lost',
                navigator.onLine ? 'success' : 'error',
                3000
            );
        };
        
        window.addEventListener('online', updateConnectionStatus);
        window.addEventListener('offline', updateConnectionStatus);
    }

    addLoadingIndicators() {
        // Create global loading overlay
        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-spinner">
                <div class="spinner-ring"></div>
                <div class="loading-text">Loading...</div>
            </div>
        `;
        document.body.appendChild(overlay);
    }

    showLoading(message = 'Loading...') {
        const overlay = document.getElementById('loading-overlay');
        const textElement = overlay.querySelector('.loading-text');
        textElement.textContent = message;
        overlay.classList.add('loading-show');
        document.body.classList.add('loading-active');
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        overlay.classList.remove('loading-show');
        document.body.classList.remove('loading-active');
    }

    // ============ PROGRESS BARS ============
    createProgressBars() {
        this.progressBars = new Map();
    }

    createProgressBar(container, options = {}) {
        const defaultOptions = {
            value: 0,
            max: 100,
            animated: true,
            striped: false,
            color: 'primary',
            size: 'normal', // small, normal, large
            showPercentage: true,
            label: '',
            id: Date.now().toString()
        };
        
        const config = { ...defaultOptions, ...options };
        
        const progressWrapper = document.createElement('div');
        progressWrapper.className = `progress-wrapper progress-${config.size}`;
        progressWrapper.innerHTML = `
            ${config.label ? `<div class="progress-label">${config.label}</div>` : ''}
            <div class="progress-bar-container">
                <div class="progress-bar progress-${config.color} ${config.animated ? 'progress-animated' : ''} ${config.striped ? 'progress-striped' : ''}"
                     role="progressbar" 
                     aria-valuenow="${config.value}" 
                     aria-valuemin="0" 
                     aria-valuemax="${config.max}">
                    <div class="progress-fill" style="width: ${(config.value / config.max) * 100}%"></div>
                </div>
                ${config.showPercentage ? `<div class="progress-percentage">${Math.round((config.value / config.max) * 100)}%</div>` : ''}
            </div>
        `;
        
        container.appendChild(progressWrapper);
        this.progressBars.set(config.id, { element: progressWrapper, config });
        
        return config.id;
    }

    updateProgressBar(id, value, options = {}) {
        const progressData = this.progressBars.get(id);
        if (!progressData) return;
        
        const { element, config } = progressData;
        const fill = element.querySelector('.progress-fill');
        const percentage = element.querySelector('.progress-percentage');
        const progressBar = element.querySelector('.progress-bar');
        
        const newPercentage = Math.min(100, Math.max(0, (value / config.max) * 100));
        
        fill.style.width = `${newPercentage}%`;
        progressBar.setAttribute('aria-valuenow', value);
        
        if (percentage) {
            percentage.textContent = `${Math.round(newPercentage)}%`;
        }
        
        if (options.label) {
            const label = element.querySelector('.progress-label');
            if (label) {
                label.textContent = options.label;
            }
        }
        
        if (options.color && options.color !== config.color) {
            progressBar.classList.remove(`progress-${config.color}`);
            progressBar.classList.add(`progress-${options.color}`);
            config.color = options.color;
        }
        
        config.value = value;
    }

    removeProgressBar(id) {
        const progressData = this.progressBars.get(id);
        if (progressData) {
            progressData.element.remove();
            this.progressBars.delete(id);
        }
    }

    // ============ EVENT LISTENERS ============
    initEventListeners() {
        // Initialize tooltips when DOM changes
        const observer = new MutationObserver(() => {
            this.initTooltips();
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        // Initialize tooltips on load
        document.addEventListener('DOMContentLoaded', () => {
            this.initTooltips();
        });
    }

    // ============ UTILITY METHODS ============
    addToastStyles() {
        // This method adds the required CSS dynamically if not present
        if (document.getElementById('advanced-ui-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'advanced-ui-styles';
        style.textContent = `
            /* Toast styles will be added here if needed */
        `;
        document.head.appendChild(style);
    }
}

// Initialize Advanced UI Components
document.addEventListener('DOMContentLoaded', () => {
    window.advancedUI = new AdvancedUIComponents();
    
    // Add example usage for demonstration
    setTimeout(() => {
        if (window.location.search.includes('demo=true')) {
            // Demo toast notifications
            window.advancedUI.showToast('Welcome to Horizon AI!', 'success', 5000, {
                title: 'Welcome',
                icon: '🌟',
                actions: [
                    {
                        text: 'Start Tour',
                        callback: () => console.log('Starting tour...')
                    }
                ]
            });
        }
    }, 1000);
});

// Export for global usage
window.AdvancedUIComponents = AdvancedUIComponents;
