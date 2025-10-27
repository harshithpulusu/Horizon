/**
 * Batch Operations Manager
 * Self-contained module for managing batch operations that runs parallel to existing chat functionality.
 */

class BatchOperationsManager {
    constructor() {
        this.isEnabled = false;
        this.currentBatch = null;
        this.activeBatches = new Map();
        this.pollInterval = null;
        this.pollFrequency = 2000; // 2 seconds
        
        // DOM elements
        this.batchPanel = null;
        this.toggleButton = null;
        this.statusContainer = null;
        this.operationsContainer = null;
        
        // Configuration
        this.config = {
            apiEndpoint: '/api/batch',
            maxDisplayedOperations: 20,
            autoRefresh: true,
            showNotifications: true,
            soundEnabled: false
        };
        
        // Batch operation templates
        this.operationTemplates = {
            text_generation: {
                name: 'Text Generation',
                description: 'Generate multiple text responses',
                icon: '📝',
                defaultData: {
                    prompts: ['Explain artificial intelligence', 'What is machine learning?']
                }
            },
            image_generation: {
                name: 'Image Generation',
                description: 'Generate multiple images',
                icon: '🎨',
                defaultData: {
                    prompts: ['A sunset over mountains', 'A futuristic city']
                }
            },
            data_analysis: {
                name: 'Data Analysis',
                description: 'Analyze multiple datasets',
                icon: '📊',
                defaultData: {
                    datasets: [
                        { name: 'sample_data_1', rows: 100, columns: 5 },
                        { name: 'sample_data_2', rows: 200, columns: 8 }
                    ]
                }
            },
            file_processing: {
                name: 'File Processing',
                description: 'Process multiple files',
                icon: '📁',
                defaultData: {
                    files: [
                        { name: 'document1.pdf', size: 1024000 },
                        { name: 'document2.docx', size: 512000 }
                    ]
                }
            }
        };
        
        console.log('🔄 Batch Operations Manager initialized');
    }
    
    /**
     * Initialize batch operations system
     */
    init() {
        try {
            // Create batch panel UI
            this.createBatchPanel();
            
            // Create toggle button
            this.createToggleButton();
            
            // Load settings
            this.loadSettings();
            
            // Setup event listeners
            this.attachEventListeners();
            
            // Start polling if enabled
            if (this.isEnabled && this.config.autoRefresh) {
                this.startPolling();
            }
            
            console.log('✅ Batch Operations Manager initialized successfully');
            return true;
            
        } catch (error) {
            console.error('Batch Operations initialization error:', error);
            return false;
        }
    }
    
    /**
     * Create the batch operations panel
     */
    createBatchPanel() {
        this.batchPanel = document.createElement('div');
        this.batchPanel.className = 'batch-operations-panel';
        this.batchPanel.style.display = 'none';
        
        // Panel header
        const header = document.createElement('div');
        header.className = 'batch-panel-header';
        header.innerHTML = `
            <h3>🔄 Batch Operations</h3>
            <div class="batch-controls">
                <button type="button" class="batch-btn batch-btn-small" id="refreshBatchStatus">🔄</button>
                <button type="button" class="batch-btn batch-btn-small" id="closeBatchPanel">✖</button>
            </div>
        `;
        
        // Quick actions
        const quickActions = document.createElement('div');
        quickActions.className = 'batch-quick-actions';
        quickActions.innerHTML = `
            <h4>Quick Start</h4>
            <div class="batch-template-buttons"></div>
        `;
        
        // Custom batch form
        const customForm = document.createElement('div');
        customForm.className = 'batch-custom-form';
        customForm.innerHTML = `
            <h4>Custom Batch</h4>
            <div class="form-group">
                <select id="batchOperationType" class="batch-select">
                    <option value="">Select operation type...</option>
                </select>
            </div>
            <div class="form-group">
                <textarea id="batchData" class="batch-textarea" placeholder="Enter batch data as JSON..." rows="4"></textarea>
            </div>
            <button type="button" id="submitBatch" class="batch-btn">Submit Batch</button>
        `;
        
        // Status container
        this.statusContainer = document.createElement('div');
        this.statusContainer.className = 'batch-status-container';
        this.statusContainer.innerHTML = '<h4>Queue Status</h4><div class="batch-status-content">Loading...</div>';
        
        // Operations container
        this.operationsContainer = document.createElement('div');
        this.operationsContainer.className = 'batch-operations-container';
        this.operationsContainer.innerHTML = '<h4>Recent Operations</h4><div class="batch-operations-content">No operations yet</div>';
        
        // Assemble panel
        this.batchPanel.appendChild(header);
        this.batchPanel.appendChild(quickActions);
        this.batchPanel.appendChild(customForm);
        this.batchPanel.appendChild(this.statusContainer);
        this.batchPanel.appendChild(this.operationsContainer);
        
        // Add to page (position it safely)
        const chatContainer = document.querySelector('.chat-container, .main-content, body');
        if (chatContainer) {
            chatContainer.appendChild(this.batchPanel);
        } else {
            document.body.appendChild(this.batchPanel);
        }
        
        // Populate operation types
        this.populateOperationTypes();
        
        // Populate template buttons
        this.populateTemplateButtons();
        
        console.log('Batch panel created');
    }
    
    /**
     * Create toggle button for batch operations
     */
    createToggleButton() {
        this.toggleButton = document.createElement('button');
        this.toggleButton.className = 'batch-operations-toggle';
        this.toggleButton.innerHTML = '🔄';
        this.toggleButton.title = 'Toggle Batch Operations';
        this.toggleButton.type = 'button';
        
        // Find a good place to insert the button
        const chatControls = document.querySelector('.chat-controls, .controls, .button-container');
        if (chatControls) {
            chatControls.appendChild(this.toggleButton);
        } else {
            // Fallback: add to top of page
            const header = document.querySelector('header, .header, .navbar');
            if (header) {
                header.appendChild(this.toggleButton);
            } else {
                document.body.appendChild(this.toggleButton);
            }
        }
        
        console.log('Batch toggle button created');
    }
    
    /**
     * Populate operation types dropdown
     */
    populateOperationTypes() {
        const select = this.batchPanel.querySelector('#batchOperationType');
        
        Object.keys(this.operationTemplates).forEach(type => {
            const template = this.operationTemplates[type];
            const option = document.createElement('option');
            option.value = type;
            option.textContent = `${template.icon} ${template.name}`;
            select.appendChild(option);
        });
    }
    
    /**
     * Populate template buttons for quick actions
     */
    populateTemplateButtons() {
        const container = this.batchPanel.querySelector('.batch-template-buttons');
        
        Object.entries(this.operationTemplates).forEach(([type, template]) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'batch-template-btn';
            button.innerHTML = `${template.icon}<br><small>${template.name}</small>`;
            button.title = template.description;
            
            button.addEventListener('click', () => {
                this.submitQuickBatch(type);
            });
            
            container.appendChild(button);
        });
    }
    
    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Toggle button
        this.toggleButton.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.togglePanel();
        });
        
        // Panel controls
        const refreshBtn = this.batchPanel.querySelector('#refreshBatchStatus');
        const closeBtn = this.batchPanel.querySelector('#closeBatchPanel');
        const submitBtn = this.batchPanel.querySelector('#submitBatch');
        
        refreshBtn.addEventListener('click', () => this.refreshStatus());
        closeBtn.addEventListener('click', () => this.hidePanel());
        submitBtn.addEventListener('click', () => this.submitCustomBatch());
        
        // Operation type change
        const typeSelect = this.batchPanel.querySelector('#batchOperationType');
        typeSelect.addEventListener('change', (e) => {
            this.updateDataTemplate(e.target.value);
        });
        
        console.log('Event listeners attached');
    }
    
    /**
     * Update data template based on selected operation type
     */
    updateDataTemplate(operationType) {
        const textarea = this.batchPanel.querySelector('#batchData');
        
        if (operationType && this.operationTemplates[operationType]) {
            const template = this.operationTemplates[operationType];
            textarea.value = JSON.stringify(template.defaultData, null, 2);
            textarea.placeholder = `Enter data for ${template.name}`;
        } else {
            textarea.value = '';
            textarea.placeholder = 'Enter batch data as JSON...';
        }
    }
    
    /**
     * Submit quick batch with template data
     */
    async submitQuickBatch(operationType) {
        try {
            const template = this.operationTemplates[operationType];
            if (!template) {
                throw new Error(`Unknown operation type: ${operationType}`);
            }
            
            const batchData = {
                type: operationType,
                data: template.defaultData
            };
            
            const operationId = await this.submitBatchOperation(batchData);
            this.showNotification(`Quick batch submitted: ${template.name}`, 'success');
            
            // Refresh status
            setTimeout(() => this.refreshStatus(), 1000);
            
        } catch (error) {
            console.error('Quick batch submission error:', error);
            this.showNotification(`Failed to submit batch: ${error.message}`, 'error');
        }
    }
    
    /**
     * Submit custom batch from form
     */
    async submitCustomBatch() {
        try {
            const typeSelect = this.batchPanel.querySelector('#batchOperationType');
            const dataTextarea = this.batchPanel.querySelector('#batchData');
            
            const operationType = typeSelect.value;
            const dataText = dataTextarea.value.trim();
            
            if (!operationType) {
                throw new Error('Please select an operation type');
            }
            
            if (!dataText) {
                throw new Error('Please enter batch data');
            }
            
            let batchData;
            try {
                batchData = JSON.parse(dataText);
            } catch (e) {
                throw new Error('Invalid JSON data format');
            }
            
            const submitData = {
                type: operationType,
                data: batchData
            };
            
            const operationId = await this.submitBatchOperation(submitData);
            this.showNotification(`Custom batch submitted successfully`, 'success');
            
            // Clear form
            typeSelect.value = '';
            dataTextarea.value = '';
            
            // Refresh status
            setTimeout(() => this.refreshStatus(), 1000);
            
        } catch (error) {
            console.error('Custom batch submission error:', error);
            this.showNotification(`Failed to submit batch: ${error.message}`, 'error');
        }
    }
    
    /**
     * Submit batch operation to API
     */
    async submitBatchOperation(batchData) {
        const response = await fetch(`${this.config.apiEndpoint}/submit`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(batchData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Unknown error');
        }
        
        return result.operation_id;
    }
    
    /**
     * Refresh queue status and operations
     */
    async refreshStatus() {
        try {
            // Get queue status
            const statusResponse = await fetch(`${this.config.apiEndpoint}/queue-status`);
            const statusData = await statusResponse.json();
            
            if (statusData.success) {
                this.updateStatusDisplay(statusData.queue_status);
            }
            
            // Get recent operations
            const opsResponse = await fetch(`${this.config.apiEndpoint}/list?limit=${this.config.maxDisplayedOperations}`);
            const opsData = await opsResponse.json();
            
            if (opsData.success) {
                this.updateOperationsDisplay(opsData.operations);
            }
            
        } catch (error) {
            console.error('Status refresh error:', error);
            this.showNotification('Failed to refresh status', 'error');
        }
    }
    
    /**
     * Update status display
     */
    updateStatusDisplay(queueStatus) {
        const content = this.statusContainer.querySelector('.batch-status-content');
        
        const html = `
            <div class="batch-status-grid">
                <div class="status-item">
                    <span class="status-label">Queue Size:</span>
                    <span class="status-value">${queueStatus.queue_size}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Processing:</span>
                    <span class="status-value">${queueStatus.processing_count}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Total Operations:</span>
                    <span class="status-value">${queueStatus.total_operations}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Workers:</span>
                    <span class="status-value">${queueStatus.worker_threads}</span>
                </div>
            </div>
            <div class="batch-status-breakdown">
                <h5>Status Breakdown:</h5>
                ${Object.entries(queueStatus.status_breakdown).map(([status, count]) => 
                    `<span class="status-badge status-${status}">${status}: ${count}</span>`
                ).join('')}
            </div>
        `;
        
        content.innerHTML = html;
    }
    
    /**
     * Update operations display
     */
    updateOperationsDisplay(operations) {
        const content = this.operationsContainer.querySelector('.batch-operations-content');
        
        if (!operations || operations.length === 0) {
            content.innerHTML = '<p class="no-operations">No operations yet</p>';
            return;
        }
        
        const html = operations.map(op => `
            <div class="batch-operation-item status-${op.status}" data-id="${op.id}">
                <div class="operation-header">
                    <span class="operation-type">${this.getOperationIcon(op.type)} ${op.type}</span>
                    <span class="operation-status status-badge status-${op.status}">${op.status}</span>
                </div>
                <div class="operation-details">
                    <small>ID: ${op.id.substring(0, 8)}...</small>
                    <small>Created: ${new Date(op.created_at).toLocaleTimeString()}</small>
                    ${op.processing_time ? `<small>Time: ${op.processing_time.toFixed(2)}s</small>` : ''}
                </div>
                ${op.error ? `<div class="operation-error">❌ ${op.error}</div>` : ''}
                ${op.status === 'processing' ? `<div class="operation-progress">Progress: ${op.progress || 0}%</div>` : ''}
            </div>
        `).join('');
        
        content.innerHTML = html;
    }
    
    /**
     * Get icon for operation type
     */
    getOperationIcon(type) {
        return this.operationTemplates[type]?.icon || '⚙️';
    }
    
    /**
     * Show notification
     */
    showNotification(message, type = 'info') {
        if (!this.config.showNotifications) return;
        
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `batch-notification batch-notification-${type}`;
        notification.textContent = message;
        
        // Position and show
        document.body.appendChild(notification);
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 3000);
        
        console.log(`Batch notification: ${message}`);
    }
    
    /**
     * Toggle batch panel visibility
     */
    togglePanel() {
        this.isEnabled = !this.isEnabled;
        
        if (this.isEnabled) {
            this.showPanel();
        } else {
            this.hidePanel();
        }
        
        this.saveSettings();
    }
    
    /**
     * Show batch panel
     */
    showPanel() {
        this.batchPanel.style.display = 'block';
        this.toggleButton.style.opacity = '1';
        this.toggleButton.title = 'Hide Batch Operations';
        
        // Start polling
        if (this.config.autoRefresh) {
            this.startPolling();
        }
        
        // Initial refresh
        this.refreshStatus();
        
        console.log('Batch panel shown');
    }
    
    /**
     * Hide batch panel
     */
    hidePanel() {
        this.batchPanel.style.display = 'none';
        this.toggleButton.style.opacity = '0.6';
        this.toggleButton.title = 'Show Batch Operations';
        
        // Stop polling
        this.stopPolling();
        
        console.log('Batch panel hidden');
    }
    
    /**
     * Start status polling
     */
    startPolling() {
        if (this.pollInterval) return;
        
        this.pollInterval = setInterval(() => {
            if (this.isEnabled && this.batchPanel.style.display !== 'none') {
                this.refreshStatus();
            }
        }, this.pollFrequency);
        
        console.log('Status polling started');
    }
    
    /**
     * Stop status polling
     */
    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
            console.log('Status polling stopped');
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
            localStorage.setItem('batchOperationsSettings', JSON.stringify(settings));
        } catch (error) {
            console.warn('Failed to save batch operations settings:', error);
        }
    }
    
    /**
     * Load settings from localStorage
     */
    loadSettings() {
        try {
            const saved = localStorage.getItem('batchOperationsSettings');
            if (saved) {
                const settings = JSON.parse(saved);
                this.isEnabled = settings.enabled !== false;
                
                if (settings.config) {
                    Object.assign(this.config, settings.config);
                }
            } else {
                this.isEnabled = false; // Default disabled
            }
            
            // Update UI
            if (this.toggleButton) {
                this.toggleButton.style.opacity = this.isEnabled ? '1' : '0.6';
            }
            
        } catch (error) {
            console.warn('Failed to load batch operations settings:', error);
            this.isEnabled = false;
        }
    }
    
    /**
     * Get current status
     */
    getStatus() {
        return {
            enabled: this.isEnabled,
            activeBatches: this.activeBatches.size,
            hasPanel: !!this.batchPanel,
            polling: !!this.pollInterval
        };
    }
    
    /**
     * Cleanup resources
     */
    destroy() {
        // Stop polling
        this.stopPolling();
        
        // Remove UI elements
        if (this.batchPanel) {
            this.batchPanel.remove();
        }
        
        if (this.toggleButton) {
            this.toggleButton.remove();
        }
        
        // Clear data
        this.activeBatches.clear();
        
        console.log('Batch Operations Manager destroyed');
    }
}

// Auto-initialize when DOM is ready
function initBatchOperations() {
    try {
        const manager = new BatchOperationsManager();
        const initialized = manager.init();
        
        if (initialized) {
            // Store global reference for debugging
            window.batchOperationsManager = manager;
            return manager;
        } else {
            console.warn('Batch Operations: initialization failed');
            return null;
        }
        
    } catch (error) {
        console.error('Batch Operations initialization error:', error);
        return null;
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initBatchOperations);
} else {
    initBatchOperations();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BatchOperationsManager;
}