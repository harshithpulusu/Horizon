/**
 * Multimodal Manager - Self-contained drag-and-drop image handling
 * Does NOT modify existing functionality - works as optional enhancement
 */

class MultiModalManager {
    constructor() {
        this.isEnabled = false;
        this.attachedImages = [];
        this.maxImages = 5;
        this.maxImageSize = 10 * 1024 * 1024; // 10MB
        this.allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
        this.uploadEndpoint = '/api/multimodal/upload';
        
        // Feature detection
        this.isSupported = this.checkSupport();
        
        if (this.isSupported) {
            this.init();
        }
        
        console.log('🖼️ MultiModalManager:', this.isSupported ? 'Initialized' : 'Not supported');
    }

    checkSupport() {
        /**
         * Check if browser supports required features
         */
        return !!(
            window.File && 
            window.FileReader && 
            window.FileList && 
            window.Blob &&
            document.querySelector('.input-area') // Ensure our target elements exist
        );
    }

    init() {
        /**
         * Initialize multimodal features without modifying existing elements
         */
        try {
            this.createDragDropOverlay();
            this.createUploadButton();
            this.createImagePreviewContainer();
            this.setupDragAndDropHandlers();
            this.isEnabled = true;
            
            console.log('✅ MultiModalManager features enabled');
        } catch (error) {
            console.warn('⚠️ MultiModalManager initialization failed:', error);
            this.isEnabled = false;
        }
    }

    createDragDropOverlay() {
        /**
         * Create drag-and-drop overlay (non-intrusive)
         */
        const inputArea = document.querySelector('.input-area');
        if (!inputArea) return;

        const overlay = document.createElement('div');
        overlay.className = 'multimodal-drag-overlay';
        overlay.innerHTML = `
            <div class="multimodal-drag-content">
                <span class="multimodal-drag-icon">🖼️</span>
                <span class="multimodal-drag-text">Drop images here for AI analysis</span>
                <span class="multimodal-drag-subtext">Supports JPG, PNG, GIF, WebP (max 10MB each)</span>
            </div>
        `;
        
        // Add to input area without modifying existing structure
        inputArea.style.position = 'relative';
        inputArea.appendChild(overlay);
    }

    createUploadButton() {
        /**
         * Add image upload button to voice controls area
         */
        const voiceControls = document.querySelector('.voice-controls');
        if (!voiceControls) return;

        // Create file input (hidden)
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.multiple = true;
        fileInput.accept = this.allowedTypes.join(',');
        fileInput.style.display = 'none';
        fileInput.addEventListener('change', (e) => this.handleFileSelection(e));

        // Create upload button
        const uploadButton = document.createElement('button');
        uploadButton.className = 'multimodal-upload-btn';
        uploadButton.innerHTML = '📎';
        uploadButton.title = 'Upload images for AI analysis';
        uploadButton.addEventListener('click', () => fileInput.click());

        // Add to voice controls (non-destructive)
        voiceControls.appendChild(fileInput);
        voiceControls.appendChild(uploadButton);
    }

    createImagePreviewContainer() {
        /**
         * Create container for image previews
         */
        const inputContainer = document.querySelector('.input-container');
        if (!inputContainer) return;

        const previewContainer = document.createElement('div');
        previewContainer.className = 'multimodal-preview-container';
        
        // Insert at the beginning of input container
        inputContainer.insertBefore(previewContainer, inputContainer.firstChild);
    }

    setupDragAndDropHandlers() {
        /**
         * Set up drag and drop event handlers
         */
        const inputArea = document.querySelector('.input-area');
        const overlay = document.querySelector('.multimodal-drag-overlay');
        
        if (!inputArea || !overlay) return;

        // Prevent default behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            document.addEventListener(eventName, this.preventDefaults, false);
            inputArea.addEventListener(eventName, this.preventDefaults, false);
        });

        // Visual feedback
        ['dragenter', 'dragover'].forEach(eventName => {
            inputArea.addEventListener(eventName, () => {
                overlay.classList.add('multimodal-drag-active');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            inputArea.addEventListener(eventName, () => {
                overlay.classList.remove('multimodal-drag-active');
            }, false);
        });

        // Handle file drop
        inputArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            this.handleFiles(files);
        }, false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleFileSelection(event) {
        /**
         * Handle file input selection
         */
        const files = event.target.files;
        this.handleFiles(files);
        
        // Clear file input for repeated uploads
        event.target.value = '';
    }

    handleFiles(files) {
        /**
         * Process selected/dropped files
         */
        if (!files || files.length === 0) return;

        Array.from(files).forEach(file => {
            if (this.validateFile(file)) {
                this.addImageToPreview(file);
            }
        });
    }

    validateFile(file) {
        /**
         * Validate file type and size
         */
        // Check file type
        if (!this.allowedTypes.includes(file.type)) {
            this.showNotification(`Unsupported file type: ${file.type}`, 'error');
            return false;
        }

        // Check file size
        if (file.size > this.maxImageSize) {
            const sizeMB = (file.size / 1024 / 1024).toFixed(2);
            this.showNotification(`File too large: ${sizeMB}MB (max 10MB)`, 'error');
            return false;
        }

        // Check image limit
        if (this.attachedImages.length >= this.maxImages) {
            this.showNotification(`Maximum ${this.maxImages} images allowed`, 'error');
            return false;
        }

        return true;
    }

    addImageToPreview(file) {
        /**
         * Add image to preview and attached images list
         */
        const reader = new FileReader();
        
        reader.onload = (e) => {
            const imageData = {
                id: Date.now() + Math.random(),
                file: file,
                dataUrl: e.target.result,
                name: file.name,
                size: file.size,
                type: file.type
            };

            this.attachedImages.push(imageData);
            this.renderImagePreview(imageData);
            this.updateUploadButtonState();
            
            console.log('📷 Image added:', file.name);
        };

        reader.readAsDataURL(file);
    }

    renderImagePreview(imageData) {
        /**
         * Render image preview in the preview container
         */
        const container = document.querySelector('.multimodal-preview-container');
        if (!container) return;

        const preview = document.createElement('div');
        preview.className = 'multimodal-image-preview';
        preview.dataset.imageId = imageData.id;
        
        preview.innerHTML = `
            <img src="${imageData.dataUrl}" alt="${imageData.name}" title="${imageData.name}">
            <button class="multimodal-remove-btn" title="Remove image">×</button>
            <div class="multimodal-image-info">
                <span class="multimodal-image-name">${imageData.name}</span>
                <span class="multimodal-image-size">${(imageData.size / 1024).toFixed(1)}KB</span>
            </div>
        `;

        // Add remove handler
        const removeBtn = preview.querySelector('.multimodal-remove-btn');
        removeBtn.addEventListener('click', () => this.removeImage(imageData.id));

        container.appendChild(preview);
    }

    removeImage(imageId) {
        /**
         * Remove image from preview and attached images list
         */
        this.attachedImages = this.attachedImages.filter(img => img.id != imageId);
        
        const preview = document.querySelector(`[data-image-id="${imageId}"]`);
        if (preview) {
            preview.remove();
        }

        this.updateUploadButtonState();
        console.log('🗑️ Image removed');
    }

    updateUploadButtonState() {
        /**
         * Update upload button appearance based on attached images
         */
        const uploadBtn = document.querySelector('.multimodal-upload-btn');
        if (!uploadBtn) return;

        if (this.attachedImages.length > 0) {
            uploadBtn.innerHTML = `📎 ${this.attachedImages.length}`;
            uploadBtn.classList.add('multimodal-has-images');
        } else {
            uploadBtn.innerHTML = '📎';
            uploadBtn.classList.remove('multimodal-has-images');
        }
    }

    // PUBLIC API METHODS

    getAttachedImages() {
        /**
         * Get currently attached images
         */
        return this.attachedImages.map(img => ({
            id: img.id,
            name: img.name,
            size: img.size,
            type: img.type,
            dataUrl: img.dataUrl
        }));
    }

    clearAttachedImages() {
        /**
         * Clear all attached images
         */
        this.attachedImages = [];
        
        const container = document.querySelector('.multimodal-preview-container');
        if (container) {
            container.innerHTML = '';
        }

        this.updateUploadButtonState();
        console.log('🧹 All images cleared');
    }

    hasImages() {
        /**
         * Check if there are attached images
         */
        return this.attachedImages.length > 0;
    }

    async uploadImages() {
        /**
         * Upload attached images to the server
         */
        if (this.attachedImages.length === 0) {
            return { success: true, images: [] };
        }

        try {
            const formData = new FormData();
            
            this.attachedImages.forEach((imageData, index) => {
                formData.append('images', imageData.file);
            });

            const response = await fetch(this.uploadEndpoint, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.success) {
                console.log('📤 Images uploaded successfully:', result.images_uploaded);
                return result;
            } else {
                throw new Error(result.error || 'Upload failed');
            }
            
        } catch (error) {
            console.error('❌ Image upload error:', error);
            this.showNotification(`Upload failed: ${error.message}`, 'error');
            return { success: false, error: error.message };
        }
    }

    showNotification(message, type = 'info') {
        /**
         * Show user notification
         */
        const notification = document.createElement('div');
        notification.className = `multimodal-notification multimodal-notification-${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Auto remove after 4 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 4000);
    }

    // STATUS AND DEBUGGING

    getStatus() {
        /**
         * Get current status for debugging
         */
        return {
            isSupported: this.isSupported,
            isEnabled: this.isEnabled,
            attachedImages: this.attachedImages.length,
            maxImages: this.maxImages,
            uploadEndpoint: this.uploadEndpoint
        };
    }
}

// Auto-initialize if DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.MultiModalManager = new MultiModalManager();
    });
} else {
    window.MultiModalManager = new MultiModalManager();
}