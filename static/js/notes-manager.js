/**
 * Notes Manager Module
 * AI-powered note-taking system with smart categorization and organization.
 * Safe implementation that captures AI responses without interfering with chat functionality.
 */

class NotesManager {
    constructor() {
        this.baseUrl = '/api/notes';
        this.notes = [];
        this.categories = {};
        this.tags = {};
        this.currentNote = null;
        this.searchResults = [];
        this.isVisible = false;
        
        // Configuration
        this.config = {
            autoSaveEnabled: true,
            autoTagEnabled: true,
            autoCategorizeEnabled: true,
            maxNotesDisplay: 50,
            searchDelay: 300,
            notePreviewLength: 150
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
            console.log('📝 Initializing Notes Manager...');
            
            // Create notes UI
            this.createNotesUI();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Load initial data
            await this.loadNotes();
            await this.loadMetadata();
            
            // Set up AI response monitoring
            this.setupAIResponseMonitoring();
            
            console.log('✅ Notes Manager initialized');
        } catch (error) {
            console.error('Notes initialization error:', error);
        }
    }
    
    createNotesUI() {
        try {
            // Create notes sidebar
            this.createNotesSidebar();
            
            // Create save buttons for AI responses
            this.createSaveButtons();
            
            // Add notes styles
            this.addNotesStyles();
            
        } catch (error) {
            console.error('Notes UI creation error:', error);
        }
    }
    
    createNotesSidebar() {
        try {
            // Find main container
            const container = document.querySelector('.container') ||
                            document.querySelector('.main-content') ||
                            document.body;
            
            // Create notes sidebar
            const notesSidebar = document.createElement('div');
            notesSidebar.className = 'notes-sidebar';
            notesSidebar.id = 'notes-sidebar';
            notesSidebar.innerHTML = `
                <div class="notes-header">
                    <h3 class="notes-title">
                        <i class="notes-icon">📝</i>
                        My Notes
                    </h3>
                    <div class="notes-header-actions">
                        <button class="notes-btn notes-btn-sm" id="notes-new-btn" title="New Note">
                            ➕
                        </button>
                        <button class="notes-btn notes-btn-sm" id="notes-search-btn" title="Search Notes">
                            🔍
                        </button>
                        <button class="notes-btn notes-btn-sm" id="notes-close-btn" title="Close Notes">
                            ✕
                        </button>
                    </div>
                </div>
                
                <div class="notes-content">
                    <!-- Search Section -->
                    <div class="notes-search-section" id="notes-search-section" style="display: none;">
                        <div class="notes-search-box">
                            <input type="text" id="notes-search-input" class="notes-input" 
                                   placeholder="Search notes...">
                            <button class="notes-btn notes-btn-sm" id="notes-search-clear">
                                ✕
                            </button>
                        </div>
                        <div class="notes-filters">
                            <select id="notes-category-filter" class="notes-select">
                                <option value="">All Categories</option>
                            </select>
                            <select id="notes-tag-filter" class="notes-select">
                                <option value="">All Tags</option>
                            </select>
                        </div>
                    </div>
                    
                    <!-- Stats Section -->
                    <div class="notes-stats" id="notes-stats">
                        <div class="notes-stat">
                            <span class="notes-stat-number" id="notes-count">0</span>
                            <span class="notes-stat-label">Notes</span>
                        </div>
                        <div class="notes-stat">
                            <span class="notes-stat-number" id="notes-words">0</span>
                            <span class="notes-stat-label">Words</span>
                        </div>
                    </div>
                    
                    <!-- Notes List -->
                    <div class="notes-list-container">
                        <div class="notes-list" id="notes-list">
                            <div class="notes-loading">Loading notes...</div>
                        </div>
                    </div>
                    
                    <!-- Categories Section -->
                    <div class="notes-categories" id="notes-categories">
                        <h4 class="notes-section-title">Categories</h4>
                        <div class="notes-categories-list" id="notes-categories-list">
                            <!-- Categories will be populated here -->
                        </div>
                    </div>
                    
                    <!-- Quick Actions -->
                    <div class="notes-actions">
                        <button class="notes-btn notes-btn-outline" id="notes-export-btn">
                            Export Notes
                        </button>
                        <button class="notes-btn notes-btn-outline" id="notes-settings-btn">
                            Settings
                        </button>
                    </div>
                </div>
            `;
            
            // Add to container
            container.appendChild(notesSidebar);
            
            // Create toggle button
            this.createNotesToggle();
            
        } catch (error) {
            console.error('Notes sidebar creation error:', error);
        }
    }
    
    createNotesToggle() {
        try {
            // Find a good place for the toggle button
            const chatContainer = document.querySelector('.chat-container') ||
                                document.querySelector('.main-content') ||
                                document.querySelector('.container');
            
            if (!chatContainer) return;
            
            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'notes-toggle-btn';
            toggleBtn.id = 'notes-toggle-btn';
            toggleBtn.innerHTML = '📝';
            toggleBtn.title = 'Toggle Notes';
            
            toggleBtn.addEventListener('click', () => this.toggleNotesSidebar());
            
            chatContainer.appendChild(toggleBtn);
            
        } catch (error) {
            console.error('Notes toggle creation error:', error);
        }
    }
    
    createSaveButtons() {
        try {
            // Use MutationObserver to add save buttons to AI responses
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    mutation.addedNodes.forEach((node) => {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            this.addSaveButtonToResponse(node);
                        }
                    });
                });
            });
            
            // Observe chat messages container
            const chatMessages = document.getElementById('chatMessages') ||
                               document.querySelector('.chat-messages') ||
                               document.querySelector('.messages');
            
            if (chatMessages) {
                observer.observe(chatMessages, {
                    childList: true,
                    subtree: true
                });
                
                // Add save buttons to existing messages
                this.addSaveButtonsToExistingMessages();
            }
            
        } catch (error) {
            console.error('Save buttons creation error:', error);
        }
    }
    
    addSaveButtonToResponse(element) {
        try {
            // Check if this is an AI response
            const isAIResponse = element.classList?.contains('ai-message') ||
                               element.classList?.contains('assistant-message') ||
                               element.querySelector('.ai-message') ||
                               element.querySelector('.assistant-message') ||
                               (element.textContent && this.looksLikeAIResponse(element.textContent));
            
            if (!isAIResponse) return;
            
            // Don't add button if already exists
            if (element.querySelector('.notes-save-btn')) return;
            
            // Create save button
            const saveBtn = document.createElement('button');
            saveBtn.className = 'notes-save-btn';
            saveBtn.innerHTML = '📝 Save';
            saveBtn.title = 'Save this response as a note';
            
            saveBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.saveResponseAsNote(element);
            });
            
            // Find good place to insert button
            const messageContent = element.querySelector('.message-content') ||
                                 element.querySelector('.content') ||
                                 element;
            
            if (messageContent) {
                // Create button container
                const btnContainer = document.createElement('div');
                btnContainer.className = 'notes-message-actions';
                btnContainer.appendChild(saveBtn);
                
                messageContent.appendChild(btnContainer);
            }
            
        } catch (error) {
            console.error('Save button addition error:', error);
        }
    }
    
    addSaveButtonsToExistingMessages() {
        try {
            const chatMessages = document.getElementById('chatMessages') ||
                               document.querySelector('.chat-messages');
            
            if (!chatMessages) return;
            
            // Find all AI messages
            const aiMessages = chatMessages.querySelectorAll('.ai-message, .assistant-message') ||
                             Array.from(chatMessages.children).filter(child => 
                                 this.looksLikeAIResponse(child.textContent)
                             );
            
            aiMessages.forEach(message => this.addSaveButtonToResponse(message));
            
        } catch (error) {
            console.error('Existing save buttons error:', error);
        }
    }
    
    looksLikeAIResponse(text) {
        if (!text || text.length < 20) return false;
        
        // Simple heuristics to identify AI responses
        const aiIndicators = [
            /^(I can|I'll|I'd|I will|I would|I understand|I see|I think)/i,
            /^(Here's|Here are|This is|These are|Let me)/i,
            /^(To|In order to|For|When|If you|You can)/i,
            /(explains?|suggests?|recommends?|proposes?)/i,
            /^(Based on|According to|Given|Considering)/i
        ];
        
        return aiIndicators.some(pattern => pattern.test(text.trim()));
    }
    
    addNotesStyles() {
        try {
            // Only add styles once
            if (document.querySelector('#notes-manager-styles')) {
                return;
            }
            
            const styles = document.createElement('style');
            styles.id = 'notes-manager-styles';
            styles.textContent = `
                /* Notes Manager Styles - Unique prefixes to avoid conflicts */
                .notes-sidebar {
                    position: fixed;
                    top: 0;
                    right: -400px;
                    width: 400px;
                    height: 100vh;
                    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                    border-left: 2px solid #444;
                    box-shadow: -5px 0 20px rgba(0,0,0,0.3);
                    z-index: 1000;
                    transition: right 0.3s ease;
                    overflow: hidden;
                    display: flex;
                    flex-direction: column;
                }
                
                .notes-sidebar.visible {
                    right: 0;
                }
                
                .notes-header {
                    padding: 16px 20px;
                    background: linear-gradient(135deg, #333 0%, #444 100%);
                    border-bottom: 2px solid #555;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    flex-shrink: 0;
                }
                
                .notes-title {
                    margin: 0;
                    color: #e5e5e5;
                    font-size: 1.2rem;
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                
                .notes-icon {
                    font-size: 1.3rem;
                }
                
                .notes-header-actions {
                    display: flex;
                    gap: 8px;
                }
                
                .notes-content {
                    flex: 1;
                    padding: 16px 20px;
                    overflow-y: auto;
                    display: flex;
                    flex-direction: column;
                    gap: 16px;
                }
                
                .notes-btn {
                    padding: 8px 16px;
                    border: none;
                    border-radius: 6px;
                    font-size: 0.9rem;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    text-decoration: none;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    gap: 6px;
                    background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                    color: white;
                }
                
                .notes-btn:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(0,123,255,0.3);
                }
                
                .notes-btn-sm {
                    padding: 6px 12px;
                    font-size: 0.8rem;
                    min-width: 32px;
                    min-height: 32px;
                }
                
                .notes-btn-outline {
                    background: transparent;
                    color: #adb5bd;
                    border: 1px solid #6c757d;
                }
                
                .notes-btn-outline:hover {
                    background: rgba(108, 117, 125, 0.1);
                    color: #e5e5e5;
                    box-shadow: 0 2px 8px rgba(108, 117, 125, 0.2);
                }
                
                .notes-search-section {
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }
                
                .notes-search-box {
                    display: flex;
                    gap: 8px;
                    align-items: center;
                }
                
                .notes-input, .notes-select {
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid #555;
                    border-radius: 6px;
                    padding: 8px 12px;
                    color: #e5e5e5;
                    font-size: 0.9rem;
                    transition: all 0.3s ease;
                    flex: 1;
                }
                
                .notes-input:focus, .notes-select:focus {
                    outline: none;
                    border-color: #007bff;
                    background: rgba(255, 255, 255, 0.15);
                    box-shadow: 0 0 8px rgba(0,123,255,0.3);
                }
                
                .notes-input::placeholder {
                    color: #adb5bd;
                }
                
                .notes-filters {
                    display: flex;
                    gap: 8px;
                }
                
                .notes-stats {
                    display: flex;
                    gap: 16px;
                    padding: 12px;
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 8px;
                    border: 1px solid #444;
                }
                
                .notes-stat {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    text-align: center;
                }
                
                .notes-stat-number {
                    font-size: 1.2rem;
                    font-weight: 600;
                    color: #007bff;
                }
                
                .notes-stat-label {
                    font-size: 0.8rem;
                    color: #adb5bd;
                    margin-top: 2px;
                }
                
                .notes-list-container {
                    flex: 1;
                    min-height: 200px;
                }
                
                .notes-list {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                    max-height: 300px;
                    overflow-y: auto;
                }
                
                .notes-item {
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid #444;
                    border-radius: 8px;
                    padding: 12px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }
                
                .notes-item:hover {
                    background: rgba(255, 255, 255, 0.1);
                    border-color: #555;
                    transform: translateY(-1px);
                }
                
                .notes-item.selected {
                    border-color: #007bff;
                    background: rgba(0, 123, 255, 0.1);
                }
                
                .notes-item-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                    margin-bottom: 8px;
                }
                
                .notes-item-title {
                    font-weight: 500;
                    color: #e5e5e5;
                    font-size: 0.9rem;
                    line-height: 1.3;
                    flex: 1;
                    margin-right: 8px;
                }
                
                .notes-item-date {
                    font-size: 0.75rem;
                    color: #adb5bd;
                    white-space: nowrap;
                }
                
                .notes-item-preview {
                    font-size: 0.8rem;
                    color: #ced4da;
                    line-height: 1.4;
                    margin-bottom: 8px;
                    display: -webkit-box;
                    -webkit-line-clamp: 2;
                    -webkit-box-orient: vertical;
                    overflow: hidden;
                }
                
                .notes-item-meta {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: 0.75rem;
                }
                
                .notes-item-category {
                    color: #007bff;
                    background: rgba(0, 123, 255, 0.2);
                    padding: 2px 6px;
                    border-radius: 4px;
                }
                
                .notes-item-tags {
                    display: flex;
                    gap: 4px;
                    flex-wrap: wrap;
                }
                
                .notes-tag {
                    color: #28a745;
                    background: rgba(40, 167, 69, 0.2);
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-size: 0.7rem;
                }
                
                .notes-categories {
                    border-top: 1px solid #444;
                    padding-top: 16px;
                }
                
                .notes-section-title {
                    margin: 0 0 12px 0;
                    color: #e5e5e5;
                    font-size: 1rem;
                    font-weight: 600;
                }
                
                .notes-categories-list {
                    display: flex;
                    flex-direction: column;
                    gap: 6px;
                }
                
                .notes-category-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 6px 8px;
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 4px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                }
                
                .notes-category-item:hover {
                    background: rgba(255, 255, 255, 0.08);
                }
                
                .notes-category-name {
                    color: #ced4da;
                    font-size: 0.85rem;
                }
                
                .notes-category-count {
                    color: #adb5bd;
                    font-size: 0.75rem;
                    background: rgba(255, 255, 255, 0.1);
                    padding: 2px 6px;
                    border-radius: 10px;
                }
                
                .notes-actions {
                    border-top: 1px solid #444;
                    padding-top: 16px;
                    display: flex;
                    gap: 8px;
                }
                
                .notes-toggle-btn {
                    position: fixed;
                    top: 120px;
                    right: 20px;
                    width: 50px;
                    height: 50px;
                    border-radius: 50%;
                    border: none;
                    background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                    color: white;
                    font-size: 1.5rem;
                    cursor: pointer;
                    box-shadow: 0 4px 16px rgba(0,123,255,0.3);
                    z-index: 999;
                    transition: all 0.3s ease;
                }
                
                .notes-toggle-btn:hover {
                    transform: scale(1.1);
                    box-shadow: 0 6px 20px rgba(0,123,255,0.4);
                }
                
                .notes-save-btn {
                    background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-size: 0.75rem;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    margin-top: 8px;
                    align-self: flex-start;
                }
                
                .notes-save-btn:hover {
                    background: linear-gradient(135deg, #1e7e34 0%, #155724 100%);
                    transform: translateY(-1px);
                }
                
                .notes-message-actions {
                    display: flex;
                    gap: 8px;
                    margin-top: 8px;
                    padding-top: 8px;
                    border-top: 1px solid rgba(255, 255, 255, 0.1);
                }
                
                .notes-loading {
                    text-align: center;
                    color: #adb5bd;
                    padding: 20px;
                    font-style: italic;
                }
                
                .notes-error {
                    background: rgba(220, 53, 69, 0.2);
                    border: 1px solid #dc3545;
                    color: #f8d7da;
                    padding: 12px;
                    border-radius: 6px;
                    font-size: 0.9rem;
                }
                
                .notes-success {
                    background: rgba(40, 167, 69, 0.2);
                    border: 1px solid #28a745;
                    color: #d4edda;
                    padding: 12px;
                    border-radius: 6px;
                    font-size: 0.9rem;
                }
                
                /* Responsive design */
                @media (max-width: 768px) {
                    .notes-sidebar {
                        width: 100%;
                        right: -100%;
                    }
                    
                    .notes-toggle-btn {
                        bottom: 20px;
                        top: auto;
                        right: 20px;
                    }
                    
                    .notes-filters {
                        flex-direction: column;
                    }
                    
                    .notes-stats {
                        justify-content: space-around;
                    }
                }
                
                /* Animations */
                @keyframes notes-fade-in {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                
                .notes-item {
                    animation: notes-fade-in 0.3s ease;
                }
            `;
            
            document.head.appendChild(styles);
            
        } catch (error) {
            console.error('Notes styles creation error:', error);
        }
    }
    
    setupEventListeners() {
        try {
            // Search functionality
            const searchInput = document.getElementById('notes-search-input');
            if (searchInput) {
                let searchTimeout;
                searchInput.addEventListener('input', (e) => {
                    clearTimeout(searchTimeout);
                    searchTimeout = setTimeout(() => {
                        this.searchNotes(e.target.value);
                    }, this.config.searchDelay);
                });
            }
            
            // Search toggle
            const searchBtn = document.getElementById('notes-search-btn');
            if (searchBtn) {
                searchBtn.addEventListener('click', () => this.toggleSearch());
            }
            
            // Search clear
            const searchClear = document.getElementById('notes-search-clear');
            if (searchClear) {
                searchClear.addEventListener('click', () => this.clearSearch());
            }
            
            // Close button
            const closeBtn = document.getElementById('notes-close-btn');
            if (closeBtn) {
                closeBtn.addEventListener('click', () => this.toggleNotesSidebar());
            }
            
            // New note button
            const newBtn = document.getElementById('notes-new-btn');
            if (newBtn) {
                newBtn.addEventListener('click', () => this.createNewNote());
            }
            
            // Export button
            const exportBtn = document.getElementById('notes-export-btn');
            if (exportBtn) {
                exportBtn.addEventListener('click', () => this.exportNotes());
            }
            
            // Category filter
            const categoryFilter = document.getElementById('notes-category-filter');
            if (categoryFilter) {
                categoryFilter.addEventListener('change', (e) => {
                    this.filterByCategory(e.target.value);
                });
            }
            
            // Tag filter
            const tagFilter = document.getElementById('notes-tag-filter');
            if (tagFilter) {
                tagFilter.addEventListener('change', (e) => {
                    this.filterByTag(e.target.value);
                });
            }
            
        } catch (error) {
            console.error('Event listeners setup error:', error);
        }
    }
    
    setupAIResponseMonitoring() {
        try {
            // Monitor for new AI responses to offer saving
            document.addEventListener('ai-response-complete', (e) => {
                if (e.detail && e.detail.element) {
                    this.addSaveButtonToResponse(e.detail.element);
                }
            });
            
            // Monitor for message additions
            const chatContainer = document.getElementById('chatMessages');
            if (chatContainer) {
                const observer = new MutationObserver((mutations) => {
                    mutations.forEach((mutation) => {
                        mutation.addedNodes.forEach((node) => {
                            if (node.nodeType === Node.ELEMENT_NODE) {
                                setTimeout(() => this.addSaveButtonToResponse(node), 500);
                            }
                        });
                    });
                });
                
                observer.observe(chatContainer, { childList: true });
            }
            
        } catch (error) {
            console.error('AI response monitoring setup error:', error);
        }
    }
    
    toggleNotesSidebar() {
        const sidebar = document.getElementById('notes-sidebar');
        if (!sidebar) return;
        
        this.isVisible = !this.isVisible;
        
        if (this.isVisible) {
            sidebar.classList.add('visible');
            
            // Load notes if not loaded
            if (this.notes.length === 0) {
                this.loadNotes();
            }
        } else {
            sidebar.classList.remove('visible');
        }
    }
    
    toggleSearch() {
        const searchSection = document.getElementById('notes-search-section');
        if (!searchSection) return;
        
        const isVisible = searchSection.style.display !== 'none';
        searchSection.style.display = isVisible ? 'none' : 'block';
        
        if (!isVisible) {
            const searchInput = document.getElementById('notes-search-input');
            if (searchInput) {
                searchInput.focus();
            }
        }
    }
    
    clearSearch() {
        const searchInput = document.getElementById('notes-search-input');
        if (searchInput) {
            searchInput.value = '';
            this.loadNotes(); // Reload all notes
        }
    }
    
    async loadNotes() {
        try {
            const response = await fetch(`${this.baseUrl}/list?limit=${this.config.maxNotesDisplay}`);
            const data = await response.json();
            
            if (data.success) {
                this.notes = data.notes || [];
                this.displayNotes(this.notes);
                this.updateStats();
            } else {
                this.showError('Failed to load notes');
            }
            
        } catch (error) {
            console.error('Notes loading error:', error);
            this.showError('Error loading notes');
        }
    }
    
    async loadMetadata() {
        try {
            // Load categories
            const categoriesResponse = await fetch(`${this.baseUrl}/categories`);
            const categoriesData = await categoriesResponse.json();
            
            if (categoriesData.success) {
                this.categories = categoriesData.categories || {};
                this.updateCategoriesDisplay();
                this.updateCategoryFilter();
            }
            
            // Load tags
            const tagsResponse = await fetch(`${this.baseUrl}/tags`);
            const tagsData = await tagsResponse.json();
            
            if (tagsData.success) {
                this.tags = tagsData.tags || {};
                this.updateTagFilter();
            }
            
        } catch (error) {
            console.error('Metadata loading error:', error);
        }
    }
    
    displayNotes(notes) {
        const notesList = document.getElementById('notes-list');
        if (!notesList) return;
        
        if (!notes || notes.length === 0) {
            notesList.innerHTML = '<div class="notes-loading">No notes found</div>';
            return;
        }
        
        const notesHtml = notes.map(note => this.renderNoteItem(note)).join('');
        notesList.innerHTML = notesHtml;
        
        // Add click listeners
        notesList.querySelectorAll('.notes-item').forEach(item => {
            item.addEventListener('click', () => {
                const noteId = item.dataset.noteId;
                this.selectNote(noteId);
            });
        });
    }
    
    renderNoteItem(note) {
        const createdDate = new Date(note.created_at).toLocaleDateString();
        const preview = this.truncateText(note.content, this.config.notePreviewLength);
        const tags = note.tags || [];
        
        return `
            <div class="notes-item" data-note-id="${note.id}">
                <div class="notes-item-header">
                    <div class="notes-item-title">${note.title}</div>
                    <div class="notes-item-date">${createdDate}</div>
                </div>
                <div class="notes-item-preview">${preview}</div>
                <div class="notes-item-meta">
                    <div class="notes-item-category">${note.category}</div>
                    <div class="notes-item-tags">
                        ${tags.slice(0, 3).map(tag => `<span class="notes-tag">${tag}</span>`).join('')}
                        ${tags.length > 3 ? `<span class="notes-tag">+${tags.length - 3}</span>` : ''}
                    </div>
                </div>
            </div>
        `;
    }
    
    updateCategoriesDisplay() {
        const categoriesList = document.getElementById('notes-categories-list');
        if (!categoriesList) return;
        
        const categoriesHtml = Object.entries(this.categories)
            .sort((a, b) => b[1] - a[1]) // Sort by count
            .map(([name, count]) => `
                <div class="notes-category-item" data-category="${name}">
                    <span class="notes-category-name">${name}</span>
                    <span class="notes-category-count">${count}</span>
                </div>
            `).join('');
        
        categoriesList.innerHTML = categoriesHtml;
        
        // Add click listeners
        categoriesList.querySelectorAll('.notes-category-item').forEach(item => {
            item.addEventListener('click', () => {
                const category = item.dataset.category;
                this.filterByCategory(category);
            });
        });
    }
    
    updateCategoryFilter() {
        const categoryFilter = document.getElementById('notes-category-filter');
        if (!categoryFilter) return;
        
        const options = Object.keys(this.categories)
            .map(category => `<option value="${category}">${category}</option>`)
            .join('');
        
        categoryFilter.innerHTML = '<option value="">All Categories</option>' + options;
    }
    
    updateTagFilter() {
        const tagFilter = document.getElementById('notes-tag-filter');
        if (!tagFilter) return;
        
        const options = Object.keys(this.tags)
            .sort()
            .map(tag => `<option value="${tag}">${tag}</option>`)
            .join('');
        
        tagFilter.innerHTML = '<option value="">All Tags</option>' + options;
    }
    
    updateStats() {
        const notesCount = document.getElementById('notes-count');
        const notesWords = document.getElementById('notes-words');
        
        if (notesCount) {
            notesCount.textContent = this.notes.length;
        }
        
        if (notesWords) {
            const totalWords = this.notes.reduce((sum, note) => sum + (note.word_count || 0), 0);
            notesWords.textContent = totalWords.toLocaleString();
        }
    }
    
    async saveResponseAsNote(element) {
        try {
            // Extract content from the response element
            const content = this.extractResponseContent(element);
            
            if (!content || content.length < 10) {
                this.showError('Response too short to save');
                return;
            }
            
            // Show saving feedback
            const saveBtn = element.querySelector('.notes-save-btn');
            if (saveBtn) {
                const originalText = saveBtn.innerHTML;
                saveBtn.innerHTML = '💾 Saving...';
                saveBtn.disabled = true;
            }
            
            // Create note
            const response = await fetch(`${this.baseUrl}/create`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    content: content,
                    source: {
                        type: 'ai_response',
                        timestamp: new Date().toISOString(),
                        element_type: element.tagName
                    }
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showSuccess('Response saved as note!');
                
                // Update save button
                if (saveBtn) {
                    saveBtn.innerHTML = '✅ Saved';
                    saveBtn.style.background = '#28a745';
                    setTimeout(() => {
                        saveBtn.innerHTML = '📝 Save';
                        saveBtn.style.background = '';
                        saveBtn.disabled = false;
                    }, 3000);
                }
                
                // Reload notes if sidebar is visible
                if (this.isVisible) {
                    await this.loadNotes();
                    await this.loadMetadata();
                }
                
            } else {
                throw new Error(data.error || 'Failed to save note');
            }
            
        } catch (error) {
            console.error('Save response error:', error);
            this.showError('Failed to save response: ' + error.message);
            
            // Reset save button
            const saveBtn = element.querySelector('.notes-save-btn');
            if (saveBtn) {
                saveBtn.innerHTML = '📝 Save';
                saveBtn.disabled = false;
            }
        }
    }
    
    extractResponseContent(element) {
        try {
            // Get text content, excluding the save button
            const clone = element.cloneNode(true);
            
            // Remove save button and other action elements
            const actionsToRemove = clone.querySelectorAll('.notes-save-btn, .notes-message-actions');
            actionsToRemove.forEach(action => action.remove());
            
            // Get clean text content
            let content = clone.textContent || clone.innerText || '';
            
            // Clean up whitespace
            content = content.replace(/\s+/g, ' ').trim();
            
            return content;
            
        } catch (error) {
            console.error('Content extraction error:', error);
            return '';
        }
    }
    
    async searchNotes(query) {
        try {
            if (!query || query.length < 2) {
                this.loadNotes(); // Show all notes
                return;
            }
            
            const response = await fetch(`${this.baseUrl}/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.searchResults = data.results || [];
                this.displayNotes(this.searchResults);
            } else {
                this.showError('Search failed');
            }
            
        } catch (error) {
            console.error('Search error:', error);
            this.showError('Search error');
        }
    }
    
    async filterByCategory(category) {
        try {
            const params = new URLSearchParams();
            if (category) params.append('category', category);
            
            const response = await fetch(`${this.baseUrl}/list?${params.toString()}`);
            const data = await response.json();
            
            if (data.success) {
                this.displayNotes(data.notes || []);
            }
            
        } catch (error) {
            console.error('Category filter error:', error);
        }
    }
    
    async filterByTag(tag) {
        try {
            if (!tag) {
                this.loadNotes();
                return;
            }
            
            const response = await fetch(`${this.baseUrl}/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: '',
                    filters: { tags: [tag] }
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.displayNotes(data.results || []);
            }
            
        } catch (error) {
            console.error('Tag filter error:', error);
        }
    }
    
    selectNote(noteId) {
        try {
            // Highlight selected note
            document.querySelectorAll('.notes-item').forEach(item => {
                item.classList.remove('selected');
            });
            
            const selectedItem = document.querySelector(`[data-note-id="${noteId}"]`);
            if (selectedItem) {
                selectedItem.classList.add('selected');
            }
            
            this.currentNote = noteId;
            
            // Could open note editor here in future
            console.log('Note selected:', noteId);
            
        } catch (error) {
            console.error('Note selection error:', error);
        }
    }
    
    async createNewNote() {
        try {
            const title = prompt('Note title:');
            if (!title) return;
            
            const content = prompt('Note content:');
            if (!content) return;
            
            const response = await fetch(`${this.baseUrl}/create`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    title: title,
                    content: content,
                    source: {
                        type: 'manual',
                        timestamp: new Date().toISOString()
                    }
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showSuccess('Note created successfully!');
                await this.loadNotes();
                await this.loadMetadata();
            } else {
                this.showError('Failed to create note');
            }
            
        } catch (error) {
            console.error('Create note error:', error);
            this.showError('Error creating note');
        }
    }
    
    async exportNotes() {
        try {
            const format = prompt('Export format (json, markdown, txt):', 'json');
            if (!format) return;
            
            const response = await fetch(`${this.baseUrl}/export`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ format: format })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showSuccess(`Notes exported: ${data.export_path}`);
            } else {
                this.showError('Export failed');
            }
            
        } catch (error) {
            console.error('Export error:', error);
            this.showError('Export error');
        }
    }
    
    truncateText(text, maxLength) {
        if (!text || text.length <= maxLength) return text || '';
        return text.substring(0, maxLength) + '...';
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
            const existingMessages = document.querySelectorAll('.notes-error, .notes-success');
            existingMessages.forEach(msg => msg.remove());
            
            // Create message element
            const messageEl = document.createElement('div');
            messageEl.className = `notes-${type}`;
            messageEl.textContent = message;
            
            // Find container to show message
            const container = document.querySelector('.notes-content');
            
            if (container) {
                container.insertBefore(messageEl, container.firstChild);
                
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
    
    destroy() {
        try {
            // Remove notes sidebar
            const sidebar = document.getElementById('notes-sidebar');
            if (sidebar) {
                sidebar.remove();
            }
            
            // Remove toggle button
            const toggleBtn = document.getElementById('notes-toggle-btn');
            if (toggleBtn) {
                toggleBtn.remove();
            }
            
            // Remove save buttons
            const saveButtons = document.querySelectorAll('.notes-save-btn');
            saveButtons.forEach(btn => btn.remove());
            
        } catch (error) {
            console.error('Notes manager cleanup error:', error);
        }
    }
}

// Initialize notes manager when script loads
const notesManager = new NotesManager();

// Export for global access
window.notesManager = notesManager;