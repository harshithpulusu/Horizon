// Enhanced AI Voice Assistant with advanced features

class EnhancedAIVoiceAssistant {
    constructor() {
        this.isListening = false;
        this.isSpeaking = false;
        this.recognition = null;
        this.wakeWordRecognition = null;  // New: separate recognition for wake words
        this.synthesis = window.speechSynthesis;
        this.currentPersonality = 'friendly';
        this.conversationHistory = [];
        this.activeTimers = [];
        this.activeReminders = [];
        this.lastResponse = '';
        this.lastInput = '';
        this.isWakeWordMode = false;  // New: wake word listening mode
        this.wakeWords = ['hey horizon', 'horizon', 'hey assistant', 'assistant'];  // New: wake words
        this.wakeWordSensitivity = 0.7;  // New: sensitivity threshold
        this.sessionId = null;  // New: conversation session tracking
        this.conversationLength = 0;  // New: track conversation length
        this.contextUsed = false;  // New: track if context was used in responses
        
        this.init();
        this.initEventListeners();
        this.loadActiveTimersReminders();
        this.startPeriodicUpdates();
        this.initializeSession();  // New: initialize conversation session
    }
    
    init() {
        // Get DOM elements first
        this.startBtn = document.getElementById('startListening');
        this.stopBtn = document.getElementById('stopListening');
        this.voiceInput = document.getElementById('voiceInput');
        this.sendBtn = document.getElementById('sendMessage');
        this.messagesContainer = document.getElementById('messages');
        this.personalitySelect = document.getElementById('personalitySelect');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.confidenceScore = document.getElementById('confidenceScore');
        this.conversationCount = document.getElementById('conversationCount');
        this.timersContainer = document.getElementById('activeTimers');
        this.remindersContainer = document.getElementById('activeReminders');
        this.feedbackContainer = document.getElementById('feedbackContainer');
        
        // Initialize speech recognition with better error handling
        this.initSpeechRecognition();
        this.initWakeWordDetection();  // New: Initialize wake word detection
        
        // Ensure personality selector is set to friendly by default
        if (this.personalitySelect) {
            this.personalitySelect.value = this.currentPersonality;
        }
        
        this.updateStatus('Ready');
    }
    
    initSpeechRecognition() {
        // Check if we're on HTTPS or localhost (required for speech recognition)
        const isSecure = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
        
        if (!isSecure) {
            this.addMessage('System', 'Speech recognition requires HTTPS or localhost. Voice features disabled.', 'error');
            this.disableVoiceFeatures();
            return;
        }
        
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            try {
                const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
                this.recognition = new SpeechRecognition();
                this.recognition.continuous = false;
                this.recognition.interimResults = false;
                this.recognition.lang = 'en-US';
                this.recognition.maxAlternatives = 1;
                
                this.recognition.onstart = () => {
                    console.log('Speech recognition started');
                    this.updateStatus('Listening... Speak now!');
                };
                
                this.recognition.onresult = (event) => this.handleSpeechResult(event);
                this.recognition.onerror = (event) => this.handleSpeechError(event);
                this.recognition.onend = () => this.handleSpeechEnd();
                
                this.addMessage('System', 'Speech recognition initialized successfully! 🎤', 'system');
            } catch (error) {
                console.error('Error initializing speech recognition:', error);
                this.addMessage('System', 'Error initializing speech recognition: ' + error.message, 'error');
                this.disableVoiceFeatures();
            }
        } else {
            this.addMessage('System', 'Speech recognition not supported in this browser. Try Chrome or Safari.', 'error');
            this.disableVoiceFeatures();
        }
    }
    
    disableVoiceFeatures() {
        if (this.startBtn) {
            this.startBtn.disabled = true;
            this.startBtn.textContent = '🎤 Voice Not Available';
        }
        if (this.stopBtn) {
            this.stopBtn.disabled = true;
        }
    }
    
    initWakeWordDetection() {
        // Only initialize if speech recognition is available
        if (!this.recognition) {
            return;
        }
        
        try {
            const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
            this.wakeWordRecognition = new SpeechRecognition();
            this.wakeWordRecognition.continuous = true;  // Keep listening
            this.wakeWordRecognition.interimResults = true;  // Get partial results
            this.wakeWordRecognition.lang = 'en-US';
            this.wakeWordRecognition.maxAlternatives = 3;
            
            this.wakeWordRecognition.onresult = (event) => this.handleWakeWordResult(event);
            this.wakeWordRecognition.onerror = (event) => {
                console.log('Wake word detection error:', event.error);
                // Restart wake word detection on error (unless it's aborted)
                if (event.error !== 'aborted' && this.isWakeWordMode) {
                    setTimeout(() => this.startWakeWordDetection(), 1000);
                }
            };
            
            this.wakeWordRecognition.onend = () => {
                // Restart wake word detection if it should be running
                if (this.isWakeWordMode && !this.isListening) {
                    setTimeout(() => this.startWakeWordDetection(), 100);
                }
            };
            
            console.log('Wake word detection initialized');
            this.addWakeWordToggle();  // Add UI toggle for wake word mode
            
        } catch (error) {
            console.error('Error initializing wake word detection:', error);
        }
    }
    
    addWakeWordToggle() {
        // Add wake word toggle to the UI
        const controlsDiv = document.querySelector('.controls');
        if (controlsDiv && !document.getElementById('wakeWordToggle')) {
            const wakeWordDiv = document.createElement('div');
            wakeWordDiv.className = 'wake-word-controls';
            wakeWordDiv.innerHTML = `
                <label class="wake-word-label">
                    <input type="checkbox" id="wakeWordToggle" class="wake-word-checkbox">
                    <span class="wake-word-text">🌟 Always Listening Mode</span>
                    <small>Say "Hey Horizon" or "Horizon" to activate</small>
                </label>
            `;
            
            // Add some CSS for the wake word toggle
            const style = document.createElement('style');
            style.textContent = `
                .wake-word-controls {
                    margin: 10px 0;
                    padding: 15px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 10px;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
                .wake-word-label {
                    display: flex;
                    flex-direction: column;
                    gap: 5px;
                    cursor: pointer;
                    color: white;
                }
                .wake-word-text {
                    font-weight: 500;
                    font-size: 14px;
                }
                .wake-word-label small {
                    color: rgba(255, 255, 255, 0.7);
                    font-size: 12px;
                }
                .wake-word-checkbox {
                    margin-right: 8px;
                    transform: scale(1.2);
                }
            `;
            document.head.appendChild(style);
            
            controlsDiv.appendChild(wakeWordDiv);
            
            // Add event listener for the toggle
            const toggle = document.getElementById('wakeWordToggle');
            toggle.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.startWakeWordDetection();
                } else {
                    this.stopWakeWordDetection();
                }
            });
        }
    }
    
    startWakeWordDetection() {
        if (!this.wakeWordRecognition || this.isWakeWordMode) {
            return;
        }
        
        try {
            this.isWakeWordMode = true;
            this.wakeWordRecognition.start();
            this.updateStatus('👂 Listening for "Hey Horizon"...');
            console.log('Wake word detection started');
        } catch (error) {
            console.error('Error starting wake word detection:', error);
            this.isWakeWordMode = false;
        }
    }
    
    stopWakeWordDetection() {
        if (!this.wakeWordRecognition || !this.isWakeWordMode) {
            return;
        }
        
        try {
            this.isWakeWordMode = false;
            this.wakeWordRecognition.abort();
            this.updateStatus('Ready');
            console.log('Wake word detection stopped');
        } catch (error) {
            console.error('Error stopping wake word detection:', error);
        }
    }
    
    handleWakeWordResult(event) {
        const results = Array.from(event.results);
        const lastResult = results[results.length - 1];
        
        if (lastResult && lastResult[0]) {
            const transcript = lastResult[0].transcript.toLowerCase().trim();
            const confidence = lastResult[0].confidence || 0;
            
            console.log('Wake word transcript:', transcript, 'Confidence:', confidence);
            
            // Check if any wake word was detected
            const wakeWordDetected = this.wakeWords.some(wakeWord => {
                return transcript.includes(wakeWord.toLowerCase()) && 
                       confidence >= this.wakeWordSensitivity;
            });
            
            if (wakeWordDetected && !this.isListening) {
                console.log('Wake word detected!');
                this.handleWakeWordDetected(transcript);
            }
        }
    }
    
    handleWakeWordDetected(transcript) {
        // Stop wake word detection temporarily
        this.wakeWordRecognition.abort();
        
        // Visual/audio feedback
        this.updateStatus('🌟 Wake word detected! Listening...');
        this.addMessage('System', '🌟 Wake word detected! How can I help?', 'system');
        
        // Start regular listening for the actual command
        setTimeout(() => {
            this.startListening();
        }, 500);
        
        // Restart wake word detection after the command is processed
        setTimeout(() => {
            if (this.isWakeWordMode && !this.isListening) {
                this.startWakeWordDetection();
            }
        }, 10000); // Wait 10 seconds before resuming wake word detection
    }
    
    initEventListeners() {
        // Voice controls
        this.startBtn?.addEventListener('click', () => this.startListening());
        this.stopBtn?.addEventListener('click', () => this.stopListening());
        this.sendBtn?.addEventListener('click', () => this.sendMessage());
        
        // Microphone test
        const testMicBtn = document.getElementById('testMicBtn');
        testMicBtn?.addEventListener('click', () => this.testMicrophone());
        
        // Clear conversation button
        const clearConversationBtn = document.getElementById('clearConversationBtn');
        clearConversationBtn?.addEventListener('click', () => this.clearConversation());
        
        // Text input
        this.voiceInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
        
        // Personality selection
        this.personalitySelect?.addEventListener('change', (e) => {
            this.setPersonality(e.target.value);
        });
        
        // Add quick command listeners directly to existing buttons
        this.initQuickCommandListeners();
        
        // Feedback buttons and other dynamic elements
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('feedback-btn')) {
                this.submitFeedback(e.target.dataset.feedback);
            } else if (e.target.classList.contains('cancel-timer-btn')) {
                const timerId = parseInt(e.target.dataset.timerId);
                this.cancelTimer(timerId);
            } else if (e.target.classList.contains('cancel-reminder-btn')) {
                const reminderId = parseInt(e.target.dataset.reminderId);
                this.cancelReminder(reminderId);
            }
        });
    }
    
    initQuickCommandListeners() {
        // Add listeners to quick command buttons
        const quickCmdButtons = document.querySelectorAll('.quick-cmd-btn');
        console.log('Found quick command buttons:', quickCmdButtons.length);
        
        if (quickCmdButtons.length === 0) {
            console.warn('No quick command buttons found!');
            return;
        }
        
        quickCmdButtons.forEach((button, index) => {
            console.log(`Button ${index}:`, button.dataset.command);
            button.addEventListener('click', (e) => {
                e.preventDefault();
                console.log('Quick command button clicked:', button.dataset.command);
                this.handleQuickCommand(button.dataset.command, button);
            });
        });
    }
    
    async startListening() {
        if (!this.recognition) {
            this.addMessage('AI', '❌ Speech recognition not available. Please check browser permissions or try a different browser.', 'error');
            return;
        }
        
        // Check if already listening
        if (this.isListening) {
            this.addMessage('System', 'Already listening...', 'system');
            return;
        }
        
        try {
            // Request microphone permission explicitly
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                try {
                    await navigator.mediaDevices.getUserMedia({ audio: true });
                    console.log('Microphone permission granted');
                } catch (permissionError) {
                    this.addMessage('AI', '❌ Microphone permission denied. Please allow microphone access and try again.', 'error');
                    return;
                }
            }
            
            this.isListening = true;
            this.updateStatus('Starting...');
            
            // Update UI
            if (this.startBtn) {
                this.startBtn.disabled = true;
                this.startBtn.textContent = '🎤 Listening...';
            }
            if (this.stopBtn) {
                this.stopBtn.disabled = false;
            }
            
            // Add visual feedback
            this.statusIndicator?.classList.add('listening');
            
            // Start recognition
            this.recognition.start();
            
            this.addMessage('System', '🎤 Listening started! Speak now...', 'system');
            
        } catch (error) {
            console.error('Speech recognition error:', error);
            this.addMessage('AI', `❌ Error starting speech recognition: ${error.message}`, 'error');
            this.resetListeningState();
        }
    }
    
    stopListening() {
        if (this.recognition && this.isListening) {
            try {
                this.recognition.stop();
                this.addMessage('System', '⏹️ Stopping speech recognition...', 'system');
            } catch (error) {
                console.error('Error stopping recognition:', error);
            }
        } else {
            this.addMessage('System', 'Not currently listening.', 'system');
        }
        this.resetListeningState();
    }
    
    resetListeningState() {
        this.isListening = false;
        this.updateStatus('Ready');
        
        // Reset buttons
        if (this.startBtn) {
            this.startBtn.disabled = false;
            this.startBtn.textContent = '🎤 Start Listening';
        }
        if (this.stopBtn) {
            this.stopBtn.disabled = true;
        }
        
        // Remove visual feedback
        this.statusIndicator?.classList.remove('listening');
    }
    
    handleSpeechResult(event) {
        console.log('Speech result received:', event);
        
        if (event.results && event.results.length > 0) {
            const transcript = event.results[0][0].transcript;
            const confidence = event.results[0][0].confidence;
            
            console.log('Transcript:', transcript, 'Confidence:', confidence);
            
            this.voiceInput.value = transcript;
            this.addMessage('You', `🎤 "${transcript}" (${Math.round(confidence * 100)}% confidence)`, 'user');
            this.processInput(transcript);
        } else {
            this.addMessage('System', '❌ No speech detected. Please try again.', 'error');
        }
        
        this.resetListeningState();
    }
    
    handleSpeechError(event) {
        console.error('Speech recognition error:', event);
        
        let errorMessage = 'Speech recognition error: ';
        switch(event.error) {
            case 'no-speech':
                errorMessage += 'No speech detected. Please try again.';
                break;
            case 'audio-capture':
                errorMessage += 'Microphone not available. Please check your microphone settings.';
                break;
            case 'not-allowed':
                errorMessage += 'Microphone permission denied. Please allow microphone access.';
                break;
            case 'network':
                errorMessage += 'Network error. Please check your internet connection.';
                break;
            case 'service-not-allowed':
                errorMessage += 'Speech service not allowed. Try using HTTPS.';
                break;
            default:
                errorMessage += event.error;
        }
        
        this.addMessage('AI', `❌ ${errorMessage}`, 'error');
        this.updateStatus(`Error: ${event.error}`);
        this.resetListeningState();
    }
    
    handleSpeechEnd() {
        console.log('Speech recognition ended');
        this.resetListeningState();
        
        // Restart wake word detection if it was enabled
        if (this.isWakeWordMode && !this.isListening) {
            setTimeout(() => {
                this.startWakeWordDetection();
            }, 1000); // Wait 1 second before restarting wake word detection
        }
    }
    
    async sendMessage() {
        const input = this.voiceInput.value.trim();
        if (!input) return;
        
        this.addMessage('You', input, 'user');
        this.voiceInput.value = '';
        await this.processInput(input);
    }
    
    handleQuickCommand(command, buttonElement) {
        // Handle quick command button clicks
        console.log('Quick command clicked:', command);
        
        // Add visual feedback to the clicked button
        if (buttonElement) {
            buttonElement.style.background = 'rgba(69, 183, 209, 0.3)';
            buttonElement.style.transform = 'translateX(8px) scale(0.95)';
            
            setTimeout(() => {
                buttonElement.style.background = '';
                buttonElement.style.transform = '';
            }, 200);
        }
        
        // Add the command to the chat and display it
        this.addMessage('You', command, 'user');
        
        // Process the command
        this.processInput(command);
        
        // Update status
        this.updateStatus('Processing quick command...');
    }
    
    async processInput(input) {
        this.updateStatus('Processing...');
        this.lastInput = input;
        
        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    input: input,
                    personality: this.currentPersonality,
                    session_id: this.sessionId
                })
            });
            
            if (!response.ok) throw new Error('Network response was not ok');
            
            const data = await response.json();
            
            // Update session information from response
            if (data.session_id) {
                this.sessionId = data.session_id;
            }
            if (data.conversation_length !== undefined) {
                this.conversationLength = data.conversation_length;
            }
            if (data.context_used !== undefined) {
                this.contextUsed = data.context_used;
            }
            
            this.handleAIResponse(data);
            
        } catch (error) {
            console.error('Error processing input:', error);
            this.addMessage('AI', 'Sorry, I encountered an error processing your request.', 'error');
            this.updateStatus('Error');
        }
    }
    
    handleAIResponse(data) {
        this.lastResponse = data.response;
        this.addMessage('AI', data.response, 'ai', data);
        // Text-to-speech disabled
        // this.speak(data.response);
        
        // Update statistics with realistic values
        const confidence = Math.round((data.confidence || 0.82) * 100);
        this.confidenceScore.textContent = confidence + '%';
        
        // Update conversation count with session information
        const conversationCount = data.conversation_length || (parseInt(this.conversationCount.textContent) + 1);
        this.conversationCount.textContent = conversationCount;
        
        // Show context indicator if context was used
        if (data.context_used && data.has_context) {
            this.showContextIndicator();
        }
        
        // Update professional UI metrics if available
        if (window.professionalUI) {
            const responseTime = parseFloat(data.response_time) || 0.8;
            window.professionalUI.updateMetrics(responseTime, data.confidence || 0.82);
        }
        
        // Show feedback options
        this.showFeedbackOptions();
        
        // Update timers and reminders if needed
        if (data.intent === 'timer' || data.intent === 'reminder') {
            setTimeout(() => this.loadActiveTimersReminders(), 1000);
        }
        
        this.updateStatus('Ready');
    }
    
    showContextIndicator() {
        // Add a visual indicator that context was used
        const lastMessage = this.messagesContainer.lastElementChild;
        if (lastMessage) {
            const contextBadge = document.createElement('span');
            contextBadge.className = 'context-badge';
            contextBadge.innerHTML = '🧠 Context-aware';
            contextBadge.title = 'This response used previous conversation context';
            lastMessage.appendChild(contextBadge);
        }
    }
    
    formatMarkdownLinks(text) {
        // Convert markdown links [text](url) to HTML <a> tags
        return text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, linkText, url) => {
            // Add target="_blank" for external links
            const isExternal = url.startsWith('http://') || url.startsWith('https://');
            const target = isExternal ? ' target="_blank" rel="noopener noreferrer"' : '';
            
            // Add special styling for logo links
            const isLogoLink = linkText.toLowerCase().includes('logo') || linkText.toLowerCase().includes('view');
            const className = isLogoLink ? ' class="logo-link"' : '';
            
            return `<a href="${url}"${target}${className}>${linkText}</a>`;
        });
    }
    
    addMessage(sender, message, type, data = null, autoScroll = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const timestamp = new Date().toLocaleTimeString();
        
        // Check if message contains image URL (both external and local)
        const imageUrlPattern = /(https?:\/\/[^\s]+\.(jpg|jpeg|png|gif|webp|bmp)|\/static\/generated_images\/[^\s]+\.(jpg|jpeg|png|gif|webp|bmp))/gi;
        
        // Check for video URLs
        const videoUrlPattern = /(https?:\/\/[^\s]+\.(mp4|avi|mov|webm|mkv)|\/static\/videos\/[^\s]+\.(mp4|avi|mov|webm|mkv))/gi;
        
        // Check for GIF URLs (animated GIFs)
        const gifUrlPattern = /(https?:\/\/[^\s]+\.gif|\/static\/gifs\/[^\s]+\.gif)/gi;
        
        // Check for full localhost URLs pointing to images
        const fullLocalUrlPattern = /(http:\/\/127\.0\.0\.1:8080\/static\/generated_images\/[^\s]+\.(png|jpg|jpeg|gif|webp))/gi;
        
        // Check for full localhost URLs pointing to videos
        const fullLocalVideoPattern = /(http:\/\/127\.0\.0\.1:8080\/static\/videos\/[^\s]+\.(mp4|avi|mov|webm|mkv))/gi;
        
        // Check for full localhost URLs pointing to GIFs
        const fullLocalGifPattern = /(http:\/\/127\.0\.0\.1:8080\/static\/gifs\/[^\s]+\.gif)/gi;
        
        // Check for markdown image syntax: ![alt](url)
        const markdownImagePattern = /!\[.*?\]\((\/static\/generated_images\/[^)]+\.(png|jpg|jpeg|gif|webp))\)/gi;
        
        // Also check for video and gif paths that might be on their own line
        const localImagePattern = /\/static\/generated_images\/[\w\-\.]+\.(png|jpg|jpeg|gif|webp)/gi;
        const localVideoPattern = /\/static\/videos\/[\w\-\.]+\.(mp4|avi|mov|webm|mkv)/gi;
        const localGifPattern = /\/static\/gifs\/[\w\-\.]+\.gif/gi;
        
        // Even more specific pattern for the exact format we're generating
        const specificImagePattern = /\/static\/generated_images\/[a-f0-9\-]+\.png/gi;
        const specificVideoPattern = /\/static\/videos\/[a-f0-9\-]+\.mp4/gi;
        const specificGifPattern = /\/static\/gifs\/[a-f0-9\-]+\.gif/gi;
        
        let imageUrls = message.match(imageUrlPattern) || [];
        let videoUrls = message.match(videoUrlPattern) || [];
        let gifUrls = message.match(gifUrlPattern) || [];
        
        const fullLocalUrls = message.match(fullLocalUrlPattern) || [];
        const fullLocalVideos = message.match(fullLocalVideoPattern) || [];
        const fullLocalGifs = message.match(fullLocalGifPattern) || [];
        
        const localImages = message.match(localImagePattern) || [];
        const localVideos = message.match(localVideoPattern) || [];
        const localGifs = message.match(localGifPattern) || [];
        
        const specificImages = message.match(specificImagePattern) || [];
        const specificVideos = message.match(specificVideoPattern) || [];
        const specificGifs = message.match(specificGifPattern) || [];
        
        // Extract URLs from markdown image syntax
        let markdownMatch;
        const markdownImages = [];
        while ((markdownMatch = markdownImagePattern.exec(message)) !== null) {
            markdownImages.push(markdownMatch[1]);
        }
        
        // Combine all patterns and remove duplicates
        imageUrls = [...imageUrls, ...fullLocalUrls, ...localImages, ...specificImages, ...markdownImages].filter((url, index, self) => self.indexOf(url) === index);
        videoUrls = [...videoUrls, ...fullLocalVideos, ...localVideos, ...specificVideos].filter((url, index, self) => self.indexOf(url) === index);
        gifUrls = [...gifUrls, ...fullLocalGifs, ...localGifs, ...specificGifs].filter((url, index, self) => self.indexOf(url) === index);
        
        // Combine all media URLs for message cleaning
        const allMediaUrls = [...imageUrls, ...videoUrls, ...gifUrls];
        
        // Remove image URLs from the message text if they will be displayed as images
        let displayMessage = message;
        if (allMediaUrls && allMediaUrls.length > 0) {
            // Remove markdown image syntax
            displayMessage = displayMessage.replace(/!\[.*?\]\([^)]+\)/gi, '');
            
            // Remove plain media URLs
            allMediaUrls.forEach(url => {
                displayMessage = displayMessage.replace(url, '').trim();
            });
            
            // Clean up any extra line breaks
            displayMessage = displayMessage.replace(/\n\s*\n/g, '\n').trim();
        }
        
        // Convert markdown links to HTML (for clickable logo URLs)
        displayMessage = this.formatMarkdownLinks(displayMessage);
        
        // Convert bold markdown to HTML
        displayMessage = displayMessage.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        let messageContent = `
            <div class="message-header">
                <strong>${sender}</strong>
                <span class="timestamp">${timestamp}</span>
            </div>
            <div class="message-content">${displayMessage}</div>
        `;
        
        // Add image display if image URLs are found
        if (imageUrls && imageUrls.length > 0) {
            messageContent += '<div class="image-gallery">';
            imageUrls.forEach(url => {
                messageContent += `
                    <div class="generated-image-container">
                        <img src="${url}" 
                             alt="Generated AI Image" 
                             class="generated-image"
                             onclick="this.classList.toggle('fullscreen')"
                             onerror="this.style.display='none'; this.nextElementSibling.style.display='block';"
                             onload="professionalUI.showToast('🎨 Image loaded successfully!', 'success', 3000);">
                        <div class="image-error" style="display: none;">
                            <p>❌ Failed to load image</p>
                            <a href="${url}" target="_blank">View original link</a>
                        </div>
                        <div class="image-actions">
                            <button onclick="window.open('${url}', '_blank')" class="action-btn">
                                🔗 Open in New Tab
                            </button>
                            <button onclick="navigator.clipboard.writeText('${url}')" class="action-btn">
                                📋 Copy URL
                            </button>
                        </div>
                    </div>
                `;
            });
            messageContent += '</div>';
        }

        // Add video display if video URLs are found
        if (videoUrls && videoUrls.length > 0) {
            messageContent += '<div class="video-gallery">';
            videoUrls.forEach(url => {
                messageContent += `
                    <div class="generated-video-container">
                        <video controls class="generated-video"
                               onloadstart="professionalUI.showToast('🎥 Video loading...', 'info', 2000);"
                               oncanplay="professionalUI.showToast('🎥 Video ready to play!', 'success', 3000);"
                               onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                            <source src="${url}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                        <div class="video-error" style="display: none;">
                            <p>❌ Failed to load video</p>
                            <a href="${url}" target="_blank">View original link</a>
                        </div>
                        <div class="video-actions">
                            <button onclick="window.open('${url}', '_blank')" class="action-btn">
                                🔗 Open in New Tab
                            </button>
                            <button onclick="navigator.clipboard.writeText('${url}')" class="action-btn">
                                📋 Copy URL
                            </button>
                        </div>
                    </div>
                `;
            });
            messageContent += '</div>';
        }

        // Add GIF display if GIF URLs are found
        if (gifUrls && gifUrls.length > 0) {
            messageContent += '<div class="gif-gallery">';
            gifUrls.forEach(url => {
                messageContent += `
                    <div class="generated-gif-container">
                        <img src="${url}" 
                             alt="Generated AI GIF" 
                             class="generated-gif"
                             onclick="this.classList.toggle('fullscreen')"
                             onload="professionalUI.showToast('🎞️ GIF loaded successfully!', 'success', 3000);"
                             onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                        <div class="gif-error" style="display: none;">
                            <p>❌ Failed to load GIF</p>
                            <a href="${url}" target="_blank">View original link</a>
                        </div>
                        <div class="gif-actions">
                            <button onclick="window.open('${url}', '_blank')" class="action-btn">
                                🔗 Open in New Tab
                            </button>
                            <button onclick="navigator.clipboard.writeText('${url}')" class="action-btn">
                                📋 Copy URL
                            </button>
                        </div>
                    </div>
                `;
            });
            messageContent += '</div>';
        }
        
        // Add enhanced metadata for AI responses
        if (data && type === 'ai') {
            const confidence = data.confidence || 0.82;
            const aiSource = data.ai_source || 'fallback';
            const responseTime = data.response_time || '0.8s';
            const intent = data.intent || 'general';
            const contextUsed = data.context_used ? ' | 🧠 Context-aware' : '';
            
            messageContent += `
                <div class="message-metadata">
                    <small>
                        Response Time: ${responseTime} | 
                        Confidence: ${Math.round(confidence * 100)}% |
                        Source: ${aiSource === 'chatgpt' ? 'ChatGPT API' : 'Smart Fallback'} |
                        Intent: ${intent}${contextUsed}
                    </small>
                </div>
            `;
        }
        
        messageDiv.innerHTML = messageContent;
        this.messagesContainer.appendChild(messageDiv);
        
        // Only auto-scroll for new messages, not historical ones
        if (autoScroll) {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }
        
        // Store in conversation history
        this.conversationHistory.push({
            sender, message, timestamp, data
        });
    }
    
    speak(text) {
        // Text-to-speech is disabled
        return;
    }
    
    cleanTextForSpeech(text) {
        // Remove emojis and clean text for better speech synthesis
        const cleanedText = text
            // Remove all emoji characters (Unicode ranges for emojis)
            .replace(/[\u{1F600}-\u{1F64F}]/gu, '') // Emoticons
            .replace(/[\u{1F300}-\u{1F5FF}]/gu, '') // Symbols & Pictographs
            .replace(/[\u{1F680}-\u{1F6FF}]/gu, '') // Transport & Map
            .replace(/[\u{1F1E0}-\u{1F1FF}]/gu, '') // Regional indicators
            .replace(/[\u{2600}-\u{26FF}]/gu, '')   // Miscellaneous Symbols
            .replace(/[\u{2700}-\u{27BF}]/gu, '')   // Dingbats
            .replace(/[\u{1F900}-\u{1F9FF}]/gu, '') // Supplemental Symbols
            .replace(/[\u{1FA70}-\u{1FAFF}]/gu, '') // Extended Pictographs
            // Remove other symbols that sound awkward when spoken
            .replace(/[⭐✨🌟⚡🎯🎉🚀]/g, '')
            .replace(/[📊📈📉📌🔧⚙️🛠️]/g, '')
            .replace(/[❌✅⚠️ℹ️]/g, '')
            // Clean up extra spaces and formatting
            .replace(/\s+/g, ' ')
            .replace(/^\s+|\s+$/g, '')
            // Replace some text patterns that don't speak well
            .replace(/\*([^*]+)\*/g, '$1') // Remove asterisk emphasis
            .replace(/`([^`]+)`/g, '$1')   // Remove backticks
            .replace(/\n+/g, '. ')        // Replace newlines with periods
            .trim();
        
        // Debug log to see the cleaning effect
        if (text !== cleanedText) {
            console.log('Speech cleaning:', {
                original: text,
                cleaned: cleanedText
            });
        }
        
        return cleanedText;
    }
    
    showFeedbackOptions() {
        if (!this.feedbackContainer) return;
        
        this.feedbackContainer.innerHTML = `
            <div class="feedback-prompt">
                <p>How was my response?</p>
                <button class="feedback-btn good" data-feedback="good">👍 Good</button>
                <button class="feedback-btn bad" data-feedback="bad">👎 Bad</button>
                <button class="feedback-btn excellent" data-feedback="excellent">⭐ Excellent</button>
            </div>
        `;
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            this.feedbackContainer.innerHTML = '';
        }, 10000);
    }
    
    async submitFeedback(feedback) {
        try {
            await fetch('/api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_input: this.lastInput,
                    ai_response: this.lastResponse,
                    feedback: feedback
                })
            });
            
            this.feedbackContainer.innerHTML = '<p class="feedback-thanks">Thanks for your feedback! 🙏</p>';
            setTimeout(() => {
                this.feedbackContainer.innerHTML = '';
            }, 3000);
            
        } catch (error) {
            console.error('Error submitting feedback:', error);
        }
    }
    
    async setPersonality(personality) {
        this.currentPersonality = personality;
        try {
            await fetch('/api/personality', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ personality })
            });
            
            this.addMessage('System', `Personality changed to ${personality}`, 'system');
        } catch (error) {
            console.error('Error setting personality:', error);
        }
    }
    
    async loadActiveTimersReminders() {
        try {
            const response = await fetch('/api/timers-reminders');
            if (!response.ok) {
                console.error('Failed to load timers/reminders:', response.status);
                return;
            }
            
            const data = await response.json();
            this.activeTimers = data.timers || [];
            this.activeReminders = data.reminders || [];
            
            console.log('Loaded timers:', this.activeTimers);
            console.log('Loaded reminders:', this.activeReminders);
            
            this.updateTimersDisplay();
            this.updateRemindersDisplay();
            
        } catch (error) {
            console.error('Error loading timers/reminders:', error);
        }
    }
    
    updateTimersDisplay() {
        if (!this.timersContainer) return;
        
        if (this.activeTimers.length === 0) {
            this.timersContainer.innerHTML = '<div class="no-items">No active timers</div>';
            return;
        }
        
        const timersHtml = this.activeTimers.map(timer => {
            const minutes = Math.floor(timer.remaining_seconds / 60);
            const seconds = timer.remaining_seconds % 60;
            const timeDisplay = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            
            return `
                <div class="timer-item" data-timer-id="${timer.id}">
                    <div class="timer-info">
                        <span class="timer-description">${timer.description}</span>
                        <span class="timer-time">${timeDisplay}</span>
                    </div>
                    <button class="cancel-timer-btn" data-timer-id="${timer.id}">Cancel</button>
                </div>
            `;
        }).join('');
        
        this.timersContainer.innerHTML = timersHtml;
    }
    
    updateRemindersDisplay() {
        if (!this.remindersContainer) return;
        
        if (this.activeReminders.length === 0) {
            this.remindersContainer.innerHTML = '<div class="no-items">No active reminders</div>';
            return;
        }
        
        const remindersHtml = this.activeReminders.map(reminder => {
            const minutesUntil = reminder.minutes_until;
            const timeText = minutesUntil > 0 ? `in ${minutesUntil} minutes` : 'overdue';
            
            return `
                <div class="reminder-item" data-reminder-id="${reminder.id}">
                    <div class="reminder-info">
                        <span class="reminder-text">${reminder.text}</span>
                        <span class="reminder-time">${timeText}</span>
                    </div>
                    <button class="cancel-reminder-btn" data-reminder-id="${reminder.id}">Cancel</button>
                </div>
            `;
        }).join('');
        
        this.remindersContainer.innerHTML = remindersHtml;
    }
    
    async cancelTimer(timerId) {
        try {
            const response = await fetch(`/api/cancel-timer/${timerId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                console.log(`Timer ${timerId} cancelled`);
                // Reload timers to update display
                await this.loadActiveTimersReminders();
            } else {
                console.error('Failed to cancel timer:', response.status);
            }
        } catch (error) {
            console.error('Error cancelling timer:', error);
        }
    }
    
    async cancelReminder(reminderId) {
        try {
            const response = await fetch(`/api/cancel-reminder/${reminderId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                console.log(`Reminder ${reminderId} cancelled`);
                // Reload reminders to update display
                await this.loadActiveTimersReminders();
            } else {
                console.error('Failed to cancel reminder:', response.status);
            }
        } catch (error) {
            console.error('Error cancelling reminder:', error);
        }
    }
    
    updateTimersDisplay() {
        if (!this.timersContainer) return;
        
        if (this.activeTimers.length === 0) {
            this.timersContainer.innerHTML = '<p>No active timers</p>';
            return;
        }
        
        let timersHTML = '<h4>Active Timers</h4>';
        this.activeTimers.forEach(timer => {
            const remainingTime = this.formatTime(timer.remaining_seconds);
            timersHTML += `
                <div class="timer-item">
                    <span>${timer.duration} (${remainingTime} left)</span>
                    <button class="cancel-timer-btn" data-timer-id="${timer.id}">Cancel</button>
                </div>
            `;
        });
        
        this.timersContainer.innerHTML = timersHTML;
    }
    
    updateRemindersDisplay() {
        if (!this.remindersContainer) return;
        
        if (this.activeReminders.length === 0) {
            this.remindersContainer.innerHTML = '<p>No active reminders</p>';
            return;
        }
        
        let remindersHTML = '<h4>Active Reminders</h4>';
        this.activeReminders.forEach(reminder => {
            remindersHTML += `
                <div class="reminder-item">
                    <span>${reminder.text} (${reminder.time})</span>
                    <button class="cancel-reminder-btn" data-reminder-id="${reminder.id}">Cancel</button>
                </div>
            `;
        });
        
        this.remindersContainer.innerHTML = remindersHTML;
    }
    
    async cancelTimer(timerId) {
        try {
            const response = await fetch(`/api/cancel-timer/${timerId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.ok) {
                this.addMessage('System', 'Timer cancelled successfully', 'system');
                this.loadActiveTimersReminders();
            }
        } catch (error) {
            console.error('Error cancelling timer:', error);
        }
    }
    
    async cancelReminder(reminderId) {
        try {
            const response = await fetch(`/api/cancel-reminder/${reminderId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.ok) {
                this.addMessage('System', 'Reminder cancelled successfully', 'system');
                this.loadActiveTimersReminders();
            }
        } catch (error) {
            console.error('Error cancelling reminder:', error);
        }
    }
    
    formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        
        if (hours > 0) {
            return `${hours}h ${minutes}m ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }
    
    startPeriodicUpdates() {
        // Update timers and reminders every 5 seconds
        setInterval(() => {
            if (this.activeTimers.length > 0 || this.activeReminders.length > 0) {
                this.loadActiveTimersReminders();
            }
        }, 5000);
        
        // Update timer countdowns every second
        setInterval(() => {
            this.activeTimers = this.activeTimers.map(timer => {
                timer.remaining_seconds = Math.max(0, timer.remaining_seconds - 1);
                return timer;
            }).filter(timer => timer.remaining_seconds > 0);
            
            this.updateTimersDisplay();
        }, 1000);
    }
    
    updateStatus(status) {
        if (this.statusIndicator) {
            this.statusIndicator.textContent = status;
        }
        console.log('Status:', status);
    }
    
    // New: Session management functions
    async initializeSession() {
        try {
            // Check if we have an existing active session
            const response = await fetch('/api/conversation/sessions');
            const data = await response.json();
            
            if (data.sessions && data.sessions.length > 0) {
                const activeSession = data.sessions.find(s => s.is_active);
                if (activeSession) {
                    this.sessionId = activeSession.id;
                    this.conversationLength = activeSession.message_count;
                    this.currentPersonality = activeSession.personality;
                    
                    // Update personality selector
                    if (this.personalitySelect) {
                        this.personalitySelect.value = this.currentPersonality;
                    }
                    
                    // Load conversation history
                    this.loadConversationHistory();
                    return;
                }
            }
            
            // Create new session if no active session exists
            this.createNewSession();
            
        } catch (error) {
            console.error('Error initializing session:', error);
            // Continue without session if there's an error
        }
    }
    
    async createNewSession() {
        try {
            const response = await fetch('/api/conversation/new-session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    personality: this.currentPersonality
                })
            });
            
            const data = await response.json();
            this.sessionId = data.session_id;
            this.conversationLength = 0;
            this.contextUsed = false;
            
            console.log('New conversation session created:', this.sessionId);
            
        } catch (error) {
            console.error('Error creating new session:', error);
        }
    }
    
    async loadConversationHistory() {
        try {
            if (!this.sessionId) return;
            
            const response = await fetch(`/api/conversation/history?session_id=${this.sessionId}&limit=10`);
            const data = await response.json();
            
            // Clear current messages and load history
            if (this.messagesContainer) {
                this.messagesContainer.innerHTML = '';
            }
            
            // Add historical messages
            for (const msg of data.history) {
                this.addMessage('You', msg.user_input, 'user', null, false);
                this.addMessage('AI', msg.ai_response, 'ai', {
                    confidence: msg.confidence,
                    intent: msg.intent,
                    timestamp: msg.timestamp
                }, false);
            }
            
            // Update conversation count
            this.conversationLength = data.message_count;
            if (this.conversationCount) {
                this.conversationCount.textContent = this.conversationLength;
            }
            
        } catch (error) {
            console.error('Error loading conversation history:', error);
        }
    }
    
    async clearConversation() {
        try {
            await this.createNewSession();
            
            // Clear messages
            if (this.messagesContainer) {
                this.messagesContainer.innerHTML = '';
            }
            
            // Reset counters
            this.conversationLength = 0;
            if (this.conversationCount) {
                this.conversationCount.textContent = '0';
            }
            
            // Add welcome message
            this.addMessage('AI', 'Hello! I\'m Horizon, your AI assistant. How can I help you today?', 'ai');
            
        } catch (error) {
            console.error('Error clearing conversation:', error);
        }
    }
    
    // Test microphone access
    async testMicrophone() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.addMessage('System', '✅ Microphone test successful!', 'system');
            stream.getTracks().forEach(track => track.stop()); // Stop the stream
            return true;
        } catch (error) {
            this.addMessage('System', `❌ Microphone test failed: ${error.message}`, 'error');
            return false;
        }
    }
    
    // Simple Horizon explanation help - now appears in chat
    showHorizonHelp() {
        // Add help information as chat messages
        this.addMessage('You', 'Help - What is Horizon?', 'user');
        
        this.addMessage('Horizon AI', `Horizon is a next-generation AI chatbot that goes beyond conversation—combining ChatGPT-style intelligence with fast-action features like quick commands, instant timers, and smart reminders. Designed for both productivity and natural interaction, Horizon delivers lightning-fast responses while helping users stay organized and in control.`, 'ai');
    }
}

// Initialize the enhanced AI assistant when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.aiAssistant = new EnhancedAIVoiceAssistant();
    
    // Add help button functionality
    const helpBtn = document.getElementById('helpBtn');
    if (helpBtn) {
        helpBtn.addEventListener('click', () => {
            window.aiAssistant.showHorizonHelp();
        });
    }
});
