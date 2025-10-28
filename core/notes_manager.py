"""
Advanced Notes Management System
Sophisticated note organization, AI-powered categorization, and intelligent content management.
"""

import json
import os
import sqlite3
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import uuid
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import threading
import time

@dataclass
class NoteMetrics:
    """Data class for note analytics and metrics"""
    word_count: int
    char_count: int
    paragraph_count: int
    code_blocks: int
    urls: int
    reading_time_minutes: float
    complexity_score: float
    sentiment_score: Optional[float] = None

class AdvancedNotesManager:
    """
    Advanced Notes Management System with AI-powered features.
    Provides intelligent categorization, content analysis, and sophisticated organization.
    """
    
    def __init__(self, data_dir: str = "data/notes"):
        self.data_dir = data_dir
        self.db_path = f"{data_dir}/notes.db"
        self.index_path = f"{data_dir}/search_index.json"
        self.config_path = f"{data_dir}/notes_config.json"
        
        # Advanced configuration
        self.config = {
            'ai_categorization': True,
            'smart_tagging': True,
            'content_analysis': True,
            'auto_backup': True,
            'backup_interval_hours': 24,
            'max_notes_per_category': 1000,
            'search_index_enabled': True,
            'full_text_search': True,
            'duplicate_detection': True,
            'content_similarity_threshold': 0.85,
            'auto_archive_days': 365,
            'compression_enabled': False,
            'encryption_enabled': False
        }
        
        # AI categorization patterns (enhanced)
        self.ai_patterns = {
            'programming': {
                'keywords': ['code', 'function', 'class', 'algorithm', 'programming', 'software'],
                'patterns': [r'```[a-z]*', r'def\s+\w+', r'class\s+\w+', r'import\s+\w+', r'function\s*\('],
                'weight': 1.0
            },
            'documentation': {
                'keywords': ['docs', 'documentation', 'guide', 'manual', 'instructions', 'tutorial'],
                'patterns': [r'step\s+\d+', r'how\s+to', r'tutorial:', r'guide:', r'instructions:'],
                'weight': 0.9
            },
            'research': {
                'keywords': ['research', 'study', 'analysis', 'findings', 'data', 'statistics'],
                'patterns': [r'research\s+shows', r'study\s+reveals', r'data\s+indicates', r'analysis\s+of'],
                'weight': 0.8
            },
            'meeting_notes': {
                'keywords': ['meeting', 'agenda', 'action items', 'discussion', 'decisions'],
                'patterns': [r'meeting\s+notes?', r'action\s+items?', r'discussed:', r'decided:'],
                'weight': 0.7
            },
            'ideas': {
                'keywords': ['idea', 'brainstorm', 'concept', 'innovation', 'creative', 'inspiration'],
                'patterns': [r'idea:', r'brainstorm', r'what\s+if', r'imagine\s+if'],
                'weight': 0.6
            },
            'tasks': {
                'keywords': ['todo', 'task', 'action', 'reminder', 'deadline', 'priority'],
                'patterns': [r'todo:', r'task:', r'deadline:', r'priority:', r'reminder:'],
                'weight': 0.5
            }
        }
        
        # Smart tagging patterns
        self.smart_tags = {
            'technology': {
                'ai': [r'artificial\s+intelligence', r'machine\s+learning', r'neural\s+network', r'deep\s+learning'],
                'web': [r'html', r'css', r'javascript', r'react', r'vue', r'angular', r'nodejs'],
                'mobile': [r'ios', r'android', r'flutter', r'react\s+native', r'swift', r'kotlin'],
                'database': [r'sql', r'mongodb', r'postgresql', r'mysql', r'database', r'query'],
                'cloud': [r'aws', r'azure', r'gcp', r'docker', r'kubernetes', r'serverless'],
                'blockchain': [r'blockchain', r'cryptocurrency', r'bitcoin', r'ethereum', r'smart\s+contract']
            },
            'business': {
                'strategy': [r'strategy', r'planning', r'roadmap', r'objectives', r'goals'],
                'finance': [r'budget', r'revenue', r'profit', r'investment', r'financial'],
                'marketing': [r'marketing', r'advertising', r'campaign', r'branding', r'seo'],
                'operations': [r'operations', r'process', r'workflow', r'efficiency', r'optimization']
            },
            'academic': {
                'science': [r'science', r'research', r'experiment', r'hypothesis', r'theory'],
                'mathematics': [r'math', r'mathematics', r'equation', r'formula', r'calculation'],
                'literature': [r'literature', r'poetry', r'novel', r'writing', r'author'],
                'history': [r'history', r'historical', r'ancient', r'civilization', r'timeline']
            }
        }
        
        # Content quality indicators
        self.quality_indicators = {
            'high_quality': [
                r'detailed\s+explanation',
                r'step\s+by\s+step',
                r'comprehensive\s+guide',
                r'best\s+practices',
                r'example.*code',
                r'case\s+study'
            ],
            'reference': [
                r'documentation',
                r'official\s+docs',
                r'api\s+reference',
                r'specification',
                r'standard'
            ],
            'temporary': [
                r'quick\s+note',
                r'reminder',
                r'temp',
                r'draft',
                r'scratch'
            ]
        }
        
        # Initialize system
        self.ensure_directories()
        self.init_database()
        self.load_config()
        self.search_index = self.load_search_index()
        
        # Background tasks
        self._stop_background = False
        self._background_thread = threading.Thread(target=self._background_tasks, daemon=True)
        self._background_thread.start()
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.data_dir,
            f"{self.data_dir}/backups",
            f"{self.data_dir}/exports",
            f"{self.data_dir}/cache",
            f"{self.data_dir}/attachments"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def init_database(self):
        """Initialize SQLite database for notes storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Notes table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS notes (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        content_hash TEXT UNIQUE,
                        category TEXT,
                        tags TEXT,  -- JSON array
                        source_data TEXT,  -- JSON object
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        word_count INTEGER DEFAULT 0,
                        char_count INTEGER DEFAULT 0,
                        reading_time REAL DEFAULT 0,
                        complexity_score REAL DEFAULT 0,
                        quality_score REAL DEFAULT 0,
                        starred BOOLEAN DEFAULT FALSE,
                        archived BOOLEAN DEFAULT FALSE,
                        pinned BOOLEAN DEFAULT FALSE,
                        version INTEGER DEFAULT 1
                    )
                ''')
                
                # Note history table for versioning
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS note_history (
                        id TEXT PRIMARY KEY,
                        note_id TEXT,
                        version INTEGER,
                        title TEXT,
                        content TEXT,
                        updated_at TEXT,
                        change_summary TEXT,
                        FOREIGN KEY (note_id) REFERENCES notes (id)
                    )
                ''')
                
                # Categories table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS categories (
                        name TEXT PRIMARY KEY,
                        description TEXT,
                        color TEXT,
                        icon TEXT,
                        note_count INTEGER DEFAULT 0,
                        created_at TEXT
                    )
                ''')
                
                # Tags table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tags (
                        name TEXT PRIMARY KEY,
                        description TEXT,
                        color TEXT,
                        usage_count INTEGER DEFAULT 0,
                        created_at TEXT
                    )
                ''')
                
                # Relationships table for advanced queries
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS note_relationships (
                        id TEXT PRIMARY KEY,
                        note_id_1 TEXT,
                        note_id_2 TEXT,
                        relationship_type TEXT,  -- similar, related, referenced
                        strength REAL DEFAULT 0.5,
                        created_at TEXT,
                        FOREIGN KEY (note_id_1) REFERENCES notes (id),
                        FOREIGN KEY (note_id_2) REFERENCES notes (id)
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_notes_created_at ON notes (created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_notes_category ON notes (category)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_notes_starred ON notes (starred)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_notes_archived ON notes (archived)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_notes_content_hash ON notes (content_hash)')
                
                conn.commit()
                print("🗄️ Notes database initialized successfully")
                
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    stored_config = json.load(f)
                    self.config.update(stored_config)
        except Exception as e:
            print(f"Config loading error: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Config saving error: {e}")
    
    def load_search_index(self) -> Dict[str, Any]:
        """Load search index from file"""
        try:
            if os.path.exists(self.index_path):
                with open(self.index_path, 'r') as f:
                    return json.load(f)
            return {'terms': {}, 'last_updated': datetime.now().isoformat()}
        except Exception as e:
            print(f"Search index loading error: {e}")
            return {'terms': {}, 'last_updated': datetime.now().isoformat()}
    
    def save_search_index(self):
        """Save search index to file"""
        try:
            self.search_index['last_updated'] = datetime.now().isoformat()
            with open(self.index_path, 'w') as f:
                json.dump(self.search_index, f, indent=2)
        except Exception as e:
            print(f"Search index saving error: {e}")
    
    def analyze_content(self, content: str) -> NoteMetrics:
        """Analyze content and extract metrics"""
        try:
            # Basic metrics
            words = content.split()
            word_count = len(words)
            char_count = len(content)
            paragraphs = content.split('\n\n')
            paragraph_count = len([p for p in paragraphs if p.strip()])
            
            # Count code blocks
            code_blocks = len(re.findall(r'```.*?```', content, re.DOTALL))
            
            # Count URLs
            urls = len(re.findall(r'https?://[^\s]+', content))
            
            # Calculate reading time (average 200 words per minute)
            reading_time = word_count / 200.0
            
            # Calculate complexity score based on various factors
            complexity_factors = [
                len(re.findall(r'[.!?]+', content)) / max(word_count, 1),  # Sentence complexity
                code_blocks * 0.2,  # Code complexity
                len(re.findall(r'\b[A-Z]{2,}\b', content)) * 0.1,  # Acronyms
                urls * 0.1,  # External references
                paragraph_count / max(word_count / 50, 1)  # Structure complexity
            ]
            
            complexity_score = min(sum(complexity_factors), 1.0)
            
            return NoteMetrics(
                word_count=word_count,
                char_count=char_count,
                paragraph_count=paragraph_count,
                code_blocks=code_blocks,
                urls=urls,
                reading_time_minutes=reading_time,
                complexity_score=complexity_score
            )
            
        except Exception as e:
            print(f"Content analysis error: {e}")
            return NoteMetrics(0, 0, 0, 0, 0, 0.0, 0.0)
    
    def smart_categorize(self, content: str, title: str = "") -> str:
        """AI-powered content categorization"""
        try:
            text = f"{title} {content}".lower()
            category_scores = {}
            
            for category, patterns in self.ai_patterns.items():
                score = 0.0
                
                # Keyword matching
                for keyword in patterns['keywords']:
                    if keyword in text:
                        score += 1.0
                
                # Pattern matching
                for pattern in patterns['patterns']:
                    matches = len(re.findall(pattern, text))
                    score += matches * 0.5
                
                # Apply weight
                category_scores[category] = score * patterns['weight']
            
            # Return category with highest score
            if category_scores:
                best_category = max(category_scores, key=category_scores.get)
                if category_scores[best_category] > 0.5:
                    return best_category
            
            return 'general'
            
        except Exception as e:
            print(f"Smart categorization error: {e}")
            return 'general'
    
    def generate_smart_tags(self, content: str, title: str = "", max_tags: int = 8) -> List[str]:
        """Generate intelligent tags based on content analysis"""
        try:
            text = f"{title} {content}".lower()
            detected_tags = []
            
            # Technology tags
            for category, tag_patterns in self.smart_tags.items():
                for tag, patterns in tag_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, text):
                            detected_tags.append(tag)
                            break
            
            # Quality-based tags
            for quality, patterns in self.quality_indicators.items():
                for pattern in patterns:
                    if re.search(pattern, text):
                        detected_tags.append(quality)
                        break
            
            # Content-type tags
            if re.search(r'```', content):
                detected_tags.append('code')
            if re.search(r'https?://', content):
                detected_tags.append('reference')
            if len(content.split()) > 500:
                detected_tags.append('long-form')
            if content.count('?') > 3:
                detected_tags.append('qa')
            
            # Remove duplicates and limit
            unique_tags = list(dict.fromkeys(detected_tags))
            return unique_tags[:max_tags]
            
        except Exception as e:
            print(f"Smart tag generation error: {e}")
            return []
    
    def calculate_quality_score(self, content: str, metrics: NoteMetrics) -> float:
        """Calculate content quality score"""
        try:
            quality_score = 0.0
            
            # Length factor (optimal range: 100-1000 words)
            if 100 <= metrics.word_count <= 1000:
                quality_score += 0.3
            elif metrics.word_count > 50:
                quality_score += 0.1
            
            # Structure factor
            if metrics.paragraph_count > 1:
                quality_score += 0.2
            
            # Code/examples factor
            if metrics.code_blocks > 0:
                quality_score += 0.2
            
            # Reference factor
            if metrics.urls > 0:
                quality_score += 0.1
            
            # Content indicators
            for quality, patterns in self.quality_indicators.items():
                for pattern in patterns:
                    if re.search(pattern, content.lower()):
                        if quality == 'high_quality':
                            quality_score += 0.3
                        elif quality == 'reference':
                            quality_score += 0.2
                        elif quality == 'temporary':
                            quality_score -= 0.1
                        break
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            print(f"Quality score calculation error: {e}")
            return 0.5
    
    def detect_duplicates(self, content: str, content_hash: str) -> List[str]:
        """Detect potential duplicate notes"""
        try:
            duplicates = []
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Exact hash match
                cursor.execute('SELECT id FROM notes WHERE content_hash = ? AND id != ?', 
                             (content_hash, ''))
                exact_matches = [row[0] for row in cursor.fetchall()]
                duplicates.extend(exact_matches)
                
                # Content similarity check (for near-duplicates)
                if self.config.get('duplicate_detection', True):
                    cursor.execute('SELECT id, content FROM notes WHERE archived = FALSE')
                    existing_notes = cursor.fetchall()
                    
                    for note_id, existing_content in existing_notes:
                        similarity = self.calculate_similarity(content, existing_content)
                        if similarity > self.config.get('content_similarity_threshold', 0.85):
                            if note_id not in duplicates:
                                duplicates.append(note_id)
            
            return duplicates
            
        except Exception as e:
            print(f"Duplicate detection error: {e}")
            return []
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate content similarity between two texts"""
        try:
            # Simple word-based similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return 0.0
    
    def update_search_index(self, note_id: str, title: str, content: str, tags: List[str]):
        """Update search index with note content"""
        try:
            if not self.config.get('search_index_enabled', True):
                return
            
            # Tokenize content
            text = f"{title} {content} {' '.join(tags)}"
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Update term frequencies
            if 'terms' not in self.search_index:
                self.search_index['terms'] = {}
            
            for word in words:
                if len(word) > 2:  # Ignore very short words
                    if word not in self.search_index['terms']:
                        self.search_index['terms'][word] = {}
                    
                    if note_id not in self.search_index['terms'][word]:
                        self.search_index['terms'][word][note_id] = 0
                    
                    self.search_index['terms'][word][note_id] += 1
        
        except Exception as e:
            print(f"Search index update error: {e}")
    
    def create_advanced_note(self, content: str, title: str = None, 
                           category: str = None, tags: List[str] = None,
                           source_data: Dict[str, Any] = None) -> Optional[str]:
        """Create a note with advanced AI analysis"""
        try:
            # Generate note ID
            note_id = str(uuid.uuid4())
            
            # Analyze content
            metrics = self.analyze_content(content)
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Check for duplicates
            duplicates = self.detect_duplicates(content, content_hash)
            if duplicates and self.config.get('duplicate_detection', True):
                print(f"⚠️ Potential duplicates found: {duplicates}")
            
            # Auto-generate title if not provided
            if not title:
                title = self.generate_smart_title(content)
            
            # Smart categorization
            if not category and self.config.get('ai_categorization', True):
                category = self.smart_categorize(content, title)
            
            # Smart tagging
            if not tags and self.config.get('smart_tagging', True):
                tags = self.generate_smart_tags(content, title)
            
            # Calculate quality score
            quality_score = self.calculate_quality_score(content, metrics)
            
            # Prepare data
            created_at = datetime.now().isoformat()
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO notes (
                        id, title, content, content_hash, category, tags, source_data,
                        created_at, updated_at, word_count, char_count, reading_time,
                        complexity_score, quality_score, starred, archived, pinned, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    note_id, title, content, content_hash, category or 'general',
                    json.dumps(tags or []), json.dumps(source_data or {}),
                    created_at, created_at, metrics.word_count, metrics.char_count,
                    metrics.reading_time_minutes, metrics.complexity_score, quality_score,
                    False, False, False, 1
                ))
                
                # Update category count
                cursor.execute('''
                    INSERT OR REPLACE INTO categories (name, note_count, created_at)
                    VALUES (?, COALESCE((SELECT note_count FROM categories WHERE name = ?), 0) + 1, ?)
                ''', (category or 'general', category or 'general', created_at))
                
                # Update tag counts
                for tag in tags or []:
                    cursor.execute('''
                        INSERT OR REPLACE INTO tags (name, usage_count, created_at)
                        VALUES (?, COALESCE((SELECT usage_count FROM tags WHERE name = ?), 0) + 1, ?)
                    ''', (tag, tag, created_at))
                
                conn.commit()
            
            # Update search index
            self.update_search_index(note_id, title, content, tags or [])
            
            print(f"📝 Advanced note created: {title} (Quality: {quality_score:.2f})")
            return note_id
            
        except Exception as e:
            print(f"Advanced note creation error: {e}")
            return None
    
    def generate_smart_title(self, content: str, max_length: int = 60) -> str:
        """Generate intelligent title from content"""
        try:
            # Try to extract from first meaningful sentence
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 10 and len(clean_sentence) <= max_length:
                    # Check if it looks like a title (not just code or random text)
                    if not re.match(r'^[A-Z_][A-Z0-9_]*$', clean_sentence):  # Not constant
                        if not clean_sentence.startswith('```'):  # Not code block
                            return clean_sentence
            
            # Fallback: use first few words
            words = content.split()[:8]
            title = ' '.join(words)
            
            if len(title) > max_length:
                title = title[:max_length-3] + '...'
            
            return title or f"Note {datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        except Exception as e:
            print(f"Smart title generation error: {e}")
            return f"Note {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _background_tasks(self):
        """Background maintenance tasks"""
        try:
            while not self._stop_background:
                try:
                    # Auto-backup
                    if self.config.get('auto_backup', True):
                        self.create_backup()
                    
                    # Save search index
                    self.save_search_index()
                    
                    # Auto-archive old notes
                    if self.config.get('auto_archive_days', 0) > 0:
                        self.auto_archive_old_notes()
                    
                    # Sleep for the backup interval
                    sleep_hours = self.config.get('backup_interval_hours', 24)
                    time.sleep(sleep_hours * 3600)
                    
                except Exception as e:
                    print(f"Background task error: {e}")
                    time.sleep(3600)  # Sleep 1 hour on error
                    
        except Exception as e:
            print(f"Background thread error: {e}")
    
    def create_backup(self):
        """Create backup of notes database"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{self.data_dir}/backups/notes_backup_{timestamp}.db"
            
            # Copy database file
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            print(f"🔒 Backup created: {backup_path}")
            
        except Exception as e:
            print(f"Backup creation error: {e}")
    
    def auto_archive_old_notes(self):
        """Automatically archive old notes"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config['auto_archive_days'])
            cutoff_iso = cutoff_date.isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE notes 
                    SET archived = TRUE 
                    WHERE created_at < ? AND archived = FALSE AND starred = FALSE
                ''', (cutoff_iso,))
                
                archived_count = cursor.rowcount
                conn.commit()
                
                if archived_count > 0:
                    print(f"📦 Auto-archived {archived_count} old notes")
                    
        except Exception as e:
            print(f"Auto-archive error: {e}")
    
    def get_advanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics and analytics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Basic counts
                cursor.execute('SELECT COUNT(*) FROM notes WHERE archived = FALSE')
                total_notes = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM notes WHERE starred = TRUE')
                starred_notes = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM notes WHERE archived = TRUE')
                archived_notes = cursor.fetchone()[0]
                
                # Word count statistics
                cursor.execute('SELECT SUM(word_count), AVG(word_count), MAX(word_count) FROM notes WHERE archived = FALSE')
                word_stats = cursor.fetchone()
                total_words, avg_words, max_words = word_stats if word_stats[0] else (0, 0, 0)
                
                # Quality statistics
                cursor.execute('SELECT AVG(quality_score), MAX(quality_score) FROM notes WHERE archived = FALSE')
                quality_stats = cursor.fetchone()
                avg_quality, max_quality = quality_stats if quality_stats[0] else (0, 0)
                
                # Category distribution
                cursor.execute('SELECT category, COUNT(*) FROM notes WHERE archived = FALSE GROUP BY category ORDER BY COUNT(*) DESC')
                categories = cursor.fetchall()
                
                # Tag distribution
                cursor.execute('SELECT name, usage_count FROM tags ORDER BY usage_count DESC LIMIT 10')
                top_tags = cursor.fetchall()
                
                # Recent activity (last 7 days)
                week_ago = (datetime.now() - timedelta(days=7)).isoformat()
                cursor.execute('SELECT COUNT(*) FROM notes WHERE created_at > ?', (week_ago,))
                recent_notes = cursor.fetchone()[0]
                
                # Reading time statistics
                cursor.execute('SELECT SUM(reading_time), AVG(reading_time) FROM notes WHERE archived = FALSE')
                reading_stats = cursor.fetchone()
                total_reading_time, avg_reading_time = reading_stats if reading_stats[0] else (0, 0)
                
                return {
                    'notes': {
                        'total': total_notes,
                        'starred': starred_notes,
                        'archived': archived_notes,
                        'recent_week': recent_notes
                    },
                    'content': {
                        'total_words': int(total_words or 0),
                        'average_words': round(float(avg_words or 0), 1),
                        'longest_note_words': int(max_words or 0),
                        'total_reading_time_hours': round(float(total_reading_time or 0) / 60, 1),
                        'average_reading_time_minutes': round(float(avg_reading_time or 0), 1)
                    },
                    'quality': {
                        'average_score': round(float(avg_quality or 0), 2),
                        'highest_score': round(float(max_quality or 0), 2)
                    },
                    'categories': [{'name': cat, 'count': count} for cat, count in categories],
                    'top_tags': [{'name': tag, 'count': count} for tag, count in top_tags],
                    'system': {
                        'database_size_mb': round(os.path.getsize(self.db_path) / 1024 / 1024, 2),
                        'search_index_terms': len(self.search_index.get('terms', {})),
                        'last_backup': self.get_last_backup_time()
                    }
                }
                
        except Exception as e:
            print(f"Statistics generation error: {e}")
            return {'error': str(e)}
    
    def get_last_backup_time(self) -> Optional[str]:
        """Get timestamp of last backup"""
        try:
            backup_dir = f"{self.data_dir}/backups"
            if not os.path.exists(backup_dir):
                return None
            
            backups = [f for f in os.listdir(backup_dir) if f.startswith('notes_backup_')]
            if not backups:
                return None
            
            latest_backup = max(backups)
            backup_path = os.path.join(backup_dir, latest_backup)
            backup_time = datetime.fromtimestamp(os.path.getmtime(backup_path))
            return backup_time.isoformat()
            
        except Exception as e:
            print(f"Last backup time error: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self._stop_background = True
            if hasattr(self, '_background_thread'):
                self._background_thread.join(timeout=5)
            self.save_search_index()
            self.save_config()
        except Exception as e:
            print(f"Cleanup error: {e}")

# Global instance
advanced_notes_manager = None

def get_advanced_notes_manager() -> AdvancedNotesManager:
    """Get or create the advanced notes manager instance"""
    global advanced_notes_manager
    if advanced_notes_manager is None:
        advanced_notes_manager = AdvancedNotesManager()
    return advanced_notes_manager