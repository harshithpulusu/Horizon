"""
Notes Management API Routes
Safe, modular implementation that captures AI responses without interfering with chat functionality.
"""

from flask import Blueprint, request, jsonify
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid
import hashlib
import re

# Create blueprint for notes management (separate from existing routes)
notes_bp = Blueprint('notes', __name__, url_prefix='/api/notes')

class NotesManager:
    """
    AI Response Notes Manager that works alongside existing chat system.
    Completely independent of existing chat/message functionality.
    """
    
    def __init__(self):
        self.notes_dir = "data/notes"
        self.metadata_file = "data/notes_metadata.json"
        self.categories_file = "data/notes_categories.json"
        self.tags_file = "data/notes_tags.json"
        
        # Notes configuration
        self.config = {
            'auto_save_enabled': False,
            'auto_categorize': True,
            'auto_tag': True,
            'max_notes': 10000,
            'export_formats': ['json', 'markdown', 'txt', 'pdf'],
            'search_index_enabled': True,
            'backup_enabled': True,
            'compression_enabled': False
        }
        
        # Auto-categorization rules
        self.category_patterns = {
            'code': [r'```', r'function\s+\w+', r'class\s+\w+', r'import\s+\w+', r'def\s+\w+'],
            'tutorial': [r'step\s+\d+', r'how\s+to', r'tutorial', r'guide', r'instructions'],
            'explanation': [r'explanation', r'because', r'reason', r'why', r'what\s+is'],
            'analysis': [r'analysis', r'data', r'statistics', r'metrics', r'report'],
            'creative': [r'story', r'poem', r'creative', r'imagine', r'generate.*image'],
            'technical': [r'technical', r'specification', r'architecture', r'implementation'],
            'research': [r'research', r'study', r'findings', r'investigation', r'survey']
        }
        
        # Auto-tagging patterns
        self.tag_patterns = {
            'python': [r'python', r'\.py', r'pip\s+install', r'import\s+\w+'],
            'javascript': [r'javascript', r'\.js', r'npm\s+install', r'function\s*\('],
            'ai': [r'artificial\s+intelligence', r'machine\s+learning', r'neural\s+network'],
            'web': [r'html', r'css', r'website', r'web\s+development', r'http'],
            'database': [r'sql', r'database', r'query', r'table', r'mongodb'],
            'api': [r'api', r'endpoint', r'rest', r'json', r'http\s+request'],
            'docker': [r'docker', r'container', r'dockerfile', r'image'],
            'important': [r'important', r'critical', r'urgent', r'note:', r'remember'],
            'todo': [r'todo', r'task', r'action\s+item', r'follow\s+up']
        }
        
        # Ensure directories exist
        self.ensure_directories()
        
        # Load metadata
        self.load_metadata()
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.notes_dir,
            f"{self.notes_dir}/exported",
            f"{self.notes_dir}/backups",
            "data"
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
    
    def load_metadata(self):
        """Load notes metadata from file"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {
                    'notes': {},
                    'categories': {},
                    'tags': {},
                    'total_notes': 0,
                    'last_updated': datetime.now().isoformat()
                }
        except Exception as e:
            print(f"Metadata loading error: {e}")
            self.metadata = {'notes': {}, 'categories': {}, 'tags': {}, 'total_notes': 0}
    
    def save_metadata(self):
        """Save notes metadata to file"""
        try:
            self.metadata['last_updated'] = datetime.now().isoformat()
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Metadata saving error: {e}")
    
    def create_note(self, content: str, title: str = None, 
                   category: str = None, tags: List[str] = None,
                   source_data: Dict[str, Any] = None) -> str:
        """
        Create a new note from AI response or user input.
        """
        try:
            # Generate note ID
            note_id = str(uuid.uuid4())
            
            # Auto-generate title if not provided
            if not title:
                title = self.generate_title(content)
            
            # Auto-categorize if enabled
            if not category and self.config['auto_categorize']:
                category = self.auto_categorize(content)
            
            # Auto-tag if enabled
            if not tags and self.config['auto_tag']:
                tags = self.auto_tag(content)
            
            # Create note object
            note = {
                'id': note_id,
                'title': title,
                'content': content,
                'category': category or 'general',
                'tags': tags or [],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'source': source_data or {},
                'word_count': len(content.split()),
                'char_count': len(content),
                'content_hash': hashlib.md5(content.encode()).hexdigest(),
                'starred': False,
                'archived': False
            }
            
            # Save note to file
            note_file = f"{self.notes_dir}/{note_id}.json"
            with open(note_file, 'w') as f:
                json.dump(note, f, indent=2)
            
            # Update metadata
            self.metadata['notes'][note_id] = {
                'title': title,
                'category': note['category'],
                'tags': note['tags'],
                'created_at': note['created_at'],
                'word_count': note['word_count'],
                'starred': note['starred'],
                'archived': note['archived']
            }
            
            # Update category count
            if note['category'] not in self.metadata['categories']:
                self.metadata['categories'][note['category']] = 0
            self.metadata['categories'][note['category']] += 1
            
            # Update tag counts
            for tag in note['tags']:
                if tag not in self.metadata['tags']:
                    self.metadata['tags'][tag] = 0
                self.metadata['tags'][tag] += 1
            
            self.metadata['total_notes'] += 1
            self.save_metadata()
            
            print(f"📝 Note created: {title} ({note_id})")
            return note_id
            
        except Exception as e:
            print(f"Note creation error: {e}")
            return None
    
    def generate_title(self, content: str, max_length: int = 50) -> str:
        """Auto-generate title from content"""
        try:
            # Clean content for title extraction
            clean_content = re.sub(r'[^\w\s]', ' ', content)
            words = clean_content.split()
            
            # Try to find a good title from first sentence
            sentences = content.split('.')
            if sentences and len(sentences[0]) < max_length:
                title = sentences[0].strip()
            else:
                # Use first few words
                title = ' '.join(words[:8])
            
            # Ensure reasonable length
            if len(title) > max_length:
                title = title[:max_length-3] + '...'
            
            return title or f"Note {datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        except Exception:
            return f"Note {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def auto_categorize(self, content: str) -> str:
        """Automatically categorize content based on patterns"""
        try:
            content_lower = content.lower()
            
            # Check each category pattern
            for category, patterns in self.category_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content_lower):
                        return category
            
            # Default category
            return 'general'
            
        except Exception:
            return 'general'
    
    def auto_tag(self, content: str) -> List[str]:
        """Automatically generate tags based on content patterns"""
        try:
            content_lower = content.lower()
            detected_tags = []
            
            # Check each tag pattern
            for tag, patterns in self.tag_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content_lower):
                        detected_tags.append(tag)
                        break  # Avoid duplicate tags
            
            # Limit to 5 tags
            return detected_tags[:5]
            
        except Exception:
            return []
    
    def get_note(self, note_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific note by ID"""
        try:
            note_file = f"{self.notes_dir}/{note_id}.json"
            if os.path.exists(note_file):
                with open(note_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Note retrieval error: {e}")
            return None
    
    def update_note(self, note_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing note"""
        try:
            note = self.get_note(note_id)
            if not note:
                return False
            
            # Update fields
            allowed_updates = ['title', 'content', 'category', 'tags', 'starred', 'archived']
            for field in allowed_updates:
                if field in updates:
                    note[field] = updates[field]
            
            note['updated_at'] = datetime.now().isoformat()
            
            # Recalculate content metrics if content changed
            if 'content' in updates:
                note['word_count'] = len(updates['content'].split())
                note['char_count'] = len(updates['content'])
                note['content_hash'] = hashlib.md5(updates['content'].encode()).hexdigest()
            
            # Save updated note
            note_file = f"{self.notes_dir}/{note_id}.json"
            with open(note_file, 'w') as f:
                json.dump(note, f, indent=2)
            
            # Update metadata
            if note_id in self.metadata['notes']:
                self.metadata['notes'][note_id].update({
                    'title': note['title'],
                    'category': note['category'],
                    'tags': note['tags'],
                    'word_count': note['word_count'],
                    'starred': note['starred'],
                    'archived': note['archived']
                })
                self.save_metadata()
            
            return True
            
        except Exception as e:
            print(f"Note update error: {e}")
            return False
    
    def delete_note(self, note_id: str) -> bool:
        """Delete a note"""
        try:
            note_file = f"{self.notes_dir}/{note_id}.json"
            if os.path.exists(note_file):
                os.remove(note_file)
                
                # Remove from metadata
                if note_id in self.metadata['notes']:
                    del self.metadata['notes'][note_id]
                    self.metadata['total_notes'] -= 1
                    self.save_metadata()
                
                return True
            return False
        except Exception as e:
            print(f"Note deletion error: {e}")
            return False
    
    def search_notes(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search notes by content, title, tags, or category"""
        try:
            results = []
            query_lower = query.lower()
            filters = filters or {}
            
            # Get filter criteria
            category_filter = filters.get('category')
            tag_filter = filters.get('tags', [])
            starred_filter = filters.get('starred')
            archived_filter = filters.get('archived', False)
            date_from = filters.get('date_from')
            date_to = filters.get('date_to')
            
            # Search through all notes
            for note_id, note_meta in self.metadata['notes'].items():
                try:
                    # Skip archived notes unless specifically requested
                    if note_meta.get('archived', False) and not archived_filter:
                        continue
                    
                    # Apply filters
                    if category_filter and note_meta.get('category') != category_filter:
                        continue
                    
                    if tag_filter and not any(tag in note_meta.get('tags', []) for tag in tag_filter):
                        continue
                    
                    if starred_filter is not None and note_meta.get('starred', False) != starred_filter:
                        continue
                    
                    if date_from:
                        note_date = datetime.fromisoformat(note_meta['created_at'])
                        filter_date = datetime.fromisoformat(date_from)
                        if note_date < filter_date:
                            continue
                    
                    if date_to:
                        note_date = datetime.fromisoformat(note_meta['created_at'])
                        filter_date = datetime.fromisoformat(date_to)
                        if note_date > filter_date:
                            continue
                    
                    # Load full note for content search
                    note = self.get_note(note_id)
                    if not note:
                        continue
                    
                    # Check if query matches
                    searchable_text = f"{note['title']} {note['content']} {' '.join(note['tags'])}".lower()
                    
                    if query_lower in searchable_text:
                        # Calculate relevance score
                        score = 0
                        if query_lower in note['title'].lower():
                            score += 10
                        if query_lower in note['content'].lower():
                            score += 5
                        if any(query_lower in tag.lower() for tag in note['tags']):
                            score += 3
                        
                        note['relevance_score'] = score
                        results.append(note)
                
                except Exception as e:
                    print(f"Search error for note {note_id}: {e}")
                    continue
            
            # Sort by relevance score
            results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            return results
            
        except Exception as e:
            print(f"Notes search error: {e}")
            return []
    
    def get_notes_list(self, limit: int = 50, offset: int = 0, 
                      sort_by: str = 'created_at', sort_order: str = 'desc',
                      filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get paginated list of notes with filtering and sorting"""
        try:
            filters = filters or {}
            all_notes = []
            
            # Get all notes with filters
            for note_id, note_meta in self.metadata['notes'].items():
                # Apply filters
                if filters.get('category') and note_meta.get('category') != filters['category']:
                    continue
                
                if filters.get('starred') is not None and note_meta.get('starred', False) != filters['starred']:
                    continue
                
                if filters.get('archived') is not None and note_meta.get('archived', False) != filters['archived']:
                    continue
                
                # Add note metadata with ID
                note_info = note_meta.copy()
                note_info['id'] = note_id
                all_notes.append(note_info)
            
            # Sort notes
            reverse_order = sort_order.lower() == 'desc'
            
            if sort_by == 'created_at':
                all_notes.sort(key=lambda x: x['created_at'], reverse=reverse_order)
            elif sort_by == 'title':
                all_notes.sort(key=lambda x: x['title'].lower(), reverse=reverse_order)
            elif sort_by == 'word_count':
                all_notes.sort(key=lambda x: x.get('word_count', 0), reverse=reverse_order)
            
            # Apply pagination
            total_count = len(all_notes)
            paginated_notes = all_notes[offset:offset + limit]
            
            return {
                'notes': paginated_notes,
                'total_count': total_count,
                'limit': limit,
                'offset': offset,
                'has_more': offset + limit < total_count
            }
            
        except Exception as e:
            print(f"Notes list error: {e}")
            return {'notes': [], 'total_count': 0, 'limit': limit, 'offset': offset, 'has_more': False}
    
    def export_notes(self, format: str = 'json', note_ids: List[str] = None,
                    filters: Dict[str, Any] = None) -> Optional[str]:
        """Export notes in specified format"""
        try:
            # Get notes to export
            if note_ids:
                notes_to_export = [self.get_note(nid) for nid in note_ids if self.get_note(nid)]
            else:
                # Export filtered notes
                if filters:
                    notes_to_export = self.search_notes('', filters)
                else:
                    # Export all notes
                    notes_to_export = [self.get_note(nid) for nid in self.metadata['notes'].keys()]
                    notes_to_export = [note for note in notes_to_export if note]
            
            if not notes_to_export:
                return None
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"notes_export_{timestamp}.{format}"
            export_path = f"{self.notes_dir}/exported/{filename}"
            
            # Export in requested format
            if format == 'json':
                with open(export_path, 'w') as f:
                    json.dump(notes_to_export, f, indent=2)
            
            elif format == 'markdown':
                with open(export_path, 'w') as f:
                    for note in notes_to_export:
                        f.write(f"# {note['title']}\n\n")
                        f.write(f"**Created:** {note['created_at']}\n")
                        f.write(f"**Category:** {note['category']}\n")
                        f.write(f"**Tags:** {', '.join(note['tags'])}\n\n")
                        f.write(f"{note['content']}\n\n")
                        f.write("---\n\n")
            
            elif format == 'txt':
                with open(export_path, 'w') as f:
                    for note in notes_to_export:
                        f.write(f"Title: {note['title']}\n")
                        f.write(f"Created: {note['created_at']}\n")
                        f.write(f"Category: {note['category']}\n")
                        f.write(f"Tags: {', '.join(note['tags'])}\n")
                        f.write(f"Content:\n{note['content']}\n")
                        f.write("\n" + "="*50 + "\n\n")
            
            return export_path
            
        except Exception as e:
            print(f"Notes export error: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notes statistics and analytics"""
        try:
            # Basic stats
            total_notes = self.metadata['total_notes']
            total_words = sum(note.get('word_count', 0) for note in self.metadata['notes'].values())
            
            # Category distribution
            categories = self.metadata.get('categories', {})
            top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Tag distribution
            tags = self.metadata.get('tags', {})
            top_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Recent activity (last 7 days)
            week_ago = datetime.now() - timedelta(days=7)
            recent_notes = 0
            
            for note_meta in self.metadata['notes'].values():
                try:
                    created_at = datetime.fromisoformat(note_meta['created_at'])
                    if created_at > week_ago:
                        recent_notes += 1
                except Exception:
                    continue
            
            # Calculate average note length
            avg_words = total_words / total_notes if total_notes > 0 else 0
            
            return {
                'total_notes': total_notes,
                'total_words': total_words,
                'average_words_per_note': round(avg_words, 1),
                'recent_notes_week': recent_notes,
                'total_categories': len(categories),
                'total_tags': len(tags),
                'top_categories': top_categories,
                'top_tags': top_tags,
                'starred_notes': sum(1 for note in self.metadata['notes'].values() if note.get('starred', False)),
                'archived_notes': sum(1 for note in self.metadata['notes'].values() if note.get('archived', False))
            }
            
        except Exception as e:
            print(f"Statistics generation error: {e}")
            return {'error': str(e)}

# Initialize notes manager
notes_manager = NotesManager()

@notes_bp.route('/create', methods=['POST'])
def create_note():
    """
    Create new note from AI response or user input.
    Completely separate from existing chat/message functionality.
    """
    try:
        data = request.get_json()
        content = data.get('content', '').strip()
        title = data.get('title')
        category = data.get('category')
        tags = data.get('tags', [])
        source_data = data.get('source', {})
        
        if not content:
            return jsonify({
                'success': False,
                'error': 'Content is required'
            }), 400
        
        note_id = notes_manager.create_note(
            content=content,
            title=title,
            category=category,
            tags=tags,
            source_data=source_data
        )
        
        if note_id:
            return jsonify({
                'success': True,
                'note_id': note_id,
                'message': 'Note created successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to create note'
            }), 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@notes_bp.route('/get/<note_id>', methods=['GET'])
def get_note(note_id: str):
    """Get specific note by ID"""
    try:
        note = notes_manager.get_note(note_id)
        
        if note:
            return jsonify({
                'success': True,
                'note': note
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Note not found'
            }), 404
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@notes_bp.route('/list', methods=['GET'])
def list_notes():
    """Get paginated list of notes"""
    try:
        limit = min(int(request.args.get('limit', 50)), 200)
        offset = int(request.args.get('offset', 0))
        sort_by = request.args.get('sort_by', 'created_at')
        sort_order = request.args.get('sort_order', 'desc')
        
        # Parse filters
        filters = {}
        if request.args.get('category'):
            filters['category'] = request.args.get('category')
        if request.args.get('starred'):
            filters['starred'] = request.args.get('starred').lower() == 'true'
        if request.args.get('archived'):
            filters['archived'] = request.args.get('archived').lower() == 'true'
        
        result = notes_manager.get_notes_list(
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order,
            filters=filters
        )
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@notes_bp.route('/search', methods=['POST'])
def search_notes():
    """Search notes by content, title, tags, or category"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        filters = data.get('filters', {})
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Search query is required'
            }), 400
        
        results = notes_manager.search_notes(query, filters)
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results),
            'query': query
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@notes_bp.route('/update/<note_id>', methods=['PUT'])
def update_note(note_id: str):
    """Update existing note"""
    try:
        data = request.get_json()
        
        success = notes_manager.update_note(note_id, data)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Note updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Note not found or update failed'
            }), 404
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@notes_bp.route('/delete/<note_id>', methods=['DELETE'])
def delete_note(note_id: str):
    """Delete note"""
    try:
        success = notes_manager.delete_note(note_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Note deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Note not found'
            }), 404
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@notes_bp.route('/export', methods=['POST'])
def export_notes():
    """Export notes in specified format"""
    try:
        data = request.get_json()
        format = data.get('format', 'json')
        note_ids = data.get('note_ids')
        filters = data.get('filters')
        
        if format not in notes_manager.config['export_formats']:
            return jsonify({
                'success': False,
                'error': f'Unsupported format. Supported: {notes_manager.config["export_formats"]}'
            }), 400
        
        export_path = notes_manager.export_notes(format, note_ids, filters)
        
        if export_path:
            return jsonify({
                'success': True,
                'export_path': export_path,
                'message': f'Notes exported successfully as {format}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Export failed or no notes to export'
            }), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@notes_bp.route('/statistics', methods=['GET'])
def get_statistics():
    """Get notes statistics and analytics"""
    try:
        stats = notes_manager.get_statistics()
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@notes_bp.route('/categories', methods=['GET'])
def get_categories():
    """Get all available categories"""
    try:
        categories = notes_manager.metadata.get('categories', {})
        
        return jsonify({
            'success': True,
            'categories': categories,
            'total_categories': len(categories)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@notes_bp.route('/tags', methods=['GET'])
def get_tags():
    """Get all available tags"""
    try:
        tags = notes_manager.metadata.get('tags', {})
        
        return jsonify({
            'success': True,
            'tags': tags,
            'total_tags': len(tags)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@notes_bp.route('/health', methods=['GET'])
def notes_health():
    """
    Health check for notes management system.
    """
    try:
        stats = notes_manager.get_statistics()
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'total_notes': stats.get('total_notes', 0),
            'system_ready': True,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'error',
            'error': str(e)
        }), 500