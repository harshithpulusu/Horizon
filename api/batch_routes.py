"""
Batch Operations API Routes
Safe, parallel processing system that doesn't interfere with existing chat functionality.
"""

from flask import Blueprint, request, jsonify
import json
import uuid
import queue
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
import os

# Create blueprint for batch operations (separate from existing routes)
batch_bp = Blueprint('batch', __name__, url_prefix='/api/batch')

class BatchStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BatchOperation:
    """Individual batch operation"""
    
    def __init__(self, operation_id: str, operation_type: str, data: Dict[str, Any]):
        self.id = operation_id
        self.type = operation_type
        self.data = data
        self.status = BatchStatus.PENDING
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.progress = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'type': self.type,
            'data': self.data,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'progress': self.progress
        }

class BatchProcessor:
    """
    Thread-safe batch processing system.
    Runs completely parallel to existing chat functionality.
    """
    
    def __init__(self):
        self.operations = {}  # Store all operations
        self.pending_queue = queue.Queue()
        self.processing = {}  # Currently processing operations
        self.max_concurrent = 3  # Maximum concurrent operations
        self.worker_threads = []
        self.shutdown_flag = threading.Event()
        
        # Start worker threads
        self.start_workers()
        
        # Supported operation types
        self.operation_handlers = {
            'text_generation': self.handle_text_generation,
            'image_generation': self.handle_image_generation,
            'data_analysis': self.handle_data_analysis,
            'file_processing': self.handle_file_processing,
            'api_calls': self.handle_api_calls
        }
    
    def start_workers(self):
        """Start worker threads for batch processing"""
        for i in range(self.max_concurrent):
            worker = threading.Thread(target=self.worker_loop, daemon=True)
            worker.start()
            self.worker_threads.append(worker)
    
    def worker_loop(self):
        """Main worker loop for processing operations"""
        while not self.shutdown_flag.is_set():
            try:
                # Get operation from queue (with timeout)
                operation_id = self.pending_queue.get(timeout=1.0)
                
                if operation_id in self.operations:
                    operation = self.operations[operation_id]
                    self.process_operation(operation)
                
                self.pending_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
    
    def process_operation(self, operation: BatchOperation):
        """Process a single batch operation"""
        try:
            # Mark as processing
            operation.status = BatchStatus.PROCESSING
            operation.started_at = datetime.now()
            self.processing[operation.id] = operation
            
            # Get handler for operation type
            handler = self.operation_handlers.get(operation.type)
            if not handler:
                raise ValueError(f"Unsupported operation type: {operation.type}")
            
            # Process operation
            result = handler(operation)
            
            # Mark as completed
            operation.status = BatchStatus.COMPLETED
            operation.result = result
            operation.completed_at = datetime.now()
            operation.progress = 100
            
        except Exception as e:
            # Mark as failed
            operation.status = BatchStatus.FAILED
            operation.error = str(e)
            operation.completed_at = datetime.now()
            
        finally:
            # Remove from processing
            if operation.id in self.processing:
                del self.processing[operation.id]
    
    def handle_text_generation(self, operation: BatchOperation) -> Dict[str, Any]:
        """Handle text generation batch operation"""
        data = operation.data
        prompts = data.get('prompts', [])
        
        if not prompts:
            raise ValueError("No prompts provided for text generation")
        
        results = []
        for i, prompt in enumerate(prompts):
            # Simulate text generation (replace with actual AI call)
            operation.progress = int((i + 1) / len(prompts) * 90)
            
            # Mock result (in real implementation, call AI engine)
            result = {
                'prompt': prompt,
                'generated_text': f"Generated response for: {prompt[:50]}...",
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
            # Small delay to simulate processing
            time.sleep(0.5)
        
        return {
            'type': 'text_generation',
            'total_prompts': len(prompts),
            'results': results,
            'processing_time': (datetime.now() - operation.started_at).total_seconds()
        }
    
    def handle_image_generation(self, operation: BatchOperation) -> Dict[str, Any]:
        """Handle image generation batch operation"""
        data = operation.data
        prompts = data.get('prompts', [])
        
        if not prompts:
            raise ValueError("No prompts provided for image generation")
        
        results = []
        for i, prompt in enumerate(prompts):
            operation.progress = int((i + 1) / len(prompts) * 90)
            
            # Mock result (in real implementation, call image generator)
            result = {
                'prompt': prompt,
                'image_url': f"/api/generated/image_{uuid.uuid4().hex[:8]}.jpg",
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
            # Simulate longer processing for images
            time.sleep(1.0)
        
        return {
            'type': 'image_generation',
            'total_images': len(prompts),
            'results': results,
            'processing_time': (datetime.now() - operation.started_at).total_seconds()
        }
    
    def handle_data_analysis(self, operation: BatchOperation) -> Dict[str, Any]:
        """Handle data analysis batch operation"""
        data = operation.data
        datasets = data.get('datasets', [])
        
        if not datasets:
            raise ValueError("No datasets provided for analysis")
        
        results = []
        for i, dataset in enumerate(datasets):
            operation.progress = int((i + 1) / len(datasets) * 90)
            
            # Mock analysis result
            result = {
                'dataset': dataset.get('name', f'dataset_{i}'),
                'analysis': {
                    'rows': dataset.get('rows', 100),
                    'columns': dataset.get('columns', 10),
                    'summary': f"Analysis complete for {dataset.get('name', 'unnamed dataset')}"
                },
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
            time.sleep(0.8)
        
        return {
            'type': 'data_analysis',
            'total_datasets': len(datasets),
            'results': results,
            'processing_time': (datetime.now() - operation.started_at).total_seconds()
        }
    
    def handle_file_processing(self, operation: BatchOperation) -> Dict[str, Any]:
        """Handle file processing batch operation"""
        data = operation.data
        files = data.get('files', [])
        
        if not files:
            raise ValueError("No files provided for processing")
        
        results = []
        for i, file_info in enumerate(files):
            operation.progress = int((i + 1) / len(files) * 90)
            
            result = {
                'filename': file_info.get('name', f'file_{i}'),
                'size': file_info.get('size', 0),
                'status': 'processed',
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
            time.sleep(0.3)
        
        return {
            'type': 'file_processing',
            'total_files': len(files),
            'results': results,
            'processing_time': (datetime.now() - operation.started_at).total_seconds()
        }
    
    def handle_api_calls(self, operation: BatchOperation) -> Dict[str, Any]:
        """Handle API calls batch operation"""
        data = operation.data
        api_calls = data.get('calls', [])
        
        if not api_calls:
            raise ValueError("No API calls provided")
        
        results = []
        for i, call_info in enumerate(api_calls):
            operation.progress = int((i + 1) / len(api_calls) * 90)
            
            result = {
                'endpoint': call_info.get('endpoint', '/unknown'),
                'method': call_info.get('method', 'GET'),
                'status': 'success',
                'response': f"Mock response for {call_info.get('endpoint', 'unknown')}",
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
            time.sleep(0.4)
        
        return {
            'type': 'api_calls',
            'total_calls': len(api_calls),
            'results': results,
            'processing_time': (datetime.now() - operation.started_at).total_seconds()
        }
    
    def add_operation(self, operation_type: str, data: Dict[str, Any]) -> str:
        """Add new operation to batch queue"""
        operation_id = str(uuid.uuid4())
        operation = BatchOperation(operation_id, operation_type, data)
        
        self.operations[operation_id] = operation
        self.pending_queue.put(operation_id)
        
        return operation_id
    
    def get_operation(self, operation_id: str) -> Optional[BatchOperation]:
        """Get operation by ID"""
        return self.operations.get(operation_id)
    
    def get_all_operations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all operations (latest first)"""
        sorted_ops = sorted(
            self.operations.values(),
            key=lambda x: x.created_at,
            reverse=True
        )
        return [op.to_dict() for op in sorted_ops[:limit]]
    
    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel operation if not yet processing"""
        if operation_id in self.operations:
            operation = self.operations[operation_id]
            if operation.status == BatchStatus.PENDING:
                operation.status = BatchStatus.CANCELLED
                return True
        return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        status_counts = {}
        for status in BatchStatus:
            status_counts[status.value] = sum(
                1 for op in self.operations.values() 
                if op.status == status
            )
        
        return {
            'queue_size': self.pending_queue.qsize(),
            'processing_count': len(self.processing),
            'total_operations': len(self.operations),
            'status_breakdown': status_counts,
            'worker_threads': len(self.worker_threads)
        }

# Initialize batch processor
batch_processor = BatchProcessor()

@batch_bp.route('/submit', methods=['POST'])
def submit_batch():
    """
    Submit new batch operation.
    Completely independent of existing /chat functionality.
    """
    try:
        data = request.get_json()
        operation_type = data.get('type')
        operation_data = data.get('data', {})
        
        if not operation_type:
            return jsonify({
                'success': False,
                'error': 'Operation type is required'
            }), 400
        
        # Add operation to batch queue
        operation_id = batch_processor.add_operation(operation_type, operation_data)
        
        return jsonify({
            'success': True,
            'operation_id': operation_id,
            'status': 'queued',
            'message': f'Batch operation {operation_id} queued successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@batch_bp.route('/status/<operation_id>', methods=['GET'])
def get_operation_status(operation_id: str):
    """Get status of specific batch operation"""
    try:
        operation = batch_processor.get_operation(operation_id)
        
        if not operation:
            return jsonify({
                'success': False,
                'error': 'Operation not found'
            }), 404
        
        return jsonify({
            'success': True,
            'operation': operation.to_dict()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@batch_bp.route('/list', methods=['GET'])
def list_operations():
    """List all batch operations"""
    try:
        limit = min(int(request.args.get('limit', 50)), 200)
        operations = batch_processor.get_all_operations(limit)
        
        return jsonify({
            'success': True,
            'operations': operations,
            'total': len(operations)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@batch_bp.route('/cancel/<operation_id>', methods=['POST'])
def cancel_operation(operation_id: str):
    """Cancel pending batch operation"""
    try:
        success = batch_processor.cancel_operation(operation_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Operation {operation_id} cancelled successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Operation cannot be cancelled (not pending or not found)'
            }), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@batch_bp.route('/queue-status', methods=['GET'])
def get_queue_status():
    """Get current batch queue status"""
    try:
        status = batch_processor.get_queue_status()
        
        return jsonify({
            'success': True,
            'queue_status': status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@batch_bp.route('/types', methods=['GET'])
def get_operation_types():
    """Get available batch operation types"""
    try:
        operation_types = {
            'text_generation': {
                'description': 'Generate multiple text responses',
                'data_format': {'prompts': ['list of text prompts']}
            },
            'image_generation': {
                'description': 'Generate multiple images',
                'data_format': {'prompts': ['list of image prompts']}
            },
            'data_analysis': {
                'description': 'Analyze multiple datasets',
                'data_format': {'datasets': [{'name': 'dataset name', 'data': 'dataset content'}]}
            },
            'file_processing': {
                'description': 'Process multiple files',
                'data_format': {'files': [{'name': 'filename', 'content': 'file content'}]}
            },
            'api_calls': {
                'description': 'Make multiple API calls',
                'data_format': {'calls': [{'endpoint': '/api/endpoint', 'method': 'GET'}]}
            }
        }
        
        return jsonify({
            'success': True,
            'operation_types': operation_types
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@batch_bp.route('/health', methods=['GET'])
def batch_health():
    """Health check for batch processing system"""
    try:
        status = batch_processor.get_queue_status()
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'queue_status': status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'error',
            'error': str(e)
        }), 500