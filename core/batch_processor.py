"""
Batch Processor Core
Advanced batch processing system with queue management and parallel execution.
"""

import threading
import queue
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import json
import uuid
import logging

class BatchPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class BatchTask:
    """Enhanced batch task with priority and dependencies"""
    id: str
    type: str
    data: Dict[str, Any]
    priority: BatchPriority = BatchPriority.NORMAL
    dependencies: List[str] = None
    max_retries: int = 3
    retry_count: int = 0
    created_at: datetime = None
    scheduled_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.dependencies is None:
            self.dependencies = []

class EnhancedBatchProcessor:
    """
    Enhanced batch processor with advanced features:
    - Priority queue management
    - Dependency resolution
    - Retry mechanism
    - Resource limiting
    - Progress tracking
    """
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.workers = []
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.task_results = {}
        self.shutdown_event = threading.Event()
        
        # Resource management
        self.resource_limits = {
            'memory_mb': 1024,  # 1GB memory limit
            'cpu_percent': 80,   # 80% CPU usage limit
            'concurrent_tasks': max_workers
        }
        
        # Task handlers
        self.task_handlers = {}
        self.progress_callbacks = {}
        
        # Statistics
        self.stats = {
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_retried': 0,
            'average_processing_time': 0
        }
        
        # Start worker threads
        self.start_workers()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start_workers(self):
        """Start worker threads for processing tasks"""
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self.worker_loop,
                name=f"BatchWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"Started {self.max_workers} batch processing workers")
    
    def worker_loop(self):
        """Main worker loop for processing tasks"""
        while not self.shutdown_event.is_set():
            try:
                # Get task from priority queue (blocking with timeout)
                priority_item = self.task_queue.get(timeout=1.0)
                
                if priority_item:
                    # Extract task from priority wrapper
                    priority, timestamp, task_id = priority_item
                    
                    if task_id in self.active_tasks:
                        task = self.active_tasks[task_id]
                        self.process_task(task)
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
    
    def process_task(self, task: BatchTask):
        """Process a single batch task with error handling and retries"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Processing task {task.id} of type {task.type}")
            
            # Check dependencies
            if not self.check_dependencies(task):
                self.reschedule_task(task, "Dependencies not met")
                return
            
            # Get task handler
            handler = self.task_handlers.get(task.type)
            if not handler:
                raise ValueError(f"No handler registered for task type: {task.type}")
            
            # Execute task
            result = handler(task)
            
            # Mark as completed
            self.complete_task(task, result, start_time)
            
        except Exception as e:
            self.handle_task_error(task, e, start_time)
    
    def check_dependencies(self, task: BatchTask) -> bool:
        """Check if all task dependencies are completed"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def complete_task(self, task: BatchTask, result: Any, start_time: datetime):
        """Mark task as completed and update statistics"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Move to completed
        self.completed_tasks[task.id] = task
        self.task_results[task.id] = {
            'result': result,
            'processing_time': processing_time,
            'completed_at': datetime.now()
        }
        
        # Remove from active
        if task.id in self.active_tasks:
            del self.active_tasks[task.id]
        
        # Update statistics
        self.stats['total_completed'] += 1
        self.update_average_processing_time(processing_time)
        
        # Call progress callback if registered
        if task.id in self.progress_callbacks:
            try:
                self.progress_callbacks[task.id]('completed', result)
            except Exception as e:
                self.logger.error(f"Progress callback error: {e}")
        
        self.logger.info(f"Task {task.id} completed in {processing_time:.2f}s")
    
    def handle_task_error(self, task: BatchTask, error: Exception, start_time: datetime):
        """Handle task execution error with retry logic"""
        self.logger.error(f"Task {task.id} failed: {error}")
        
        task.retry_count += 1
        
        if task.retry_count <= task.max_retries:
            # Retry task
            self.stats['total_retried'] += 1
            self.logger.info(f"Retrying task {task.id} (attempt {task.retry_count})")
            
            # Add delay before retry (exponential backoff)
            delay = min(2 ** task.retry_count, 60)  # Max 60 seconds
            task.scheduled_at = datetime.now() + timedelta(seconds=delay)
            
            # Re-queue task
            self.reschedule_task(task, f"Retry after error: {error}")
            
        else:
            # Max retries exceeded, mark as failed
            self.fail_task(task, error, start_time)
    
    def fail_task(self, task: BatchTask, error: Exception, start_time: datetime):
        """Mark task as permanently failed"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Move to failed
        self.failed_tasks[task.id] = task
        self.task_results[task.id] = {
            'error': str(error),
            'processing_time': processing_time,
            'failed_at': datetime.now(),
            'retry_count': task.retry_count
        }
        
        # Remove from active
        if task.id in self.active_tasks:
            del self.active_tasks[task.id]
        
        # Update statistics
        self.stats['total_failed'] += 1
        
        # Call progress callback if registered
        if task.id in self.progress_callbacks:
            try:
                self.progress_callbacks[task.id]('failed', str(error))
            except Exception as e:
                self.logger.error(f"Progress callback error: {e}")
        
        self.logger.error(f"Task {task.id} permanently failed after {task.retry_count} retries")
    
    def reschedule_task(self, task: BatchTask, reason: str):
        """Reschedule task for later execution"""
        self.logger.info(f"Rescheduling task {task.id}: {reason}")
        
        # Calculate priority for queue (negative for min-heap behavior)
        priority = -task.priority.value
        timestamp = time.time()
        
        # Add to queue
        self.task_queue.put((priority, timestamp, task.id))
    
    def register_handler(self, task_type: str, handler: Callable):
        """Register a handler for a specific task type"""
        self.task_handlers[task_type] = handler
        self.logger.info(f"Registered handler for task type: {task_type}")
    
    def submit_task(self, task_type: str, data: Dict[str, Any], 
                   priority: BatchPriority = BatchPriority.NORMAL,
                   dependencies: List[str] = None,
                   progress_callback: Callable = None) -> str:
        """Submit a new task to the batch processor"""
        
        task_id = str(uuid.uuid4())
        task = BatchTask(
            id=task_id,
            type=task_type,
            data=data,
            priority=priority,
            dependencies=dependencies or []
        )
        
        # Store task
        self.active_tasks[task_id] = task
        
        # Register progress callback if provided
        if progress_callback:
            self.progress_callbacks[task_id] = progress_callback
        
        # Add to queue
        priority_value = -priority.value  # Negative for min-heap
        timestamp = time.time()
        self.task_queue.put((priority_value, timestamp, task_id))
        
        # Update statistics
        self.stats['total_submitted'] += 1
        
        self.logger.info(f"Submitted task {task_id} of type {task_type}")
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get detailed status of a specific task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                'id': task_id,
                'status': 'active',
                'type': task.type,
                'priority': task.priority.name,
                'created_at': task.created_at.isoformat(),
                'retry_count': task.retry_count,
                'dependencies': task.dependencies
            }
        
        elif task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            result_info = self.task_results.get(task_id, {})
            return {
                'id': task_id,
                'status': 'completed',
                'type': task.type,
                'priority': task.priority.name,
                'created_at': task.created_at.isoformat(),
                'completed_at': result_info.get('completed_at', '').isoformat() if result_info.get('completed_at') else None,
                'processing_time': result_info.get('processing_time'),
                'result': result_info.get('result')
            }
        
        elif task_id in self.failed_tasks:
            task = self.failed_tasks[task_id]
            result_info = self.task_results.get(task_id, {})
            return {
                'id': task_id,
                'status': 'failed',
                'type': task.type,
                'priority': task.priority.name,
                'created_at': task.created_at.isoformat(),
                'failed_at': result_info.get('failed_at', '').isoformat() if result_info.get('failed_at') else None,
                'processing_time': result_info.get('processing_time'),
                'error': result_info.get('error'),
                'retry_count': result_info.get('retry_count', 0)
            }
        
        else:
            return {
                'id': task_id,
                'status': 'not_found'
            }
    
    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics"""
        return {
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'worker_count': len(self.workers),
            'statistics': self.stats.copy(),
            'resource_limits': self.resource_limits.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def update_average_processing_time(self, new_time: float):
        """Update running average of processing time"""
        current_avg = self.stats['average_processing_time']
        completed_count = self.stats['total_completed']
        
        if completed_count == 1:
            self.stats['average_processing_time'] = new_time
        else:
            # Running average calculation
            self.stats['average_processing_time'] = (
                (current_avg * (completed_count - 1) + new_time) / completed_count
            )
    
    def shutdown(self, timeout: float = 30.0):
        """Graceful shutdown of batch processor"""
        self.logger.info("Shutting down batch processor...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
        
        self.logger.info("Batch processor shutdown complete")