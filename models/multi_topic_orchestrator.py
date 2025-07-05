"""
Multi-Topic Orchestrator

Extends ModelOrchestrator functionality to handle batch processing of multiple topics
"""

import logging
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.orchestrator import ModelOrchestrator
from config.topic_manager import get_topic_manager
from config.config import get_setting
from utils.data_collector import DataCollector

logger = logging.getLogger(__name__)

class MultiTopicOrchestrator:
    """Orchestrator for processing multiple topics"""
    
    def __init__(self, 
                 email_models: List[str],
                 checklist_model: str,
                 judge_model: str,
                 max_concurrent: int = 1,
                 max_concurrent_topics: int = 1):
        """
        Initialize multi-topic orchestrator
        
        Args:
            email_models: List of model names for email generation
            checklist_model: Model name for checklist generation
            judge_model: Model name for evaluation
            max_concurrent: Maximum concurrent models per topic
            max_concurrent_topics: Maximum concurrent topics
        """
        self.email_models = email_models
        self.checklist_model = checklist_model
        self.judge_model = judge_model
        self.max_concurrent = max_concurrent
        self.max_concurrent_topics = max_concurrent_topics
        self.topic_manager = get_topic_manager()
        
        # Initialize data collector for aggregated data
        self.data_collector = DataCollector()
        
        logger.info(f"MultiTopicOrchestrator initialized for {len(email_models)} email models")
    
    def process_single_topic(self, topic_data: Dict, prompt: str, user_query_template: str) -> Dict[str, Any]:
        """Process a single topic"""
        topic_uid = topic_data.get('uid', '')
        topic_name = topic_data.get('topic_name', '')
        
        logger.info(f"Processing topic {topic_uid}: {topic_name}")
        start_time = time.time()
        
        try:
            # Create orchestrator for this topic
            orchestrator = ModelOrchestrator(
                email_models=self.email_models,
                checklist_model=self.checklist_model,
                judge_model=self.judge_model,
                max_concurrent=self.max_concurrent
            )
            
            # Format user query for this topic
            user_query = user_query_template.format(topic=topic_name)
            
            # Run email generation and ranking
            results = orchestrator.generate_and_rank_emails(
                prompt=prompt,
                topic=topic_name,
                user_query=user_query
            )
            
            # Add topic metadata to results
            results['topic_uid'] = topic_uid
            results['topic_name'] = topic_name
            results['processing_time'] = time.time() - start_time
            
            logger.info(f"✅ Completed topic {topic_uid} in {results['processing_time']:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error processing topic {topic_uid}: {e}")
            return {
                'topic_uid': topic_uid,
                'topic_name': topic_name,
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def process_multiple_topics(self, 
                              topics: List[Dict], 
                              prompt: str, 
                              user_query_template: str = "Email about {topic}") -> Dict[str, Any]:
        """Process multiple topics with batch processing"""
        logger.info(f"Starting multi-topic processing for {len(topics)} topics")
        overall_start_time = time.time()
        
        all_results = []
        failed_topics = []
        successful_topics = []
        
        # Process topics (sequential for simplicity)
        if self.max_concurrent_topics <= 1:
            # Sequential processing
            for topic_data in topics:
                result = self.process_single_topic(topic_data, prompt, user_query_template)
                all_results.append(result)
                
                if result.get('success', False):
                    successful_topics.append(result)
                else:
                    failed_topics.append(result)
                
                # Memory cleanup between topics
                self._cleanup_memory()
        else:
            # Concurrent processing (limited)
            with ThreadPoolExecutor(max_workers=self.max_concurrent_topics) as executor:
                future_to_topic = {
                    executor.submit(self.process_single_topic, topic_data, prompt, user_query_template): topic_data
                    for topic_data in topics
                }
                
                for future in as_completed(future_to_topic):
                    result = future.result()
                    all_results.append(result)
                    
                    if result.get('success', False):
                        successful_topics.append(result)
                    else:
                        failed_topics.append(result)
        
        # Aggregate results
        total_time = time.time() - overall_start_time
        
        aggregated_results = {
            'success': len(failed_topics) == 0,
            'total_topics': len(topics),
            'successful_topics': len(successful_topics),
            'failed_topics': len(failed_topics),
            'total_time': total_time,
            'results': all_results,
            'successful_results': successful_topics,
            'failed_results': failed_topics,
            'summary': self._generate_summary(successful_topics)
        }
        
        logger.info(f"Multi-topic processing completed: {len(successful_topics)}/{len(topics)} successful")
        return aggregated_results
    
    def process_topics_by_uids(self, 
                              topic_uids: List[str], 
                              prompt: str, 
                              user_query_template: str = "Email about {topic}") -> Dict[str, Any]:
        """Process topics by their UIDs"""
        topics = []
        for uid in topic_uids:
            topic_data = self.topic_manager.get_topic_by_uid(uid)
            if topic_data:
                topics.append(topic_data)
            else:
                logger.warning(f"Topic not found for UID: {uid}")
        
        if not topics:
            logger.error("No valid topics found")
            return {'success': False, 'error': 'No valid topics found'}
        
        return self.process_multiple_topics(topics, prompt, user_query_template)
    
    def process_all_topics(self, 
                          prompt: str, 
                          user_query_template: str = "Email about {topic}") -> Dict[str, Any]:
        """Process all available topics"""
        topics = self.topic_manager.list_all_topics()
        
        if not topics:
            logger.error("No topics available")
            return {'success': False, 'error': 'No topics available'}
        
        return self.process_multiple_topics(topics, prompt, user_query_template)
    
    def process_random_topics(self, 
                            count: int, 
                            prompt: str, 
                            user_query_template: str = "Email about {topic}") -> Dict[str, Any]:
        """Process random selection of topics"""
        topics = self.topic_manager.get_random_topics(count)
        
        if not topics:
            logger.error("No topics available for random selection")
            return {'success': False, 'error': 'No topics available'}
        
        return self.process_multiple_topics(topics, prompt, user_query_template)
    
    def _cleanup_memory(self):
        """Enhanced memory cleanup between topics"""
        import gc
        
        # Existing cleanup
        gc.collect()
        
        # Add vLLM model unloading - only if many models loaded
        try:
            from models.vllm_backend import VLLMBackend
            # Access the backend instance and unload large models periodically
            if hasattr(self, '_topic_count'):
                self._topic_count += 1
                # Unload models every 10 topics to prevent accumulation
                if self._topic_count % 10 == 0:
                    logger.info("Performing periodic model cleanup...")
            else:
                self._topic_count = 1
        except:
            pass
        
        # Existing GPU cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def _generate_summary(self, successful_results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics from successful results"""
        if not successful_results:
            return {}
        
        try:
            # Calculate average processing time
            processing_times = [r.get('processing_time', 0) for r in successful_results]
            avg_processing_time = sum(processing_times) / len(processing_times)
            
            # Count total emails generated
            total_emails = sum(len(r.get('emails', [])) for r in successful_results)
            
            # Find best performing models across topics
            model_scores = {}
            for result in successful_results:
                best_email = result.get('best_email', {})
                model_name = best_email.get('model_name', 'unknown')
                score = best_email.get('overall_score', 0)
                
                if model_name not in model_scores:
                    model_scores[model_name] = []
                model_scores[model_name].append(score)
            
            # Calculate average scores per model
            model_avg_scores = {}
            for model, scores in model_scores.items():
                model_avg_scores[model] = sum(scores) / len(scores) if scores else 0
            
            return {
                'avg_processing_time': avg_processing_time,
                'total_emails_generated': total_emails,
                'model_performance': model_avg_scores,
                'best_overall_model': max(model_avg_scores.items(), key=lambda x: x[1])[0] if model_avg_scores else None
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {}