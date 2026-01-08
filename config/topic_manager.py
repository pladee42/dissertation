import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class TopicManager:
    """Simple topic manager for loading and managing topics"""
    
    def __init__(self, topics_file: str = "config/topics.json"):
        """
        Initialize topic manager
        """
        self.topics_file = topics_file
        self.topics = []
        self._load_topics()
        
        logger.info(f"TopicManager initialized with {len(self.topics)} topics")
    
    def _load_topics(self):
        """Load topics from JSON file"""
        try:
            topics_path = Path(self.topics_file)
            if topics_path.exists():
                with open(topics_path, 'r', encoding='utf-8') as f:
                    self.topics = json.load(f)
                logger.info(f"Loaded {len(self.topics)} topics from {self.topics_file}")
            else:
                logger.warning(f"Topics file not found: {self.topics_file}")
                self.topics = []
                
        except Exception as e:
            logger.error(f"Error loading topics: {e}")
            self.topics = []
    
    def get_topic_by_uid(self, uid: str) -> Optional[Dict]:
        """
        Get topic by UID
        """
        for topic in self.topics:
            if topic.get('uid') == uid:
                return topic
        return None
    
    def get_topic_name_by_uid(self, uid: str) -> Optional[str]:
        """
        Get topic name by UID
        """
        topic = self.get_topic_by_uid(uid)
        return topic.get('topic_name') if topic else None
    
    def list_all_topics(self) -> List[Dict]:
        """
        List all available topics
        """
        return self.topics.copy()
    
    def get_topics_by_range(self, start_uid: str, end_uid: str) -> List[Dict]:
        """
        Get topics within UID range
        """
        try:
            start_num = int(start_uid[1:])  # Remove 'T' prefix
            end_num = int(end_uid[1:])      # Remove 'T' prefix
            
            selected_topics = []
            for topic in self.topics:
                uid = topic.get('uid', '')
                if uid.startswith('T'):
                    topic_num = int(uid[1:])
                    if start_num <= topic_num <= end_num:
                        selected_topics.append(topic)
            
            return selected_topics
            
        except Exception as e:
            logger.error(f"Error getting topics by range: {e}")
            return []
    
    def validate_uid_format(self, uid: str) -> bool:
        """
        Validate UID format (T followed by 4 digits)
        """
        try:
            return (len(uid) == 5 and 
                   uid.startswith('T') and 
                   uid[1:].isdigit())
        except:
            return False
    
    def get_random_topics(self, count: int) -> List[Dict]:
        """
        Get random selection of topics
        """
        import random
        
        if count >= len(self.topics):
            return self.topics.copy()
        
        return random.sample(self.topics, count)

# Global topic manager instance
_topic_manager = None

def get_topic_manager() -> TopicManager:
    """Get global topic manager instance"""
    global _topic_manager
    if _topic_manager is None:
        _topic_manager = TopicManager()
    return _topic_manager