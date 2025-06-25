"""
SGLang RadixAttention Cache Optimizer
Optimizes KV cache reuse patterns across multi-agent workflows
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class CachePrefix:
    """RadixAttention cache prefix information"""
    prefix_hash: str
    prefix_text: str
    usage_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)
    agent_type: str = ""
    topic: str = ""
    estimated_size_mb: float = 0.0

@dataclass
class CacheInteraction:
    """Multi-agent cache interaction pattern"""
    interaction_id: str
    agent_sequence: List[str]  # e.g., ["email", "checklist", "judge"]
    shared_prefixes: List[str]
    cache_hit_rate: float
    total_tokens_saved: int
    execution_time_saved: float

class RadixCacheOptimizer:
    """Optimizes RadixAttention cache usage across multi-agent workflows"""
    
    def __init__(self, max_cache_size_gb: float = 8.0):
        self.max_cache_size_gb = max_cache_size_gb
        self.cache_prefixes: Dict[str, CachePrefix] = {}
        self.cache_interactions: List[CacheInteraction] = []
        self._lock = threading.RLock()
        
        # Optimization strategies
        self.optimization_strategies = {
            "topic_clustering": True,
            "agent_sequence_optimization": True,
            "prefix_preloading": True,
            "cache_warming": True
        }
        
        # Cache statistics
        self.cache_stats = {
            "total_hits": 0,
            "total_misses": 0,
            "bytes_saved": 0,
            "time_saved_seconds": 0.0
        }
        
        logger.info("RadixCacheOptimizer initialized")
    
    def register_cache_prefix(self, 
                            prefix_text: str, 
                            agent_type: str, 
                            topic: str = "",
                            estimated_size_mb: float = 0.0) -> str:
        """Register a new cache prefix for tracking"""
        
        prefix_hash = self._compute_prefix_hash(prefix_text)
        
        with self._lock:
            if prefix_hash in self.cache_prefixes:
                # Update existing prefix
                cached_prefix = self.cache_prefixes[prefix_hash]
                cached_prefix.usage_count += 1
                cached_prefix.last_used = datetime.now()
                
                logger.debug(f"Updated cache prefix: {prefix_hash[:8]} (usage: {cached_prefix.usage_count})")
            else:
                # Create new prefix
                cached_prefix = CachePrefix(
                    prefix_hash=prefix_hash,
                    prefix_text=prefix_text,
                    usage_count=1,
                    agent_type=agent_type,
                    topic=topic,
                    estimated_size_mb=estimated_size_mb
                )
                self.cache_prefixes[prefix_hash] = cached_prefix
                
                logger.debug(f"Registered new cache prefix: {prefix_hash[:8]} for {agent_type}")
        
        return prefix_hash
    
    def track_multi_agent_interaction(self, 
                                    agent_sequence: List[str], 
                                    shared_context: str,
                                    topic: str) -> str:
        """Track cache usage patterns across multi-agent interactions"""
        
        interaction_id = f"{int(time.time())}_{hashlib.md5('_'.join(agent_sequence).encode()).hexdigest()[:8]}"
        
        # Identify shared prefixes
        shared_prefixes = self._identify_shared_prefixes(shared_context, agent_sequence, topic)
        
        # Calculate cache metrics
        cache_hit_rate = self._calculate_cache_hit_rate(shared_prefixes)
        tokens_saved = self._estimate_tokens_saved(shared_prefixes)
        time_saved = self._estimate_time_saved(tokens_saved)
        
        interaction = CacheInteraction(
            interaction_id=interaction_id,
            agent_sequence=agent_sequence,
            shared_prefixes=shared_prefixes,
            cache_hit_rate=cache_hit_rate,
            total_tokens_saved=tokens_saved,
            execution_time_saved=time_saved
        )
        
        with self._lock:
            self.cache_interactions.append(interaction)
            
            # Update global stats
            if cache_hit_rate > 0:
                self.cache_stats["total_hits"] += len(shared_prefixes)
                self.cache_stats["bytes_saved"] += tokens_saved * 4  # Rough bytes estimate
                self.cache_stats["time_saved_seconds"] += time_saved
            else:
                self.cache_stats["total_misses"] += len(agent_sequence)
        
        logger.info(f"Tracked interaction {interaction_id}: {cache_hit_rate:.2f} hit rate, {tokens_saved} tokens saved")
        
        return interaction_id
    
    def optimize_agent_sequence(self, 
                              agent_types: List[str], 
                              topic: str,
                              shared_context: str) -> List[str]:
        """Optimize agent execution sequence for maximum cache reuse"""
        
        if not self.optimization_strategies["agent_sequence_optimization"]:
            return agent_types
        
        logger.info(f"Optimizing agent sequence for topic: {topic}")
        
        # Analyze cache affinity between agents
        agent_affinity = self._calculate_agent_cache_affinity(agent_types, topic)
        
        # Find optimal sequence that maximizes cache reuse
        optimized_sequence = self._find_optimal_sequence(agent_types, agent_affinity, shared_context)
        
        logger.info(f"Optimized sequence: {' -> '.join(optimized_sequence)}")
        
        return optimized_sequence
    
    def warm_cache_for_topic(self, topic: str, agent_types: List[str]) -> Dict[str, Any]:
        """Pre-warm cache with common prefixes for a topic"""
        
        if not self.optimization_strategies["cache_warming"]:
            return {"cache_warming": "disabled"}
        
        logger.info(f"Warming cache for topic: {topic}")
        
        warming_results = {
            "topic": topic,
            "prefixes_warmed": 0,
            "estimated_speedup": 0.0,
            "cache_size_mb": 0.0
        }
        
        with self._lock:
            # Find frequently used prefixes for this topic
            topic_prefixes = [
                prefix for prefix in self.cache_prefixes.values()
                if prefix.topic == topic and prefix.usage_count > 1
            ]
            
            # Sort by usage frequency
            topic_prefixes.sort(key=lambda p: p.usage_count, reverse=True)
            
            # Warm top prefixes
            total_size = 0.0
            warmed_count = 0
            
            for prefix in topic_prefixes:
                if total_size + prefix.estimated_size_mb > self.max_cache_size_gb * 0.8:  # Use 80% of cache
                    break
                
                # Simulate cache warming (in real implementation, this would pre-load into SGLang)
                total_size += prefix.estimated_size_mb
                warmed_count += 1
                
                logger.debug(f"Warmed prefix: {prefix.prefix_hash[:8]} ({prefix.estimated_size_mb:.1f}MB)")
            
            warming_results.update({
                "prefixes_warmed": warmed_count,
                "estimated_speedup": warmed_count * 0.2,  # Rough estimate
                "cache_size_mb": total_size
            })
        
        logger.info(f"Cache warming complete: {warmed_count} prefixes, {total_size:.1f}MB")
        
        return warming_results
    
    def optimize_cache_memory(self) -> Dict[str, Any]:
        """Optimize cache memory usage by evicting least useful prefixes"""
        
        logger.info("Optimizing cache memory usage")
        
        optimization_results = {
            "prefixes_evicted": 0,
            "memory_freed_mb": 0.0,
            "cache_hit_rate_before": self._calculate_global_hit_rate(),
            "cache_hit_rate_after": 0.0
        }
        
        with self._lock:
            # Calculate current cache size
            current_size = sum(p.estimated_size_mb for p in self.cache_prefixes.values())
            
            if current_size <= self.max_cache_size_gb * 1024:  # Convert GB to MB
                optimization_results["cache_hit_rate_after"] = optimization_results["cache_hit_rate_before"]
                return optimization_results
            
            # Sort prefixes by utility score (usage_count / size / age)
            prefixes_by_utility = []
            now = datetime.now()
            
            for prefix in self.cache_prefixes.values():
                age_hours = (now - prefix.last_used).total_seconds() / 3600
                size_mb = max(prefix.estimated_size_mb, 0.1)  # Minimum size
                utility_score = prefix.usage_count / (size_mb * max(age_hours, 1))
                
                prefixes_by_utility.append((utility_score, prefix))
            
            # Sort by utility (lowest first for eviction)
            prefixes_by_utility.sort(key=lambda x: x[0])
            
            # Evict low-utility prefixes
            memory_freed = 0.0
            prefixes_evicted = 0
            target_size = self.max_cache_size_gb * 1024 * 0.8  # Target 80% of max
            
            for utility_score, prefix in prefixes_by_utility:
                if current_size - memory_freed <= target_size:
                    break
                
                memory_freed += prefix.estimated_size_mb
                prefixes_evicted += 1
                
                # Remove from tracking
                del self.cache_prefixes[prefix.prefix_hash]
                
                logger.debug(f"Evicted prefix: {prefix.prefix_hash[:8]} (utility: {utility_score:.4f})")
            
            optimization_results.update({
                "prefixes_evicted": prefixes_evicted,
                "memory_freed_mb": memory_freed,
                "cache_hit_rate_after": self._calculate_global_hit_rate()
            })
        
        logger.info(f"Cache optimization complete: evicted {prefixes_evicted} prefixes, freed {memory_freed:.1f}MB")
        
        return optimization_results
    
    def get_cache_recommendations(self, 
                                topic: str, 
                                agent_sequence: List[str]) -> Dict[str, Any]:
        """Get recommendations for optimizing cache usage"""
        
        recommendations = {
            "topic": topic,
            "agent_sequence": agent_sequence,
            "optimizations": [],
            "estimated_speedup": 0.0,
            "memory_efficiency": 0.0
        }
        
        with self._lock:
            # Analyze current cache state for this topic
            topic_prefixes = [p for p in self.cache_prefixes.values() if p.topic == topic]
            
            if len(topic_prefixes) > 0:
                # Recommend sequence optimization
                if len(set(agent_sequence)) > 1:
                    recommendations["optimizations"].append({
                        "type": "sequence_optimization",
                        "description": "Reorder agents to maximize cache reuse",
                        "potential_speedup": 0.3
                    })
                
                # Recommend cache warming
                high_usage_prefixes = [p for p in topic_prefixes if p.usage_count > 2]
                if high_usage_prefixes:
                    recommendations["optimizations"].append({
                        "type": "cache_warming",
                        "description": f"Pre-warm {len(high_usage_prefixes)} frequently used prefixes",
                        "potential_speedup": 0.2
                    })
                
                # Calculate overall estimates
                recommendations["estimated_speedup"] = sum(
                    opt["potential_speedup"] for opt in recommendations["optimizations"]
                )
                
                # Memory efficiency based on hit rate
                hit_rate = self._calculate_topic_hit_rate(topic)
                recommendations["memory_efficiency"] = hit_rate
        
        return recommendations
    
    def export_cache_analysis(self, output_path: Optional[Path] = None) -> Path:
        """Export detailed cache analysis to file"""
        
        if output_path is None:
            output_path = Path("output/cache_analysis.json")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        analysis = {
            "cache_statistics": self.cache_stats,
            "cache_prefixes": {
                prefix_hash: {
                    "usage_count": prefix.usage_count,
                    "last_used": prefix.last_used.isoformat(),
                    "agent_type": prefix.agent_type,
                    "topic": prefix.topic,
                    "estimated_size_mb": prefix.estimated_size_mb
                }
                for prefix_hash, prefix in self.cache_prefixes.items()
            },
            "cache_interactions": [
                {
                    "interaction_id": interaction.interaction_id,
                    "agent_sequence": interaction.agent_sequence,
                    "shared_prefixes": interaction.shared_prefixes,
                    "cache_hit_rate": interaction.cache_hit_rate,
                    "total_tokens_saved": interaction.total_tokens_saved,
                    "execution_time_saved": interaction.execution_time_saved
                }
                for interaction in self.cache_interactions
            ],
            "optimization_summary": {
                "total_prefixes": len(self.cache_prefixes),
                "total_interactions": len(self.cache_interactions),
                "global_hit_rate": self._calculate_global_hit_rate(),
                "memory_usage_mb": sum(p.estimated_size_mb for p in self.cache_prefixes.values()),
                "total_time_saved": self.cache_stats["time_saved_seconds"]
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Cache analysis exported to: {output_path}")
        
        return output_path
    
    def _compute_prefix_hash(self, prefix_text: str) -> str:
        """Compute hash for cache prefix"""
        return hashlib.sha256(prefix_text.encode()).hexdigest()
    
    def _identify_shared_prefixes(self, 
                                shared_context: str, 
                                agent_sequence: List[str], 
                                topic: str) -> List[str]:
        """Identify prefixes that can be shared across agents"""
        
        shared_prefixes = []
        
        # Common prompt components that can be cached
        common_components = [
            f"Topic: {topic}",
            f"User Query: {shared_context}",
            "Email Generation Context:",
            "Evaluation Context:",
            "Checklist Context:"
        ]
        
        for component in common_components:
            if len(component) > 50:  # Only cache substantial prefixes
                prefix_hash = self.register_cache_prefix(
                    component, 
                    "shared", 
                    topic, 
                    len(component) / 1000  # Rough MB estimate
                )
                shared_prefixes.append(prefix_hash)
        
        return shared_prefixes
    
    def _calculate_cache_hit_rate(self, shared_prefixes: List[str]) -> float:
        """Calculate cache hit rate for given prefixes"""
        
        if not shared_prefixes:
            return 0.0
        
        with self._lock:
            hits = sum(
                1 for prefix_hash in shared_prefixes 
                if prefix_hash in self.cache_prefixes and self.cache_prefixes[prefix_hash].usage_count > 1
            )
            
            return hits / len(shared_prefixes)
    
    def _calculate_global_hit_rate(self) -> float:
        """Calculate global cache hit rate"""
        
        total_requests = self.cache_stats["total_hits"] + self.cache_stats["total_misses"]
        if total_requests == 0:
            return 0.0
        
        return self.cache_stats["total_hits"] / total_requests
    
    def _calculate_topic_hit_rate(self, topic: str) -> float:
        """Calculate cache hit rate for specific topic"""
        
        with self._lock:
            topic_prefixes = [p for p in self.cache_prefixes.values() if p.topic == topic]
            
            if not topic_prefixes:
                return 0.0
            
            total_usage = sum(p.usage_count for p in topic_prefixes)
            cache_hits = sum(p.usage_count - 1 for p in topic_prefixes if p.usage_count > 1)
            
            return cache_hits / total_usage if total_usage > 0 else 0.0
    
    def _estimate_tokens_saved(self, shared_prefixes: List[str]) -> int:
        """Estimate tokens saved through cache reuse"""
        
        with self._lock:
            total_tokens = 0
            
            for prefix_hash in shared_prefixes:
                if prefix_hash in self.cache_prefixes:
                    prefix = self.cache_prefixes[prefix_hash]
                    # Rough estimate: 1MB ≈ 250k tokens
                    tokens = int(prefix.estimated_size_mb * 250000)
                    saved_uses = max(0, prefix.usage_count - 1)
                    total_tokens += tokens * saved_uses
            
            return total_tokens
    
    def _estimate_time_saved(self, tokens_saved: int) -> float:
        """Estimate time saved through cache reuse (in seconds)"""
        
        # Rough estimate: 1000 tokens ≈ 0.1 seconds saved
        return tokens_saved * 0.0001
    
    def _calculate_agent_cache_affinity(self, 
                                      agent_types: List[str], 
                                      topic: str) -> Dict[Tuple[str, str], float]:
        """Calculate cache affinity between agent pairs"""
        
        affinity_matrix = {}
        
        with self._lock:
            for i, agent1 in enumerate(agent_types):
                for j, agent2 in enumerate(agent_types):
                    if i != j:
                        # Calculate shared prefix ratio
                        agent1_prefixes = [
                            p for p in self.cache_prefixes.values() 
                            if p.agent_type == agent1 and p.topic == topic
                        ]
                        agent2_prefixes = [
                            p for p in self.cache_prefixes.values() 
                            if p.agent_type == agent2 and p.topic == topic
                        ]
                        
                        if agent1_prefixes and agent2_prefixes:
                            # Simple affinity calculation based on prefix overlap
                            shared_count = len(set(p.prefix_hash for p in agent1_prefixes) & 
                                             set(p.prefix_hash for p in agent2_prefixes))
                            total_count = len(set(p.prefix_hash for p in agent1_prefixes) | 
                                            set(p.prefix_hash for p in agent2_prefixes))
                            
                            affinity = shared_count / total_count if total_count > 0 else 0.0
                        else:
                            affinity = 0.0
                        
                        affinity_matrix[(agent1, agent2)] = affinity
        
        return affinity_matrix
    
    def _find_optimal_sequence(self, 
                             agent_types: List[str], 
                             affinity_matrix: Dict[Tuple[str, str], float],
                             shared_context: str) -> List[str]:
        """Find optimal agent sequence for cache reuse"""
        
        if len(agent_types) <= 2:
            return agent_types  # No optimization needed for small sequences
        
        # Simple greedy optimization: start with agent that has most cache potential
        remaining_agents = agent_types.copy()
        optimized_sequence = []
        
        # Start with agent that has highest total affinity
        agent_scores = {}
        for agent in remaining_agents:
            total_affinity = sum(
                affinity_matrix.get((agent, other), 0.0) 
                for other in remaining_agents if other != agent
            )
            agent_scores[agent] = total_affinity
        
        # Add first agent (highest score)
        first_agent = max(agent_scores, key=agent_scores.get)
        optimized_sequence.append(first_agent)
        remaining_agents.remove(first_agent)
        
        # Greedily add remaining agents based on affinity with last added
        while remaining_agents:
            last_agent = optimized_sequence[-1]
            
            # Find agent with highest affinity to last agent
            best_agent = max(
                remaining_agents,
                key=lambda a: affinity_matrix.get((last_agent, a), 0.0)
            )
            
            optimized_sequence.append(best_agent)
            remaining_agents.remove(best_agent)
        
        return optimized_sequence

# Global cache optimizer instance
_cache_optimizer = None

def get_cache_optimizer() -> RadixCacheOptimizer:
    """Get global cache optimizer instance"""
    
    global _cache_optimizer
    
    if _cache_optimizer is None:
        _cache_optimizer = RadixCacheOptimizer()
    
    return _cache_optimizer