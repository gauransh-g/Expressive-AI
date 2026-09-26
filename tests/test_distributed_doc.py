import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from src.summarization import TextSummarizer
from src.evaluation import Evaluator

sample_distributed_doc = (
    "In January 2026, CloudScale Systems initiated a comprehensive migration of their real-time telemetry processing infrastructure from monolithic servers to a distributed Kubernetes microservices cluster. "
    "The initial system suffered from excessive API response latencies averaging 450ms and recurring service downtime during traffic peaks exceeding 15000 requests per second. "
    "To address these critical bottlenecks, the engineering team engineered an asynchronous pub-sub telemetry pipeline utilizing Apache Kafka partition queuing, Redis cache clustering, and adaptive autoscaling algorithms. "
    "Benchmarking across 50 simulated stress workloads revealed that throughput increased by 310% to 62000 requests per second, while 99th percentile end-to-end latency dropped significantly from 450ms to 42ms. "
    "However, the implementation revealed notable technical limitations, including a 22% increase in cloud hosting memory overhead and intermittent serialization errors when payloads exceeded 15MB. "
    "In conclusion, the engineering team fully resolved the serialization bottleneck by implementing Protocol Buffers v3 with zstandard stream compression, and the cluster officially launched into production on March 15, 2026 with 99.99% operational uptime."
)

def run_distributed_test():
    summarizer = TextSummarizer()
    
    print("=== ORIGINAL DOCUMENT (Word count:", len(sample_distributed_doc.split()), ") ===")
    print(sample_distributed_doc)
    print("\n" + "="*70)
    
    targets = [50, 75, 100, 120, 150]
    for target in targets:
        print(f"\n--- TESTING TARGET LENGTH: {target} words ---")
        summary = summarizer.summarize(sample_distributed_doc, target_max_length=target)
        summary_words = len(summary.split())
        print(f"Generated Summary ({summary_words} words):")
        print(summary)
        
        # Verify content distribution across zones
        has_beginning = any(w in summary.lower() for w in ["cloudscale", "monolithic", "telemetry", "450ms", "january"])
        has_middle = any(w in summary.lower() for w in ["kafka", "redis", "310%", "62000", "42ms", "limitation", "overhead", "15mb"])
        has_conclusion = any(w in summary.lower() for w in ["conclusion", "protocol buffers", "march 15", "99.99%", "uptime", "production", "resolved"])
        
        print(f"-> Checks: [Beginning Preserved: {has_beginning}] | [Middle/Methods/Numbers: {has_middle}] | [Conclusion/Outcome: {has_conclusion}]")
        
        evaluator = Evaluator()
        eval_metrics = evaluator.evaluate_sample(f"target_{target}", sample_distributed_doc, summary)
        print(f"-> Metrics: Consistency: {eval_metrics['factual_consistency_score']*100:.1f}% | Retention: {eval_metrics['entity_retention_rate']*100:.1f}% | Hallucination: {eval_metrics['hallucination_rate']*100:.1f}% | Repetition Clean: {eval_metrics['repetition_cleanliness_score']*100:.1f}%")
        
        assert has_beginning, "Failed to preserve beginning context!"
        assert has_middle, "Failed to preserve middle technical details/metrics!"
        assert has_conclusion, "Failed to preserve final conclusion/resolution!"
        print(f"PASS: Target {target} words successfully represents entire document without error!")

if __name__ == "__main__":
    run_distributed_test()
