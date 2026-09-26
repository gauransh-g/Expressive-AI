import urllib.request
import json
import time

sample_text = """Customer Support Transcript: Cloud Infrastructure Billing Discrepancy.
Customer (User_921): Hello, I am noticing an unexpected overage charge of $480 on my Kubernetes cluster invoice for August.
Agent (Sarah M.): Thank you for contacting CloudCorp Billing Support. I understand your concern and I will investigate the invoice line items immediately. Can you confirm your project ID?
Customer: The project ID is prj-prod-us-east-8842. We scaled down our node pool on August 15th, but the billing seems to reflect peak capacity throughout the entire month.
Agent: Looking at the telemetry logs, it appears that while the primary node pool was scaled down from 16 to 4 nodes, an automated auto-scaler triggered extra spot instances during weekend nightly maintenance due to a misconfigured cron task.
Customer: Ah, that explains the discrepancy! Is it possible to adjust the charges since those pods were idle during that maintenance window?
Agent: Since you have maintained an enterprise Tier 1 subscription for over two years with a perfect payment history, I have processed a one-time courtesy credit of $480 directly back to your account. I have also reconfigured your auto-scaler thresholds to prevent unneeded pod spinning.
Customer: That is fantastic customer service, Sarah! Thank you so much for resolving this so swiftly.
Agent: You are very welcome! Have a wonderful day and please reach out if you need anything else."""

# Test pipeline endpoint
payload = json.dumps({
    "text": sample_text,
    "target_max_length": 75,
    "speaker": "default",
    "speed": 1.0
}).encode("utf-8")

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/pipeline",
    data=payload,
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode("utf-8"))

print("=== PIPELINE EXECUTION COMPLETED ===")
print("Summary Words:", res["summary_word_count"])
print("Factual Consistency:", res["factual_consistency_score"])
print("Entity Retention:", res["entity_retention_rate"])
print("Repetition Cleanliness:", res["repetition_cleanliness_score"])
print("Summary Content:\n", res["summary"])
print("\nAudio URL:", res["audio_url"])
