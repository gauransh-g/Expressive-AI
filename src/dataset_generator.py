"""
Synthetic & Curated Multi-Domain Dataset Generator for Summarization and Neural TTS Pipelines.
Generates 100 to 5,000+ realistic conversations, news articles, technical documentations, and support transcripts.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils import get_logger, save_json

logger = get_logger("DatasetGenerator")

DOMAINS = [
    "customer_support",
    "technology_news",
    "healthcare",
    "finance",
    "documentation",
    "science",
    "legal",
    "aviation",
    "ecommerce",
    "cybersecurity",
]

TEMPLATES = {
    "customer_support": [
        (
            "Customer Support Transcript: Cloud Infrastructure Billing Issue.\n"
            "Customer (User_921): Hello, I am noticing an unexpected overage charge of $480 on my Kubernetes cluster invoice for August.\n"
            "Agent (Sarah M.): Thank you for contacting CloudCorp Billing Support. I understand your concern and I will investigate the invoice line items immediately. Can you confirm your project ID?\n"
            "Customer: The project ID is prj-prod-us-east-8842. We scaled down our node pool on August 15th, but the billing seems to reflect peak capacity throughout the entire month.\n"
            "Agent: Looking at the telemetry logs, it appears that while the primary node pool was scaled down from 16 to 4 nodes, an automated auto-scaler triggered extra spot instances during weekend nightly maintenance due to a misconfigured cron task.\n"
            "Customer: Ah, that explains the discrepancy! Is it possible to adjust the charges since those pods were idle during that maintenance window?\n"
            "Agent: Since you have maintained an enterprise Tier 1 subscription for over two years with a perfect payment history, I have processed a one-time courtesy credit of $480 directly back to your account. I have also reconfigured your auto-scaler thresholds to prevent unneeded pod spinning.\n"
            "Customer: That is fantastic customer service, Sarah! Thank you so much for resolving this so swiftly.\n"
            "Agent: You are very welcome! Have a wonderful day and please reach out if you need anything else."
        ),
        (
            "Customer Support Interaction: Mobile Banking Application Authentication Failure.\n"
            "Customer: Good morning. Since updating the mobile banking app to version 4.12 yesterday, I cannot log in using biometric authentication or my security PIN.\n"
            "Agent: Hello, thank you for reaching out to Apex Digital Banking. We are aware of an intermittent token synchronization issue with iOS devices following yesterday's release.\n"
            "Customer: I need to transfer funds for my mortgage payment today, so it is quite urgent.\n"
            "Agent: Understood. Let us perform a secure session reset. Please uninstall the current build, clear browser cache, and download the hotfix version 4.12.1 that our engineering team pushed 30 minutes ago.\n"
            "Customer: I just reinstalled version 4.12.1 and FaceID login worked immediately. The transfer was successfully executed.\n"
            "Agent: Excellent! We apologize for the inconvenience and appreciate your patience."
        )
    ],
    "technology_news": [
        (
            "Global AI Breakthrough: Next-Generation Transformer Architectures Achieve 4x Efficiency Gains.\n"
            "Researchers at leading computer vision and natural language institutes have unveiled a novel linear-attention mechanism that dramatically reduces memory complexity from quadratic O(N^2) to linear O(N). "
            "In benchmark tests conducted on sequence lengths exceeding 128,000 tokens, the new architecture achieved a 4.2x speedup in training throughput while reducing GPU memory consumption by 65%. "
            "The model retains full representational fidelity across complex document summarization, low-latency text-to-speech alignment, and multi-modal reasoning tasks. "
            "Industry analysts predict that this advancement will lower the operational barrier for real-time conversational agents, edge devices, and automated speech synthesis pipelines."
        ),
        (
            "Semiconductor Advances: 2nm Gate-All-Around Transistor Yields Reach Commercial Viability.\n"
            "Leading chip foundries have announced that test yields for next-generation 2-nanometer GAAFET microprocessors have crossed the 80% commercial production threshold. "
            "Mass production is scheduled to commence across high-volume fabrication facilities next quarter. "
            "The new node delivers up to a 15% clock frequency increase at identical power consumption, or a 30% reduction in thermal envelope. "
            "Mobile device manufacturers and server providers are already lining up initial silicon allocations for high-performance AI inference accelerators."
        )
    ],
    "healthcare": [
        (
            "Clinical Telehealth Consultation: Post-Operative Knee Rehabilitation Protocol.\n"
            "Patient: Doctor, it has been two weeks since my arthroscopic meniscus repair, and while the swelling has decreased by 70%, I feel slight stiffness when flexing past 90 degrees.\n"
            "Physician: Hello David. The progress you are describing is very typical for the 14-day post-operative timeline. The key is to avoid forceful flexion while gradually rebuilding quadricep engagement.\n"
            "Patient: Should I continue using the cold compression therapy three times a day?\n"
            "Physician: Yes, maintain the 20-minute icing cycles following your physical therapy exercises. We will begin low-resistance stationary cycling next Tuesday to promote synovial fluid circulation.\n"
            "Patient: Perfect, I will adhere strictly to the schedule and see you at the clinic next week."
        ),
        (
            "Cardiovascular Research Summary: High-Sensitivity Troponin Biomarkers in Early Ischemic Detection.\n"
            "Clinical trials spanning 12,000 patients have demonstrated that dual-pathway biomarker assays detecting ultra-low levels of cardiac troponin I and T enable diagnostic accuracy exceeding 97% within one hour of symptom onset. "
            "Rapid stratification protocols reduced unnecessary hospital admissions by 42% while identifying microvascular ischemia before irreversible myocardial necrosis occurred."
        )
    ],
    "finance": [
        (
            "Quarterly Earnings Summary: FinTech Global Exceeds Wall Street Forecasts.\n"
            "FinTech Global today reported fiscal third-quarter revenue of $3.45 billion, representing a 22% year-over-year growth that surpassed consensus estimates by $180 million. "
            "Net operating margins expanded by 340 basis points to 28.5%, fueled by rapid adoption of automated merchant settlement platforms and enterprise AI fraud detection engines. "
            "Free cash flow generation totaled $820 million, allowing the board of directors to authorize an expanded $500 million share repurchase program."
        ),
        (
            "Global Macroeconomic Analysis: Central Bank Liquidity Policies and Yield Curve Dynamics.\n"
            "Sovereign bond markets experienced renewed steepening as monetary policy committees signaled balanced inflation trajectories and adjusted target discount rates. "
            "Institutional portfolio managers shifted allocations toward short-duration treasuries and investment-grade corporate credit to hedge against intermediate liquidity tightening."
        )
    ],
    "documentation": [
        (
            "Software Architecture Documentation: Distributed Microservices Event Bus Pattern.\n"
            "The architecture utilizes Apache Kafka as a persistent distributed event stream connecting the ingestion gateway, payment processor, and notification dispatcher. "
            "Each message payload is serialized in Apache Avro with strict schema registry validation. "
            "Consumer groups scale independently across Kubernetes pods using custom Prometheus queue lag metrics. "
            "In the event of network partition or database downtime, an exponential backoff dead-letter queue (DLQ) buffers unprocessable events with a 72-hour retention guarantee."
        )
    ],
    "science": [
        (
            "Astronomical Discovery: James Webb Space Telescope Detects Atmospheric Water Vapor on Exoplanet.\n"
            "Spectroscopic observations from the James Webb Space Telescope have revealed unambiguous signatures of atmospheric water vapor and carbon dioxide on a sub-Neptune exoplanet orbiting within the habitable zone of a nearby red dwarf star. "
            "Transmission spectroscopy collected during planetary transit confirms the presence of reflective clouds at high altitudes, providing crucial insights into planetary formation and chemical evolution in distant solar systems."
        )
    ],
    "cybersecurity": [
        (
            "Incident Response Briefing: Zero-Day Privilege Escalation Mitigation.\n"
            "The Security Operations Center identified an anomalous RPC execution pattern targeting domain controllers across internal subnet ranges. "
            "Incident responders isolated compromised service accounts, revoked Kerberos ticket granting tokens, and deployed emergency memory-integrity rules. "
            "Forensic analysis confirmed that sensitive customer data stores remained unbreached due to end-to-end envelope encryption and hardware security module key isolation."
        )
    ],
    "legal": [
        (
            "Contract Analysis Summary: Mutual Non-Disclosure and IP Rights Covenants.\n"
            "The covenant stipulates that proprietary source code, algorithmic weights, and commercial trade secrets disclosed during technical evaluations remain protected under strict confidential obligations for five years. "
            "Neither party acquires ownership or licensing rights to pre-existing background intellectual property, and arbitration jurisdiction is designated to the State of Delaware."
        )
    ],
    "aviation": [
        (
            "Flight Operations Safety Audit: Automated Flight Control and Weather Routing Systems.\n"
            "The aviation safety review examined next-generation predictive turbulence radar integration across commercial twin-engine widebody aircraft. "
            "Real-time downlink telemetry with oceanic air traffic centers permitted automated waypoint diversions around convective thunderstorm clusters, reducing severe turbulence encounters by 84%."
        )
    ],
    "ecommerce": [
        (
            "Supply Chain Logistics Overview: Automated Fulfillment Center Robotics.\n"
            "Autonomous mobile sorting robots and computer vision barcode scanners processed over 1.4 million package units during peak holiday sales volume with a 99.98% routing accuracy rate. "
            "Predictive inventory stocking algorithms positioned high-velocity SKU items adjacent to primary packing conveyor belts, cutting median dock-to-delivery fulfillment times from 48 to 18 hours."
        )
    ],
}


def generate_sample_conversations(
    num_samples: int = 5000,
    output_path: str = "data/raw/sample_conversations.json",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generates a dataset of structured multi-domain conversations and articles for testing and benchmarking.
    """
    random.seed(seed)
    samples = []

    domain_pool = list(TEMPLATES.keys())

    for idx in range(1, num_samples + 1):
        domain = random.choice(domain_pool)
        template_list = TEMPLATES[domain]
        base_text = random.choice(template_list)

        sample_id = f"conv_{idx:05d}"
        variation = (
            f" [Reference Case #{idx:05d}. Recorded context date: 2026-Q{random.randint(1,4)}. "
            f"Classification: {domain.upper()}]."
        )
        full_text = base_text + variation

        words = full_text.split()
        sample_obj = {
            "id": sample_id,
            "domain": domain,
            "text": full_text,
            "length": len(words),
            "char_length": len(full_text),
        }
        samples.append(sample_obj)

    save_json(samples, output_path)
    logger.info(f"Generated {len(samples)} samples in {output_path}")
    return samples


if __name__ == "__main__":
    generate_sample_conversations(5000)
