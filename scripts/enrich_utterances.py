"""
Enrich trajectory utterances with diverse natural language using Llama 3 via Ollama.

Strategy:
1. Enumerate all unique (action, args, role) tuples across the dataset
2. For each tuple, prompt Llama 3 to generate 20 diverse NL utterances
3. Cache the utterance bank to disk
4. Patch all trajectory files by sampling from the bank with contextual variation

Usage:
    python scripts/enrich_utterances.py --phase generate   # Step 1-2: build utterance bank
    python scripts/enrich_utterances.py --phase patch       # Step 3-4: patch trajectory files
    python scripts/enrich_utterances.py --phase all         # Both steps
"""

from __future__ import annotations

import argparse
import collections
import copy
import glob
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

# ─── Config ──────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"
BANK_PATH = "data/utterance_bank.json"
NUM_VARIATIONS = 20
SEED = 42

# Room name mappings for naturalness
ROOM_NAMES = {
    "room_0": ["the living room", "the main hall", "the front room"],
    "room_1": ["the kitchen", "the cooking area", "the kitchen area"],
    "room_2": ["the bedroom", "the master bedroom", "the sleeping quarters"],
    "room_3": ["the study", "the office", "the library"],
    "room_4": ["the pantry", "the storage room", "the supply closet"],
}

OBJECT_NAMES = {
    "red_mug": ["the red mug", "a red coffee mug", "the red cup", "that red mug"],
}


# ─── Ollama API ──────────────────────────────────────────────────────────────

def query_llama(prompt: str, temperature: float = 0.9, max_retries: int = 3) -> str:
    """Query Llama 3 via Ollama REST API."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.95,
            "num_predict": 1024,
        },
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["response"]
        except Exception as e:
            print(f"  [Attempt {attempt+1}/{max_retries}] Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    return ""


def parse_numbered_list(text: str) -> list[str]:
    """Extract numbered items from LLM output."""
    lines = text.strip().split("\n")
    results = []
    for line in lines:
        line = line.strip()
        # Match patterns like "1. ...", "1) ...", "- ...", "* ..."
        m = re.match(r'^(?:\d+[\.\)]\s*|[-*]\s+)(.*)', line)
        if m:
            utterance = m.group(1).strip().strip('"').strip("'").strip()
            if utterance and len(utterance) > 3:
                results.append(utterance)
    return results


# ─── Utterance Generation ────────────────────────────────────────────────────

def build_action_description(action: str, args: str, role: str) -> str:
    """Create a human-readable description of what the action does."""
    descs = {
        "nop": f"The {role} agent is idle / doing nothing / waiting for something to happen",
        "wait": f"The {role} agent decides to wait and observe the situation",
        "look": f"The {role} agent looks around to survey the environment",
        "done": f"The {role} agent signals that they have completed their part of the task",
        "navigate": f"The {role} agent moves to {args} ({', '.join(ROOM_NAMES.get(args, [args])[:2])})",
        "pick_up": f"The {role} agent picks up {args} ({', '.join(OBJECT_NAMES.get(args, [args])[:2])})",
        "put_down": f"The {role} agent puts down / places {args} ({', '.join(OBJECT_NAMES.get(args, [args])[:2])})",
        "report": f"The {role} agent reports their {args} to the team",
        "say": f"The {role} agent communicates their current {args} to the team",
    }
    return descs.get(action, f"The {role} agent performs {action}({args})")


def generate_utterance_bank_for_key(action: str, args: str, role: str) -> list[str]:
    """Generate diverse NL utterances for a single (action, args, role) tuple."""
    desc = build_action_description(action, args, role)
    
    prompt = f"""You are helping create training data for a multi-agent dialogue system where AI agents coordinate on household tasks.

Context: A team of agents (navigator, manipulator, researcher) work together on tasks like finding objects, cleaning rooms, cooking, etc. They communicate naturally while performing actions.

An agent with the role "{role}" is performing this action: {desc}

Generate exactly {NUM_VARIATIONS} diverse, natural dialogue utterances that this agent might say while performing this action. The utterances should:
- Sound like natural spoken dialogue between cooperating teammates
- Vary in style: some brief, some detailed, some questioning, some declarative
- Include filler words, hedging, or personality sometimes (e.g., "Hmm, let me...", "Alright,", "Got it,")
- Reference the task context when appropriate
- NOT include the agent's name/ID prefix (just the utterance itself)
- Be 5-30 words each
- Each be meaningfully different from the others

Format: Return ONLY a numbered list (1. through {NUM_VARIATIONS}.) with one utterance per line.
"""
    
    response = query_llama(prompt, temperature=0.9)
    utterances = parse_numbered_list(response)
    
    # If LLM didn't return enough, do a second pass
    if len(utterances) < NUM_VARIATIONS // 2:
        print(f"  Only got {len(utterances)} utterances, retrying with different prompt...")
        prompt2 = f"""Generate {NUM_VARIATIONS} short dialogue lines (5-25 words each) that a "{role}" teammate might say while: {desc}

These are natural speech utterances for a multi-agent household task system. Vary style and tone.

Return a numbered list 1-{NUM_VARIATIONS}:"""
        response2 = query_llama(prompt2, temperature=1.0)
        extras = parse_numbered_list(response2)
        utterances.extend(extras)
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in utterances:
        u_lower = u.lower().strip(".")
        if u_lower not in seen:
            seen.add(u_lower)
            unique.append(u)
    
    return unique[:NUM_VARIATIONS * 2]  # Keep extras for more variety


def build_utterance_bank(data_dir: str = "data") -> dict:
    """Build the full utterance bank by querying Llama 3 for all action combos."""
    # Collect unique keys
    keys = set()
    for fpath in sorted(glob.glob(f"{data_dir}/*/batch_*.json")):
        data = json.load(open(fpath))
        for traj in data:
            roles = traj["agent_roles"]
            for step in traj["steps"]:
                role = roles[step["agent_id"]]
                keys.add((step["action"], step["args"], role))
    
    print(f"Found {len(keys)} unique (action, args, role) combinations")
    
    # Load existing bank if any (for resumability)
    bank = {}
    bank_path = Path(BANK_PATH)
    if bank_path.exists():
        bank = json.load(open(bank_path))
        print(f"Loaded existing bank with {len(bank)} keys")
    
    # Generate for each key
    keys_sorted = sorted(keys)
    for i, (action, args, role) in enumerate(keys_sorted):
        key_str = f"{action}|{args}|{role}"
        if key_str in bank and len(bank[key_str]) >= NUM_VARIATIONS // 2:
            print(f"[{i+1}/{len(keys_sorted)}] SKIP {key_str} (already have {len(bank[key_str])} utterances)")
            continue
        
        print(f"[{i+1}/{len(keys_sorted)}] Generating for: {key_str}")
        utterances = generate_utterance_bank_for_key(action, args, role)
        bank[key_str] = utterances
        print(f"  → Got {len(utterances)} utterances")
        
        # Save after each key (resumability)
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bank_path, "w") as f:
            json.dump(bank, f, indent=2)
    
    # Print stats
    total = sum(len(v) for v in bank.values())
    print(f"\nUtterance bank complete: {len(bank)} keys, {total} total utterances")
    print(f"Saved to {BANK_PATH}")
    
    return bank


# ─── Trajectory Patching ─────────────────────────────────────────────────────

def contextual_select(
    utterances: list[str],
    step_idx: int,
    total_steps: int,
    task_desc: str,
    rng: random.Random,
) -> str:
    """Select an utterance with some contextual awareness."""
    if not utterances:
        return "..."
    
    # Weight selection slightly toward different styles based on position
    n = len(utterances)
    if n == 1:
        return utterances[0]
    
    # Just pick randomly — the diversity is already in the bank
    return rng.choice(utterances)


def patch_trajectory_files(data_dir: str = "data", backup: bool = True):
    """Patch all trajectory files with natural language utterances from the bank."""
    bank_path = Path(BANK_PATH)
    if not bank_path.exists():
        print(f"ERROR: Utterance bank not found at {BANK_PATH}. Run --phase generate first.")
        sys.exit(1)
    
    bank = json.load(open(bank_path))
    print(f"Loaded utterance bank: {len(bank)} keys")
    
    rng = random.Random(SEED)
    
    stats = collections.Counter()
    
    for fpath in sorted(glob.glob(f"{data_dir}/*/batch_*.json")):
        fpath = Path(fpath)
        print(f"Patching {fpath}...")
        
        # Backup original
        if backup:
            backup_path = fpath.with_suffix(".json.bak")
            if not backup_path.exists():
                with open(fpath) as f:
                    original = f.read()
                with open(backup_path, "w") as f:
                    f.write(original)
        
        data = json.load(open(fpath))
        
        for traj in data:
            roles = traj["agent_roles"]
            total_steps = len(traj["steps"])
            
            for step in traj["steps"]:
                role = roles[step["agent_id"]]
                key_str = f"{step['action']}|{step['args']}|{role}"
                
                utterances = bank.get(key_str, [])
                if utterances:
                    nl_utterance = contextual_select(
                        utterances,
                        step["step"],
                        total_steps,
                        traj["task_description"],
                        rng,
                    )
                    step["utterance"] = nl_utterance
                    stats["patched"] += 1
                else:
                    # Fallback: keep original but strip prefix
                    old = step["utterance"]
                    step["utterance"] = old.split(": ", 1)[-1] if ": " in old else old
                    stats["fallback"] += 1
        
        # Write patched file
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)
    
    print(f"\nPatching complete!")
    print(f"  Patched: {stats['patched']}")
    print(f"  Fallback: {stats['fallback']}")
    print(f"  Total: {stats['patched'] + stats['fallback']}")


# ─── Verification ─────────────────────────────────────────────────────────────

def verify_patch(data_dir: str = "data"):
    """Show sample utterances from patched data to verify quality."""
    files = sorted(glob.glob(f"{data_dir}/*/batch_*.json"))
    if not files:
        print("No data files found.")
        return
    
    rng = random.Random(SEED + 1)
    
    print("=" * 80)
    print("SAMPLE UTTERANCES FROM PATCHED DATA")
    print("=" * 80)
    
    for fpath in files[:3]:  # First 3 files
        data = json.load(open(fpath))
        traj = rng.choice(data)
        print(f"\nFile: {fpath}")
        print(f"Task: {traj['task_description']}")
        print(f"Roles: {traj['agent_roles']}")
        print(f"Steps ({len(traj['steps'])}):")
        for step in traj["steps"][:10]:
            role = traj["agent_roles"][step["agent_id"]]
            print(f"  [{step['step']}] Agent {step['agent_id']} ({role}) "
                  f"[{step['action']}({step['args']})]: \"{step['utterance']}\"")
        print()
    
    # Count unique utterances
    all_utterances = set()
    for fpath in files:
        data = json.load(open(fpath))
        for traj in data:
            for step in traj["steps"]:
                all_utterances.add(step["utterance"])
    
    print(f"Total unique utterances across all data: {len(all_utterances)}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enrich trajectory utterances with LLM-generated NL")
    parser.add_argument("--phase", type=str, choices=["generate", "patch", "verify", "all"],
                       default="all", help="Which phase to run")
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()
    
    if args.phase in ("generate", "all"):
        print("=" * 60)
        print("PHASE 1: Generating utterance bank via Llama 3")
        print("=" * 60)
        build_utterance_bank(args.data_dir)
    
    if args.phase in ("patch", "all"):
        print("\n" + "=" * 60)
        print("PHASE 2: Patching trajectory files")
        print("=" * 60)
        patch_trajectory_files(args.data_dir)
    
    if args.phase in ("verify", "all"):
        print("\n" + "=" * 60)
        print("VERIFICATION: Sample patched utterances")
        print("=" * 60)
        verify_patch(args.data_dir)


if __name__ == "__main__":
    main()
