from pathlib import Path
import json
import re
import subprocess
import sys

audit, stamp, commit, detailed, brief_draft = sys.argv[1:]
log=Path(audit).read_text()
data,_=json.JSONDecoder().raw_decode(log[log.index("\n{")+1:])
assert "Verified online tests for 21 required runs." in log
run_name, = (k for k in data["runs"] if k.startswith("handoff-rho-ppo-650-s42"))
ppo=data["runs"][run_name]
assert ppo["logged_iterations"] == ppo["expected_iterations"] == 650
assert ppo["primary_continuations"] == 332800 and ppo["mc_tokens"] == 0
table_path, brief_path = map(Path, (detailed, brief_draft))
full_report=table_path.read_text()
brief=brief_path.read_text()
formatter = Path(__file__).with_name("vp-format-report-20261121.py")
tables = subprocess.check_output([sys.executable, str(formatter), audit], text=True).strip().split("\n\n")
results, group, phases, validation, health, paired = tables
assert len(results.splitlines()) == 35
table_sections=[
    results, phases, validation, health, paired,
]
for table in table_sections:
    header=table.splitlines()[0]
    old=re.escape(header)+r"\n[^\n]*\n(?:\|[^\n]*\n)+"
    assert len(re.findall(old,full_report))==1, header
    full_report=re.sub(old,lambda _:table+"\n",full_report,count=1)

row = next(l for l in results.splitlines() if l.startswith("| PPO 650 |"))
parts=[s.strip() for s in row.strip("|").split("|")]
assert len(parts)==10 and int(parts[2])==ppo["selection_iteration"]
score=f"{ppo['test_pass_at_1'][0] * 100:.2f}%"
archive_states=[v.strip().split("|")[-2].strip() for v in results.splitlines()[14:]]
assert len(archive_states)==21
whole=sum(v.startswith("Full (") for v in archive_states)
pending=21-whole
wall=parts[5]
winner=parts[2]
pair=next(l for l in paired.splitlines() if "VinePPO 650 (dedup) minus PPO 650" in l)
pp=pair.split("|")[2].strip()
restored=ppo["resume_iterations"]
assert restored == sorted(set(restored)) and all(i in ppo["changes_across_resumes"] for i in map(str,restored))
resume_rows = []
for i in restored:
    v=ppo["changes_across_resumes"][str(i)]
    resume_rows.append(f"{i} → {i+1} (reward {v['reward_mean']:+.3f}, value loss {v['critic/value_loss']:+.4f})")
resume_details=", ".join(resume_rows)
measured=ppo["phase_seconds"]
critic_pct=100*measured["critic_update_seconds"]/measured["iteration_seconds"]
checkpoint_pct=100*measured["checkpoint_seconds"]/measured["iteration_seconds"]
gaps="All PPO rollover positions used the expected actor and critic LRs after restore."
full_report=re.sub(r"\| PPO 650, seed 42 \| [^\n]+",lambda _:f"| PPO 650, seed 42 | {len(restored)} restores | {resume_details}. {gaps} |",full_report,count=1)
intro=("All 21 required training and baseline runs have a completion marker, a 16-round held-out result and a verified online Forge run. "
       f"{whole} required runs have whole-lineage bucket verification. The other {pending} retained all required local result files, with incomplete bucket coverage under the Hugging Face storage restriction. "
       "Optional long seeds 43 and 44 were dropped first to protect required storage.")
full_report=re.sub(r"Twenty of the 21 required training[^\n]+",lambda _:intro,full_report,count=1)
for old,new in [
    ("VinePPO reached 56.90% Pass@1 on iteration 550, GRPO 53.63% on iteration 625, and RLOO 51.48% on iteration 525.",
     f"VinePPO reached 56.90% Pass@1 on iteration 550, GRPO 53.63% on iteration 625, RLOO 51.48% on iteration 525 and PPO {score} on iteration {winner}."),
    ("The paired gaps from VinePPO were +3.27 points against GRPO and +5.42 against RLOO.",
     f"The paired gaps from VinePPO were +3.27 points against GRPO, +5.42 against RLOO and {pp} against PPO."),
    ("RLOO took 7:37:08, GRPO 10:33:54, and VinePPO 15:37:05 of allocated H100 time.",
     f"Allocated GPU wall was 7:37:08 for RLOO, 10:33:54 for GRPO, 15:37:05 for VinePPO and {wall} for PPO."),
    ("the three finished RhoMath 650 runs", "all four RhoMath 650 runs"),
    ("PPO's FP32 critic is its dominant training phase.",f"PPO's FP32 critic accounted for {critic_pct:.1f}% of the long run's recorded iteration time, with {checkpoint_pct:.1f}% in checkpointing."),
    ("The three methods with finished 650 runs used the same accelerator class.", "The long-run time comparison uses the actual Slurm allocations in the results table."),
]:
    assert full_report.count(old)==1, old
    full_report=full_report.replace(old,new)
old="Twenty of 21 required 16-round tests are complete and visible in Forge. PPO 650 is the remaining training run."
assert brief.count(old)==1
brief=brief.replace(old,"All 21 required 16-round tests are complete and verified in Forge.")
old="PPO remains active."
assert brief.count(old)==1
brief=brief.replace(old,f"PPO reached {score} at iteration {winner} in {wall} of allocated GPU time.")
old="Thirteen full production mirrors passed SHA256 verification. Two short PPO runs and several long or OLMo runs retain incomplete bucket coverage."
assert brief.count(old)==1
brief=brief.replace(old,f"{whole} full production mirrors passed SHA256 verification. {pending} required lineages have incomplete bucket coverage, including two short PPO runs, both OLMo GRPO runs and the long RhoMath arms.")
for path,doc in ((table_path,full_report),(brief_path,brief)):
    doc,n=re.subn(r"^Draft through .+?$",f"{stamp} EST · `{commit}`",doc,flags=re.M)
    assert n==1 and "remains active" not in doc
    path.write_text(doc)
print(f"21 final test rows, {whole} required full archives, {pending} archive gaps, {len(restored)} successful PPO restores.")
