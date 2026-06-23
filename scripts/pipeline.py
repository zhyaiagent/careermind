"""Full data pipeline: collect real data -> rebuild KB."""
import os, sys, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def step(name, script):
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    result = subprocess.run(["python", os.path.join(ROOT, script)], cwd=ROOT)
    if result.returncode != 0:
        print(f"  FAILED")
    else:
        print(f"  DONE")

if __name__ == "__main__":
    print("CareerMind Data Pipeline")
    print("=" * 50)

    # Step 1: Collect real JDs (optional - requires network)
    ans = input("\nCollect real JDs from Bing? This opens a browser. (y/n): ")
    if ans.lower() == 'y':
        step("Collect JDs", "scripts/crawl_real_jds.py")

    # Step 2: Build salary DB
    step("Build Salary DB", "scripts/build_salary_db.py")

    # Step 3: Rebuild Knowledge Base
    step("Rebuild KB", "scripts/rebuild_kb.py")

    print("\nPipeline complete! Start: python -m api.main")
