import csv, argparse
from pathlib import Path

SKIP_CONFIDENCES = {"SKIP", "PROVIDER_FAILED"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="rename_manifest.csv")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--auto-approve", action="store_true")
    args = parser.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    actions = []
    skipped = 0
    missing = 0
    conflicts = 0

    for row in rows:
        conf      = row.get("confidence", "")
        approved  = row.get("approved", "").strip().lower()
        filepath  = row.get("filepath", "").strip()
        suggested = row.get("suggested_name", "").strip()
        agent     = row.get("agent", "")

        if conf in SKIP_CONFIDENCES:
            skipped += 1
            continue

        ok = (approved == "yes") or (args.auto_approve and conf in ("HIGH", "MEDIUM"))
        if not ok:
            skipped += 1
            continue

        src = Path(filepath)
        if not src.exists():
            print("  [MISSING] " + str(src))
            missing += 1
            continue

        dst = src.parent / suggested
        if src == dst:
            skipped += 1
            continue

        if dst.exists():
            print("  [CONFLICT] " + dst.name)
            conflicts += 1
            continue

        actions.append((src, dst, agent))

    print("")
    print("Plan: " + str(len(actions)) + " renames, " + str(skipped) + " skipped, " + str(missing) + " missing, " + str(conflicts) + " conflicts")
    print("")

    for src, dst, agent in actions:
        print("  [" + agent + "] " + src.name)
        print("          -> " + dst.name)

    if not args.apply:
        print("\nDry run. Add --apply to rename.")
        return

    done = 0
    for src, dst, agent in actions:
        try:
            src.rename(dst)
            done += 1
        except Exception as e:
            print("  [ERROR] " + src.name + ": " + str(e))

    print("\nDone. " + str(done) + " files renamed.")

main()