from __future__ import annotations
import argparse, json, sys
from .serialization import pack_from_dict

def main() -> int:
    parser = argparse.ArgumentParser(description="Pack rigid cuboids into rectangular containers")
    parser.add_argument("input", nargs="?", help="JSON input file; stdin when omitted")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    with (open(args.input, encoding="utf-8") if args.input else sys.stdin) as source:
        result = pack_from_dict(json.load(source))
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as target: target.write(encoded + "\n")
    else: print(encoded)
    return 0

if __name__ == "__main__": raise SystemExit(main())
